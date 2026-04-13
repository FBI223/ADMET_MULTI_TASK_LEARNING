import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

PRIMARY_FEATURES = [
    'dipole', 'homo_lumo', 'electrons', 'energy',
    'qed', 'MolWt', 'TPSA', 'MolLogP', 'MolMR', 'FractionCSP3',
    'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'HeavyAtomCount', 'NumValenceElectrons',
    'NumAromaticRings', 'NumSaturatedRings', 'RingCount',
    'fr_Al_OH', 'fr_Ar_OH', 'fr_Ar_N', 'fr_Ar_NH', 'fr_COO', 'fr_C_O', 'fr_NH0', 'fr_NH1', 'fr_NH2',
    'fr_benzene', 'fr_ester', 'fr_ether', 'fr_halogen', 'fr_ketone', 'fr_nitro', 'fr_phenol',
    'fr_piperdine', 'fr_pyridine', 'fr_sulfonamd', 'fr_urea'
]

# =====================================================================
# 1. PRZYGOTOWANIE DANYCH GRAFOWYCH
# =====================================================================

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    # Cechy atomów (Atomy -> Węzły)
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().numerator,
            float(atom.GetIsAromatic())
        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # Wiązania (Krawędzie)
    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([start, end])
        edge_indices.append([end, start])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return x, edge_index


class MTLHybridDataset(Dataset):
    def __init__(self, X_tab, y, mask, smiles_list):
        self.X_tab = torch.tensor(X_tab, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.mask = torch.tensor(mask, dtype=torch.float)
        self.smiles_list = smiles_list

    def __len__(self): return len(self.X_tab)

    def __getitem__(self, idx):
        # Generujemy graf w locie (można też zapreprocesować dla szybkości)
        x_atom, edge_index = smiles_to_graph(self.smiles_list[idx])

        # Tworzymy obiekt Data z PyG
        graph_data = Data(x=x_atom, edge_index=edge_index)

        return graph_data, self.X_tab[idx], self.y[idx], self.mask[idx]


def collate_hybrid(data_list):
    """Specjalna funkcja do łączenia batchy grafowo-tabelarycznych."""
    graphs, tabs, targets, masks = zip(*data_list)
    batch_graph = Batch.from_data_list(graphs)
    return batch_graph, torch.stack(tabs), torch.stack(targets), torch.stack(masks)


# =====================================================================
# 2. ARCHITEKTURA MODELU (GNN + MLP)
# =====================================================================

class MTLHybridNet(nn.Module):
    def __init__(self, tab_dim, n_tasks, gnn_hidden=64, head_neurons=256):
        super().__init__()

        # --- GAŁĄŹ GRAFOWA (GCN) ---
        self.conv1 = GCNConv(5, gnn_hidden)  # 5 cech z smiles_to_graph
        self.conv2 = GCNConv(gnn_hidden, gnn_hidden)
        self.conv3 = GCNConv(gnn_hidden, gnn_hidden)
        self.bn_gnn = nn.BatchNorm1d(gnn_hidden)

        # --- GAŁĄŹ TABELARYCZNA (Twoje deskryptory + FP) ---
        self.tab_backbone = nn.Sequential(
            nn.Linear(tab_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # --- POŁĄCZENIE (FUSION) I GŁOWICE ---
        combined_dim = gnn_hidden + 256

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, head_neurons),
            nn.BatchNorm1d(head_neurons),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.heads = nn.ModuleList([nn.Linear(head_neurons, 1) for _ in range(n_tasks)])

    def forward(self, graph_data, x_tab):
        # 1. Grafy
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Global Pooling (Mean) -> jeden wektor na cząsteczkę
        x_g = global_mean_pool(x, batch)
        x_g = self.bn_gnn(x_g)

        # 2. Tabelaryczne
        x_t = self.tab_backbone(x_tab)

        # 3. Fusion
        combined = torch.cat([x_g, x_t], dim=1)
        shared_rep = self.shared(combined)

        logits = torch.cat([head(shared_rep) for head in self.heads], dim=1)
        return logits


# =====================================================================
# 3. UTILS & DATA PREP
# =====================================================================

def prepare_mtl_hybrid_data(df, n_bits=1024):
    tasks = sorted(df['task'].dropna().unique().tolist())
    pivot_df = df.pivot_table(index='smiles', columns='task', values='label', aggfunc='first')

    # Usuwamy duplikaty smiles zachowując resztę danych
    merged_df = df.drop_duplicates(subset=['smiles']).set_index('smiles').join(pivot_df, how='inner', rsuffix='_task')

    # 1. Fingerprinty
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = np.array([
        gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.zeros(n_bits)
        for s in merged_df.index
    ]).astype(np.float32)

    # 2. Deskryptory RDKit (PRIMARY_FEATURES)
    valid_features = [f for f in PRIMARY_FEATURES if f in merged_df.columns]
    scaler = StandardScaler()
    rdkit_scaled = scaler.fit_transform(merged_df[valid_features].fillna(0)).astype(np.float32)

    X_tab = np.hstack([rdkit_scaled, fps])
    y_raw = merged_df[tasks].values.astype(np.float32)
    mask = (~np.isnan(y_raw)).astype(np.float32)
    y = np.nan_to_num(y_raw)

    smiles_list = merged_df.index.tolist()
    splits = merged_df['split'].values

    return X_tab, y, mask, tasks, splits, smiles_list


def train_hybrid_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    for batch_g, x_t, y, m in loader:
        batch_g, x_t, y, m = batch_g.to(device), x_t.to(device), y.to(device), m.to(device)

        opt.zero_grad()
        logits = model(batch_g, x_t)

        # Masked Loss
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        loss = (loss * m).sum() / (m.sum() + 1e-8)

        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def plot_hybrid_mtl_training(history, tasks):
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Loss
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
    ax1.set_title('Hybrid MTL Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Avg AUC
    ax2.plot(epochs_range, history['val_auc_avg'], 'g-', label='Avg Val AUC', linewidth=3)
    ax2.set_title('Mean Validation AUC (GNN + MLP)', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.show()

    # 3. Per Task AUC Grid
    cols = 4
    rows = int(np.ceil(len(tasks) / cols))
    fig2, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    axes = axes.flatten()

    for i, t in enumerate(tasks):
        axes[i].plot(epochs_range, history['val_aucs_per_task'][t], label=t, color='darkorange')
        axes[i].set_title(f"{t}", fontsize=10)
        axes[i].set_ylim(0.5, 1.0)
        axes[i].grid(True, alpha=0.2)
        if i % cols == 0: axes[i].set_ylabel('AUC')

    for j in range(len(tasks), len(axes)): fig2.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


