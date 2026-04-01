import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. ZARZĄDZANIE DANYCH (GWARANCJA SPÓJNOŚCI)
# ==========================================

def fix_data_leakage(df):
    """Gwarantuje, że jeden SMILES ma przypisany tylko jeden split."""
    print(">>> Naprawiam spójność splitów na poziomie SMILES...")
    smiles_to_split = df.groupby('smiles')['split'].first().to_dict()
    df['split'] = df['smiles'].map(smiles_to_split)
    return df


class ADMETDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)
        self.task_idx = torch.tensor(df['task_idx'].values, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.task_idx[idx]


class ADMETDataManager:
    def __init__(self, file_path):
        print(f"Ładowanie danych: {file_path}")
        self.df = pd.read_parquet(file_path)
        self.df = fix_data_leakage(self.df)

        # Definicja kolumn
        self.emb_cols = [f'fp_{i}' for i in range(300)]
        self.qc_cols = ['dipole', 'homo_lumo', 'electrons', 'energy']
        self.mask_cols = ['mask_dipole', 'mask_homo_lumo', 'mask_electrons', 'mask_energy']
        metadata = ['smiles', 'task', 'label', 'split', 'success']
        self.rdkit_cols = [c for c in self.df.columns if
                           c not in (self.emb_cols + self.qc_cols + self.mask_cols + metadata)]

        self.tasks = sorted(self.df['task'].unique())
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.df['task_idx'] = self.df['task'].map(self.task_to_idx)
        self.feature_cols = self.emb_cols + self.rdkit_cols + self.qc_cols + self.mask_cols

    def preprocess(self):
        print("Preprocessing: Skalowanie cech...")
        train_mask = self.df['split'] == 'train'
        scaler = StandardScaler()
        scaler.fit(self.df.loc[train_mask, self.feature_cols])
        self.df[self.feature_cols] = scaler.transform(self.df[self.feature_cols])
        return self.df


# ==========================================
# 2. MODELE (IMPROVED MTL & STL)
# ==========================================

class ImprovedMTLModel(nn.Module):
    """Shared Bottom + Deep Task-Specific Heads."""

    def __init__(self, input_dim, num_tasks):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x, task_indices):
        z = self.encoder(x)
        all_logits = torch.stack([head(z).squeeze(-1) for head in self.heads], dim=1)
        idx = torch.arange(x.size(0), device=x.device)
        return all_logits[idx, task_indices]


class STLModel(nn.Module):
    """Independent Task Network (Baseline)."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x): return self.net(x).squeeze(-1)


# ==========================================
# 3. STRATA I TRENING
# ==========================================

class RobustQWMTLLoss(nn.Module):
    def __init__(self, num_tasks, task_freq):
        super().__init__()
        self.log_betas = nn.Parameter(torch.zeros(num_tasks))
        self.register_buffer("task_freq", task_freq)

    def forward(self, logits, targets, task_indices):
        betas = F.softplus(self.log_betas)
        weights = torch.pow(self.task_freq + 1e-6, betas)
        raw_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        total_loss = 0
        for t_id in range(len(self.log_betas)):
            mask = (task_indices == t_id)
            if mask.any():
                total_loss += weights[t_id] * raw_loss[mask].mean()

        return total_loss + 0.01 * torch.sum(self.log_betas ** 2)


def run_mtl_training(loaders, dm, device, task_freq):
    print("\n>>> Start MTL Training...")
    model = ImprovedMTLModel(len(dm.feature_cols), len(dm.tasks)).to(device)
    criterion = RobustQWMTLLoss(len(dm.tasks), task_freq).to(device)
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 0.0001},
        {'params': model.heads.parameters(), 'lr': 0.001},
        {'params': criterion.parameters(), 'lr': 0.001}
    ])

    best_val_auc = 0
    for epoch in range(20):
        model.train()
        for x, y, t in loaders['train']:
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad();
            criterion(model(x, t), y, t).backward();
            optimizer.step()

        # Walidacja
        model.eval()
        p, targets, t_idx = [], [], []
        with torch.no_grad():
            for x, y, t in loaders['val']:
                p.extend(torch.sigmoid(model(x.to(device), t.to(device))).cpu().numpy())
                targets.extend(y.numpy());
                t_idx.extend(t.numpy())

        current_auc = roc_auc_score(targets, p)
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            torch.save(model.state_dict(), 'best_mtl.pth')
            print(f"Epoch {epoch:02d} | New Best Val AUC: {current_auc:.4f}")

    model.load_state_dict(torch.load('best_mtl.pth'))
    return model


def run_stl_baseline(df, tasks, feature_cols, device):
    print("\n>>> Start STL Training (Baseline)...")
    results = {}
    for task in tasks:
        task_df = df[df['task'] == task]
        train_loader = DataLoader(ADMETDataset(task_df[task_df['split'] == 'train'], feature_cols), batch_size=128,
                                  shuffle=True)
        test_loader = DataLoader(ADMETDataset(task_df[task_df['split'] == 'test'], feature_cols), batch_size=128)

        model = STLModel(len(feature_cols)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(10):
            model.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad();
                criterion(model(x), y).backward();
                optimizer.step()

        model.eval()
        p, t = [], []
        with torch.no_grad():
            for x, y, _ in test_loader:
                p.extend(torch.sigmoid(model(x.to(device))).cpu().numpy())
                t.extend(y.numpy())
        results[task] = roc_auc_score(t, p)
        print(f"  {task:<35} | AUC: {results[task]:.4f}")
    return results


# ==========================================
# 4. WIZUALIZACJA I URUCHOMIENIE
# ==========================================

def plot_results(mtl_res, stl_res):
    tasks = list(mtl_res.keys())
    gains = [mtl_res[t] - stl_res[t] for t in tasks]

    plt.figure(figsize=(14, 8))
    df_plot = pd.DataFrame({'Task': tasks * 2, 'AUC': [mtl_res[t] for t in tasks] + [stl_res[t] for t in tasks],
                            'Model': ['MTL'] * len(tasks) + ['STL'] * len(tasks)}).sort_values('AUC')
    sns.barplot(data=df_plot, x='AUC', y='Task', hue='Model', palette='magma')
    plt.title('Final Comparison: MTL vs STL')
    plt.xlim(0.5, 1.0)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=gains, y=tasks, palette='coolwarm')
    plt.axvline(0, color='black')
    plt.title('MTL Gain per Task (Positive = Success)')
    plt.xlabel('Gain (MTL - STL)')
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dm = ADMETDataManager('dataset_with_embeddings.parquet')
    df = dm.preprocess()

    loaders = {
        s: DataLoader(ADMETDataset(df[df['split'] == s], dm.feature_cols), batch_size=128, shuffle=(s == 'train'))
        for s in ['train', 'val', 'test']}

    # MTL
    task_freq = torch.tensor(
        df[df['split'] == 'train']['task_idx'].value_counts().sort_index().values / len(df[df['split'] == 'train']),
        dtype=torch.float32).to(device)
    mtl_model = run_mtl_training(loaders, dm, device, task_freq)

    # Ewaluacja MTL na teście
    mtl_model.eval()
    mtl_test_res = {}
    for i, task in enumerate(dm.tasks):
        sub = df[(df['split'] == 'test') & (df['task_idx'] == i)]
        ds = DataLoader(ADMETDataset(sub, dm.feature_cols), batch_size=128)
        p, t = [], []
        with torch.no_grad():
            for x, y, idx in ds:
                p.extend(torch.sigmoid(mtl_model(x.to(device), idx.to(device))).cpu().numpy())
                t.extend(y.numpy())
        mtl_test_res[task] = roc_auc_score(t, p)

    # STL
    stl_test_res = run_stl_baseline(df, dm.tasks, dm.feature_cols, device)

    # Raport
    print("\n" + "=" * 70)
    print(f"{'ZADANIE':<35} | {'STL':<8} | {'MTL':<8} | {'ZYSK'}")
    print("-" * 70)
    for t in dm.tasks:
        print(f"{t:<35} | {stl_test_res[t]:.4f} | {mtl_test_res[t]:.4f} | {mtl_test_res[t] - stl_test_res[t]:+.4f}")

    plot_results(mtl_test_res, stl_test_res)