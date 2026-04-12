import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

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


class MTLDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.mask = torch.tensor(mask)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.mask[idx]


class MTLNet(nn.Module):
    def __init__(self, input_dim, n_tasks, *, head_neurons: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, head_neurons),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.heads = nn.ModuleList([nn.Linear(head_neurons, 1) for _ in range(n_tasks)])

    def forward(self, x):
        shared_rep = self.shared(x)
        # Przelewamy wspólną reprezentację przez każdą z głowic
        logits = torch.cat([head(shared_rep) for head in self.heads], dim=1)
        return logits

def masked_loss(logits, y, mask):
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def calculate_mtl_auc(logits, targets, mask):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = targets.cpu().numpy()
    mask = mask.cpu().numpy()
    task_aucs = []

    for i in range(targets.shape[1]):
        valid_idx = mask[:, i] == 1
        y_t, y_p = targets[valid_idx, i], probs[valid_idx, i]
        if len(np.unique(y_t)) > 1:
            task_aucs.append(roc_auc_score(y_t, y_p))
        else:
            task_aucs.append(np.nan)
    return task_aucs


def prepare_mtl_data(df, n_bits=1024):
    tasks = sorted(df['task'].dropna().unique().tolist())
    pivot_df = df.pivot_table(index='smiles', columns='task', values='label', aggfunc='first')

    features_df = df.drop_duplicates(subset=['smiles']).set_index('smiles')
    merged_df = features_df.join(pivot_df, how='inner')

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = np.array(
        [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.zeros(n_bits) for s in
         merged_df.index])

    valid_features = [f for f in PRIMARY_FEATURES if f in merged_df.columns]
    scaler = StandardScaler()
    rdkit_scaled = scaler.fit_transform(merged_df[valid_features].fillna(0))

    X = np.hstack([rdkit_scaled, fps]).astype(np.float32)
    y_raw = merged_df[tasks].values.astype(np.float32)
    mask = (~np.isnan(y_raw)).astype(np.float32)
    y = np.nan_to_num(y_raw)
    splits = merged_df['split'].values

    feature_names = valid_features + [f'bit_{i}' for i in range(n_bits)]
    return X, y, mask, tasks, splits, feature_names


def prepare_mtl_data_without_morgan(df):
    # 1. Agregacja zadań (Long -> Wide)
    tasks = sorted(df['task'].dropna().unique().tolist())
    pivot_df = df.pivot_table(index='smiles', columns='task', values='label', aggfunc='first')

    # 2. Pobranie unikalnych cech per SMILES
    features_df = df.drop_duplicates(subset=['smiles']).set_index('smiles')
    merged_df = features_df.join(pivot_df, how='inner')

    # 3. Wybór i skalowanie tylko deskryptorów (bez Morgana)
    valid_features = [f for f in PRIMARY_FEATURES if f in merged_df.columns]
    scaler = StandardScaler()
    X = scaler.fit_transform(merged_df[valid_features].fillna(0)).astype(np.float32)

    # 4. Przygotowanie etykiet i maski
    y_raw = merged_df[tasks].values.astype(np.float32)
    mask = (~np.isnan(y_raw)).astype(np.float32)
    y = np.nan_to_num(y_raw)

    # 5. Podziały i nazwy cech
    splits = merged_df['split'].values
    feature_names = valid_features

    return X, y, mask, tasks, splits, feature_names

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        opt.zero_grad()
        loss = masked_loss(model(xb), yb, mb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, task_names):
    model.eval()
    total_loss = 0
    all_l, all_y, all_m = [], [], []

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            # Forward pass
            logits = model(xb)

            # Liczymy Validation Loss
            loss = masked_loss(logits, yb, mb)
            total_loss += loss.item()

            # Zbieramy do AUC (przenosimy z powrotem na CPU, żeby nie zapchać RAMu GPU)
            all_l.append(logits.cpu())
            all_y.append(yb.cpu())
            all_m.append(mb.cpu())

    avg_loss = total_loss / len(loader)
    aucs = calculate_mtl_auc(torch.cat(all_l), torch.cat(all_y), torch.cat(all_m))

    # Teraz zwracamy DWIE rzeczy: loss i słownik z AUC
    return avg_loss, dict(zip(task_names, aucs))


def permutation_importance_mtl(model, loader, device, feature_names):
    X_test, y_test, m_test = next(iter(loader))
    X_test = X_test.to(device)
    model.eval()

    base_auc = np.nanmean(calculate_mtl_auc(model(X_test).cpu(), y_test, m_test))
    imps = []

    for i in range(X_test.shape[1]):
        hold = X_test[:, i].clone()
        X_test[:, i] = X_test[torch.randperm(X_test.shape[0]), i]
        new_auc = np.nanmean(calculate_mtl_auc(model(X_test).cpu(), y_test, m_test))
        imps.append(base_auc - new_auc)
        X_test[:, i] = hold

    return pd.DataFrame({'feature': feature_names, 'importance': imps}).sort_values('importance', ascending=False)


def plot_comprehensive_head_experiment(results, head_sizes, tasks, epochs):
    epochs_range = range(1, epochs + 1)
    colors = {64: 'tab:blue', 128: 'tab:green', 256: 'tab:red'}
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Lewy wykres: MTL Train Loss vs Val Loss
    for size in head_sizes:
        ax1.plot(epochs_range, results[size]['mtl_history']['train_loss'],
                 label=f'Train Loss ({size})', color=colors[size], linewidth=2, linestyle='-')
        ax1.plot(epochs_range, results[size]['mtl_history']['val_loss'],
                 label=f'Val Loss ({size})', color=colors[size], linewidth=2, linestyle='--', alpha=0.7)

    ax1.set_title('MTL Loss (Train - ciągła, Val - przerywana)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoka', fontsize=12)
    ax1.set_ylabel('Loss (BCE)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Prawy wykres: MTL vs STL (Średnie Val AUC)
    for size in head_sizes:
        ax2.plot(epochs_range, results[size]['mtl_history']['val_auc_avg'],
                 label=f'MTL Avg AUC ({size})', color=colors[size], linewidth=2.5, linestyle='-')
        ax2.plot(epochs_range, results[size]['stl_history']['val_auc_avg'],
                 label=f'STL Avg AUC ({size})', color=colors[size], linewidth=2, linestyle='--', alpha=0.7)

    ax2.set_title('Średnie Val AUC (MTL - ciągła, STL - przerywana)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoka', fontsize=12)
    ax2.set_ylabel('ROC-AUC', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # FIGURA 2: SIATKA WYNIKÓW DLA KAŻDEGO TASKU
    cols = 4
    rows = int(np.ceil(len(tasks) / cols))

    fig2, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]

        # Rysujemy 6 linii dla każdego tasku (3 rozmiary x 2 modele)
        for size in head_sizes:
            mtl_vals = results[size]['mtl_history']['val_aucs_per_task'][task]
            stl_vals = results[size]['stl_history']['val_aucs_per_task'][task]

            ax.plot(epochs_range, mtl_vals, label=f'MTL ({size})', color=colors[size], linestyle='-', linewidth=2)
            ax.plot(epochs_range, stl_vals, label=f'STL ({size})', color=colors[size], linestyle='--', linewidth=2,
                    alpha=0.7)

        ax.set_title(task, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Dodajemy oś Y tylko do lewej kolumny
        if i % cols == 0:
            ax.set_ylabel('Val AUC')

        # Legendę dajemy tylko na pierwszym wykresie, żeby nie śmiecić
        if i == 0:
            ax.legend(fontsize=9, loc='lower right')

    # Usuwamy puste wykresy na końcu siatki
    for j in range(len(tasks), len(axes)):
        fig2.delaxes(axes[j])

    plt.suptitle('Analiza per Task: Wpływ wielkości głowicy i architektury (MTL vs STL)', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()