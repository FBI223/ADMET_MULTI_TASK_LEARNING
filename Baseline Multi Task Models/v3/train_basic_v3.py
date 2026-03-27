import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import deepchem as dc
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from tdc.single_pred import ADME
from tqdm import tqdm
from rdkit import RDLogger
import xgboost as xgb
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Wyciszenie logów RDKit
RDLogger.DisableLog('rdApp.*')

os.makedirs('results/plots', exist_ok=True)

# --- KONFIGURACJA ZADAŃ ---
TASK_CONFIG = {
    'Caco2_Wang': 'reg',
    'Lipophilicity_AstraZeneca': 'reg',
    'Solubility_AqSolDB': 'reg',
    'BBB_Martins': 'clf'
}
TASK_NAMES = list(TASK_CONFIG.keys())


# --- 1. FABRYKA CECH MOLEKULARNYCH ---
class ADMETFeaturizer:
    def __init__(self):
        self.graph_feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    def get_morgan(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros(1024)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    def get_rdkit(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros(5)
        return np.array([Descriptors.MolLogP(mol), Descriptors.MolWt(mol),
                         Descriptors.NumHDonors(mol), Descriptors.TPSA(mol),
                         Descriptors.NumRotatableBonds(mol)])

    def get_graph_emb(self, smiles):
        try:
            feat = self.graph_feat.featurize([smiles])
            if len(feat) == 0 or feat[0] is None or isinstance(feat[0], np.ndarray):
                return np.zeros(30)
            return np.mean(feat[0].node_features, axis=0)
        except:
            return np.zeros(30)

    def prepare_data(self, df, mode='ABC'):
        X = []
        print(f"\n[INFO] Generowanie cech dla trybu: {mode}")
        for s in tqdm(df['smiles']):
            feats = []
            if 'A' in mode: feats.append(self.get_morgan(s))
            if 'B' in mode: feats.append(self.get_rdkit(s))
            if 'C' in mode: feats.append(self.get_graph_emb(s))
            X.append(np.concatenate(feats))
        return np.array(X)


# --- 2. ARCHITEKTURA MODELU MTL (RESNET) ---
class MultiTaskResNet(nn.Module):
    def __init__(self, input_dim, n_tasks, hidden_dim=512):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4)
            ) for _ in range(3)
        ])

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x):
        x = torch.relu(self.input_fc(x))
        for layer in self.shared_layers:
            x = layer(x) + x  # Residual Connection
        return torch.cat([head(x) for head in self.heads], dim=1)


# --- 3. LOGIKA UCZENIA I STRATY ---
def masked_loss(output, target):
    mask = ~torch.isnan(target)
    if not mask.any(): return torch.tensor(0.0, requires_grad=True)
    return torch.sqrt(nn.MSELoss()(output[mask], target[mask]))


def masked_weighted_loss(output, target, task_weights):
    total_loss = 0
    for i in range(target.shape[1]):
        mask = ~torch.isnan(target[:, i])
        if mask.any():
            # MSE dla danego zadania * jego waga
            loss = nn.MSELoss()(output[mask, i], target[mask, i])
            total_loss += loss * task_weights[i]
    return total_loss


class Trainer:
    def __init__(self, model, lr=1e-3, patience=7):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Dodaj to:
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
        self.patience = patience
        self.best_loss = float('inf')

    def fit(self, train_loader, val_loader, epochs=50, task_weights=None):
        if task_weights is None:
            task_weights = torch.ones(len(TASK_NAMES))

        for epoch in range(epochs):
            self.model.train()
            for x, y in train_loader:
                self.optimizer.zero_grad()
                # Użyj nowej funkcji straty:
                loss = masked_weighted_loss(self.model(x), y, task_weights)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    val_loss += masked_weighted_loss(self.model(x), y, task_weights).item()
            val_loss /= len(val_loader)

            # Dodaj to:
            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                counter = 0
            else:
                counter += 1
                if counter >= self.patience: break




# --- 4. ANALIZA I WYKRESY ---
def run_data_analysis(df):
    print("\n" + "=" * 50 + "\nANALIZA STATYSTYCZNA I KORELACJE\n" + "=" * 50)
    corr_matrix = df[TASK_NAMES].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Macierz Korelacji ADMET")
    plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc


def plot_classification_results(model, loader, task_names, mode):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            all_preds.append(model(x))
            all_targets.append(y)

    preds = torch.cat(all_preds).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()

    clf_tasks = [n for n in task_names if TASK_CONFIG[n] == 'clf']
    if not clf_tasks: return

    plt.figure(figsize=(8, 6))
    for name in clf_tasks:
        idx = task_names.index(name)
        mask = ~np.isnan(targets[:, idx])
        y_true, y_score = targets[mask, idx], preds[mask, idx]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Tryb {mode}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # ZAPIS I WYŚWIETLANIE
    plt.savefig(f'results/plots/ROC_{mode}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_parity_results(model, loader, task_names, mode):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            all_preds.append(model(x))
            all_targets.append(y)

    preds = torch.cat(all_preds).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()

    plt.figure(figsize=(15, 10))
    reg_tasks = [n for n in task_names if TASK_CONFIG[n] == 'reg']
    for i, name in enumerate(reg_tasks):
        plt.subplot(2, 2, i + 1)
        idx = task_names.index(name)
        mask = ~np.isnan(targets[:, idx])
        y_true, y_pred = targets[mask, idx], preds[mask, idx]

        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, '--r', alpha=0.75, zorder=3)
        plt.title(f"Parity Plot: {name} (Mode {mode})")
        plt.xlabel("Rzeczywiste")
        plt.ylabel("Przewidziane")

    plt.tight_layout()
    # ZAPIS I WYŚWIETLANIE
    plt.savefig(f'results/plots/Parity_{mode}.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_report_plots(df):
    sns.set_theme(style="whitegrid")

    # Heatmapa Zysku %
    pivot_stl = df.pivot(index='Task', columns='Mode', values='STL_Score')
    pivot_mtl = df.pivot(index='Task', columns='Mode', values='MTL_Score')
    gain_pct = pd.DataFrame(index=pivot_stl.index, columns=pivot_stl.columns)

    for task in TASK_NAMES:
        s, m = pivot_stl.loc[task], pivot_mtl.loc[task]
        # Dla regresji: im mniejszy RMSE tym lepiej, więc zysk to (STL-MTL)/STL
        if TASK_CONFIG[task] == 'reg':
            gain_pct.loc[task] = (s - m) / s * 100
        else:
            # Dla klasyfikacji: im wyższy AUROC tym lepiej
            gain_pct.loc[task] = (m - s) / s * 100

    plt.figure(figsize=(12, 6))
    sns.heatmap(gain_pct.astype(float), annot=True, cmap='RdYlGn', center=0, fmt=".1f")
    plt.title("Zysk/Strata MTL względem XGBoost STL (%)")

    # ZAPIS I WYŚWIETLANIE
    plt.savefig('results/plots/Heatmap_Gain_MTL.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Zapisz też same liczby do CSV
    gain_pct.to_csv("results/mtl_gain_stats.csv")

# --- 5. MAIN PIPELINE ---
def main():
    # Pobieranie i łączenie danych
    task_dfs = []
    print("\nPobieranie danych z TDC...")
    for tn in TASK_NAMES:
        data = ADME(name=tn).get_data()
        task_dfs.append(data[['Drug', 'Y']].rename(columns={'Drug': 'smiles', 'Y': tn}))

    full_df = task_dfs[0]
    for df in task_dfs[1:]: full_df = pd.merge(full_df, df, on='smiles', how='outer')
    full_df = full_df.drop_duplicates(subset=['smiles']).reset_index(drop=True)

    run_data_analysis(full_df)

    featurizer = ADMETFeaturizer()
    modes = ['A', 'B', 'C', 'AB', 'AC', 'ABC']
    final_comparison = []

    for mode in modes:
        print(f"\n{'=' * 40}\nEKSPERYMENT: {mode}\n{'=' * 40}")
        try:
            X = featurizer.prepare_data(full_df, mode=mode)
            Y = full_df[TASK_NAMES].values.astype(float)
            X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

            sc = StandardScaler()
            X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)

            # Wewnątrz pętli for mode in modes w main():
            print(f">>> Trenowanie Single-Task Baseline (XGBoost) dla trybu {mode}...")
            stl_res = {}
            for i, name in enumerate(TASK_NAMES):
                mask_tr, mask_te = ~np.isnan(Y_tr[:, i]), ~np.isnan(Y_te[:, i])
                x_tr, y_tr = X_tr[mask_tr], Y_tr[mask_tr, i]
                x_te, y_te = X_te[mask_te], Y_te[mask_te, i]

                if TASK_CONFIG[name] == 'reg':
                    model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1)
                    model_xgb.fit(x_tr, y_tr)
                    stl_res[name] = np.sqrt(mean_squared_error(y_te, model_xgb.predict(x_te)))
                else:
                    model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss',
                                                  n_jobs=-1)
                    model_xgb.fit(x_tr, y_tr)
                    stl_res[name] = roc_auc_score(y_te, model_xgb.predict_proba(x_te)[:, 1])

            # MTL (ResNet)
            train_l = DataLoader(
                TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)),
                batch_size=64, shuffle=True)
            test_l = DataLoader(
                TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(Y_te, dtype=torch.float32)),
                batch_size=64)

            model = MultiTaskResNet(input_dim=X_tr.shape[1], n_tasks=len(TASK_NAMES))
            Trainer(model).fit(train_l, test_l)
            model.load_state_dict(torch.load('best_model.pth'))

            # Eval MTL
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
                mtl_res = {}
                for i, name in enumerate(TASK_NAMES):
                    mask = ~np.isnan(Y_te[:, i])
                    if TASK_CONFIG[name] == 'reg':
                        mtl_res[name] = np.sqrt(mean_squared_error(Y_te[mask, i], preds[mask, i]))
                    else:
                        mtl_res[name] = roc_auc_score(Y_te[mask, i], preds[mask, i])

            for name in TASK_NAMES:
                final_comparison.append(
                    {'Mode': mode, 'Task': name, 'STL_Score': stl_res[name], 'MTL_Score': mtl_res[name]})

            # Dla ostatniego, najpełniejszego trybu rysujemy Parity Plot
            if mode == 'ABC':
                print("\n[PLOT] Generowanie wykresów regresji (Parity Plots)...")
                plot_parity_results(model, test_l, TASK_NAMES)

                print("[PLOT] Generowanie wykresów klasyfikacji (ROC Curve)...")
                plot_classification_results(model, test_l, TASK_NAMES)

        except Exception as e:
            print(f"Błąd w trybie {mode}: {e}")

    # Raport końcowy do CSV
    results_df = pd.DataFrame(final_comparison)
    results_df.to_csv("results/full_admet_comparison.csv", index=False)

    # Wygeneruj heatmapę
    generate_report_plots(results_df)

    print("\n[INFO] Wszystkie pliki (CSV i PNG) zostały zapisane w folderze 'results/'")


if __name__ == "__main__":
    main()