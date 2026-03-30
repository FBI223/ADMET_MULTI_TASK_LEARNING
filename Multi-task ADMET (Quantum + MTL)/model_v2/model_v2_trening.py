import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch.nn.functional as F

# Ignorowanie ostrzeżeń o palecie w seaborn
warnings.filterwarnings("ignore", category=FutureWarning)


def plot_compare_mtl_stl(mtl_results, stl_results):
    """
    Tworzy zestawienie słupkowe porównujące wyniki AUC dla MTL i STL.
    Pozwala zidentyfikować zadania, które najbardziej zyskały na transferze wiedzy.
    """
    tasks = list(mtl_results.keys())

    # Tworzymy listę słowników do łatwej konwersji na DataFrame
    data = []
    for t in tasks:
        # Dodajemy wynik MTL
        data.append({
            'Zadanie': t,
            'ROC-AUC': mtl_results[t],
            'Model': 'QW-MTL (Multi-Task)'
        })
        # Dodajemy wynik STL
        data.append({
            'Zadanie': t,
            'ROC-AUC': stl_results.get(t, np.nan),
            'Model': 'Baseline (Single-Task)'
        })

    df_plot = pd.DataFrame(data)

    # Sortujemy zadania, aby te z najwyższym AUC w MTL były na górze
    order = df_plot[df_plot['Model'] == 'QW-MTL (Multi-Task)'].sort_values('ROC-AUC', ascending=False)['Zadanie']

    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")

    # Rysowanie słupków obok siebie
    ax = sns.barplot(
        data=df_plot,
        x='ROC-AUC',
        y='Zadanie',
        hue='Model',
        order=order,
        palette=['#4C72B0', '#DD8452']  # Stonowany niebieski i pomarańczowy
    )

    # Linia odniesienia (losowy model)
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Random Baseline')

    plt.title('Porównanie Skuteczności: QW-MTL vs Single-Task ResNet', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('ROC-AUC Score', fontsize=12)
    plt.ylabel('Zadania ADMET', fontsize=12)
    plt.xlim(0.4, 1.05)  # Skala od 0.4 dla lepszej czytelności różnic

    plt.legend(title='Typ Modelu', loc='lower right', frameon=True)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Rysuje wykres straty (Loss) dla zbioru treningowego i walidacyjnego."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o', color='#1f77b4')
    plt.plot(history['val_loss'], label='Val Loss', marker='x', color='#ff7f0e')
    plt.title('Krzywa Uczenia: Weighted Multi-Task Loss', fontsize=14)
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('Weighted Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_metrics_per_task(metrics, title="Wyniki AUC dla Zadań ADMET"):
    """Tworzy wykres słupkowy z metrykami AUC dla każdego zadania."""
    # Filtrujemy tylko te zadania, które mają policzony wynik (nie są NaN)
    clean_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}

    # Sortujemy od najlepszego do najgorszego
    sorted_metrics = dict(sorted(clean_metrics.items(), key=lambda item: item[1], reverse=True))

    names = list(sorted_metrics.keys())
    values = list(sorted_metrics.values())

    plt.figure(figsize=(12, 8))
    # Używamy palety barw od zielonej do niebieskiej
    sns.barplot(x=values, y=names, palette='magma')

    # Linia odniesienia dla losowego zgadywania (0.5)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random baseline (0.5)')

    plt.title(title, fontsize=16)
    plt.xlabel('ROC-AUC Score', fontsize=12)
    plt.xlim(0.4, 1.0)  # Skupiamy się na zakresie powyżej losowości
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_task_correlation(model, loader, tasks, device):
    model.eval()
    all_preds = []

    # Zbieramy surowe logity (przed sigmoidą), by zobaczyć "pewność" modelu
    with torch.no_grad():
        for x, y, t in loader:
            x = x.to(device)
            # Dla korelacji musimy przepuścić każdą cząsteczkę przez WSZYSTKIE głowy
            # (nie tylko tę, która jest w long format)
            #features = model.encoder(x)
            features = model.shared(torch.relu(model.input_proj(x)))
            batch_preds = []
            for head in model.heads:
                batch_preds.append(head(features).squeeze(-1).cpu().numpy())
            all_preds.append(np.array(batch_preds).T)

    # Łączymy wyniki w jeden DataFrame
    full_preds = np.vstack(all_preds)
    corr_df = pd.DataFrame(full_preds, columns=tasks)

    # Obliczamy korelacje (Pearson)
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Korelacja Przewidywań Między Zadaniami (Model Insight)")
    plt.tight_layout()
    plt.show()
    return corr_matrix


def plot_final_analysis(auc_metrics, beta_values):
    """Generuje raport graficzny: AUC oraz Wyuczone Beta dla każdego zadania."""
    # Przygotowanie danych
    tasks = list(auc_metrics.keys())
    auc_scores = [auc_metrics[t] for t in tasks]
    betas = [beta_values[t] for t in tasks]

    # Tworzenie DataFrame do łatwego sortowania
    df = pd.DataFrame({
        'Task': tasks,
        'AUC': auc_scores,
        'Beta': betas
    }).sort_values('AUC', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # WYKRES 1: AUC (Skuteczność)
    sns.barplot(x='AUC', y='Task', data=df, ax=ax1, palette='viridis')
    ax1.axvline(0.5, color='red', linestyle='--', label='Random (0.5)')
    ax1.axvline(0.9, color='gold', linestyle=':', label='Excellent (>0.9)')
    ax1.set_title('Skuteczność modelu (ROC-AUC)', fontsize=15, fontweight='bold')
    ax1.set_xlim(0.4, 1.02)
    ax1.legend()

    # WYKRES 2: Wyuczone Wagi Beta (Priorytetyzacja)
    sns.barplot(x='Beta', y='Task', data=df, ax=ax2, palette='magma')
    ax2.set_title('Wyuczone parametry Beta (QW-MTL Weighting)', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Beta Value (Higher = More Penalized Data Scale)')

    # Dodanie etykiet z wartościami na słupkach
    for i, v in enumerate(df['Beta']):
        ax2.text(v + 0.05, i, f'{v:.2f}', color='black', va='center')

    plt.tight_layout()
    plt.show()






def check_split_integrity(df):
    # Sprawdza, czy ta sama cząsteczka (SMILES) ma tylko jeden przypisany split
    split_counts = df.groupby('smiles')['split'].nunique()
    conflicts = split_counts[split_counts > 1].count()
    if conflicts > 0:
        print(f"!!! UWAGA: Masz {conflicts} cząsteczek z różnymi splitami! Napraw to.")
    else:
        print("✓ Splity są spójne: brak wycieku danych.")



# ==========================================
# 1. ZARZĄDZANIE DANYCH (FIX: No Data Leakage)
# ==========================================
class ADMETDataManager:
    def __init__(self, file_path):
        print(f"Ładowanie danych z: {file_path}...")
        self.df = pd.read_parquet(file_path)

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
        print("Preprocessing: Skalowanie dopasowane TYLKO do zbioru treningowego...")
        train_mask = self.df['split'] == 'train'

        # A. Embeddings
        scaler_emb = StandardScaler()
        scaler_emb.fit(self.df.loc[train_mask, self.emb_cols])
        self.df[self.emb_cols] = scaler_emb.transform(self.df[self.emb_cols])

        # B. RDKit - skalujemy tylko tam, gdzie success == 1 I split == train
        rdkit_fit_mask = train_mask & (self.df['success'] == 1)
        if rdkit_fit_mask.any():
            scaler_rdkit = StandardScaler()
            scaler_rdkit.fit(self.df.loc[rdkit_fit_mask, self.rdkit_cols])
            self.df[self.rdkit_cols] = scaler_rdkit.transform(self.df[self.rdkit_cols])
            self.df.loc[self.df['success'] == 0, self.rdkit_cols] = 0

        # C. Quantum - indywidualnie dla każdej cechy
        for q_col, m_col in zip(self.qc_cols, self.mask_cols):
            q_fit_mask = train_mask & (self.df[m_col] == 1)
            if q_fit_mask.any():
                scaler_q = StandardScaler()
                scaler_q.fit(self.df.loc[q_fit_mask, [q_col]])
                self.df[[q_col]] = scaler_q.transform(self.df[[q_col]])
                self.df.loc[self.df[m_col] == 0, q_col] = 0

        return self.df


class ADMETDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)
        self.task_idx = torch.tensor(df['task_idx'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.task_idx[idx]


# ==========================================
# 2. MODEL I LOSS (Większa Regularyzacja)
# ==========================================

class STLNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, 512)
        # Identyczny Shared Encoder jak w MTL
        self.backbone = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512)
        )
        # Identyczna głowa jak w MTL
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.backbone(x)
        return self.head(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        # x + self.net(x) to właśnie to słynne połączenie rezydualne
        return F.relu(x + self.net(x))


class MTLNet(nn.Module):
    def __init__(self, in_dim, n_tasks):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, 512)

        # Shared Encoder: 2 bloki rezydualne wyciągają wspólne cechy ADMET
        self.shared = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # Głębsze głowy (Task-specific) pozwalają zadaniom na odrębność
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x, task_indices):
        x = F.relu(self.input_proj(x))
        features = self.shared(x)

        # Logika dla formatu "long"
        logits = torch.zeros(x.size(0), device=x.device)
        for i, head in enumerate(self.heads):
            mask = (task_indices == i)
            if mask.any():
                logits[mask] = head(features[mask]).squeeze(-1)
        return logits


class QWMTLNetwork(nn.Module):
    def __init__(self, input_dim, num_tasks):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Zwiększony Dropout (z 0.3 na 0.5)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_tasks)])

    def forward(self, x, task_indices):
        features = self.encoder(x)
        logits = torch.zeros(x.size(0), device=x.device)
        for i, head in enumerate(self.heads):
            mask = (task_indices == i)
            if mask.any():
                logits[mask] = head(features[mask]).squeeze(-1)
        return logits


class QWMTLLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_betas = nn.Parameter(torch.zeros(num_tasks))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets, task_indices):
        raw_loss = self.bce(logits, targets)
        batch_size = targets.size(0)
        task_counts = torch.bincount(task_indices, minlength=len(self.log_betas)).float()
        r_t = task_counts / (batch_size + 1e-6)

        betas = torch.nn.functional.softplus(self.log_betas)
        task_weights = torch.pow(r_t + 1e-6, betas)

        total_loss = 0
        for t_id in range(len(self.log_betas)):
            mask = (task_indices == t_id)
            if mask.any():
                total_loss += task_weights[t_id] * raw_loss[mask].mean()
        return total_loss


# ==========================================
# 3. FUNKCJE EWALUACJI
# ==========================================
def evaluate_model(model, loader, tasks, device):
    model.eval()
    all_preds, all_targets, all_tasks = [], [], []
    with torch.no_grad():
        for x, y, t in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, t.to(device))
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_tasks.extend(t.numpy())

    res_df = pd.DataFrame({'pred': all_preds, 'target': all_targets, 'task_idx': all_tasks})
    metrics = {name: np.nan for name in tasks}
    for i, name in enumerate(tasks):
        sub = res_df[res_df['task_idx'] == i]
        if len(sub['target'].unique()) > 1:
            metrics[name] = roc_auc_score(sub['target'], sub['pred'])
    return metrics


# ==========================================
# 4. GŁÓWNA PĘTLA
# ==========================================


def run_baseline_stl(df, feature_cols, task_name, device):
    print(f"--- Trenowanie Baseline STL dla: {task_name} ---")

    # 1. Filtrowanie danych tylko dla tego zadania
    task_df = df[df['task'] == task_name].copy()

    # Tworzenie loaderów
    def make_loader(split_name, b_size, shuffle):
        sub_df = task_df[task_df['split'] == split_name]
        ds = ADMETDataset(sub_df, feature_cols)
        return DataLoader(ds, batch_size=b_size, shuffle=shuffle)

    train_loader = make_loader('train', 64, True)
    val_loader = make_loader('val', 128, False)
    test_loader = make_loader('test', 128, False)

    # 2. Inicjalizacja modelu
    model = STLNet(len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0

    # 3. Krótki trening (Single task zazwyczaj szybciej zbiega)
    for epoch in range(15):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Szybka walidacja
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = model(x.to(device))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(y.numpy())

        current_val_auc = roc_auc_score(val_targets, val_preds)
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            torch.save(model.state_dict(), f'temp_stl_{task_name}.pth')

    # 4. Finalny test na oficjalnym zbiorze TEST
    model.load_state_dict(torch.load(f'temp_stl_{task_name}.pth'))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            logits = model(x.to(device))
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_targets.extend(y.numpy())

    final_test_auc = roc_auc_score(test_targets, test_preds)
    return final_test_auc


# ... (Wszystkie Twoje klasy i funkcje pomocnicze zostają bez zmian powyżej) ...

def run_mtl_pipeline(loaders, dm, device, epochs=20):
    """Izoluje proces treningu Multi-Task Learning."""
    print(f"\n>>> [KROK: MTL] Rozpoczynam trening QW-MTL na {device}...")

    model = MTLNet(len(dm.feature_cols), len(dm.tasks)).to(device)
    criterion = QWMTLLoss(len(dm.tasks)).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.0005},
        {'params': criterion.parameters(), 'lr': 0.001}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y, t in loaders['train']:
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, t), y, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, t in loaders['val']:
                x, y, t = x.to(device), y.to(device), t.to(device)
                val_loss += criterion(model(x, t), y, t).item()

        avg_train = train_loss / len(loaders['train'])
        avg_val = val_loss / len(loaders['val'])
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_qw_mtl_model.pth')
            print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train:.4f} | NEW BEST Val Loss: {avg_val:.4f}")
        else:
            print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # Finalna ekstrakcja wag i modelu
    model.load_state_dict(torch.load('best_qw_mtl_model.pth', weights_only=True))

    final_betas = {}
    with torch.no_grad():
        betas_tensor = torch.nn.functional.softplus(criterion.log_betas).cpu().numpy()
        for name, b in zip(dm.tasks, betas_tensor):
            final_betas[name] = float(b)

    return model, history, final_betas


def run_stl_pipeline(df_clean, feature_cols, tasks, device):
    """Izoluje proces obliczania baselinów Single-Task."""
    print(f"\n>>> [KROK: STL] Rozpoczynam obliczanie Single-Task Baselines...")
    stl_results = {}
    for task in tasks:
        stl_auc = run_baseline_stl(df_clean, feature_cols, task, device)
        stl_results[task] = stl_auc
    return stl_results


def display_final_comparison(test_auc_mtl, stl_results, final_betas):
    """Wypisuje końcowe statystyki per task w formie tabeli."""
    print("\n" + "=" * 85)
    print(f"{'ZADANIE ADMET':<35} | {'STL AUC':<10} | {'MTL AUC':<10} | {'BETA':<8} | {'ZYSK'}")
    print("-" * 85)

    total_gain = 0
    for task in sorted(test_auc_mtl.keys()):
        stl = stl_results.get(task, np.nan)
        mtl = test_auc_mtl.get(task, np.nan)
        beta = final_betas.get(task, np.nan)
        gain = mtl - stl
        total_gain += gain

        print(f"{task:<35} | {stl:.4f}     | {mtl:.4f}     | {beta:.2f}     | {gain:+.4f}")

    print("-" * 85)
    print(f"{'ŚREDNI ZYSK MTL':<63} | {total_gain / len(test_auc_mtl):+.4f}")
    print("=" * 85)


# ==========================================
# GŁÓWNA PĘTLA WYKONAWCZA
# ==========================================
if __name__ == "__main__":
    FILE_PATH = 'dataset_with_embeddings.parquet'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. PRZYGOTOWANIE DANYCH
    dm = ADMETDataManager(FILE_PATH)
    df_processed = dm.preprocess()
    check_split_integrity(dm.df)

    loaders = {
        s: DataLoader(ADMETDataset(df_processed[df_processed['split'] == s], dm.feature_cols),
                      batch_size=256, shuffle=(s == 'train'))
        for s in ['train', 'val', 'test']
    }

    # 2. PROCES MULTI-TASK (MTL)
    mtl_model, mtl_history, mtl_betas = run_mtl_pipeline(loaders, dm, device, epochs=20)
    test_auc_mtl = evaluate_model(mtl_model, loaders['test'], dm.tasks, device)

    # 3. PROCES SINGLE-TASK (STL)
    stl_results = run_stl_pipeline(df_processed, dm.feature_cols, dm.tasks, device)

    # 4. TABELA WYNIKÓW
    display_final_comparison(test_auc_mtl, stl_results, mtl_betas)

    # 5. WIZUALIZACJE
    print("\n>>> Generowanie wizualizacji raportowych...")
    plot_training_history(mtl_history)
    plot_metrics_per_task(test_auc_mtl, title="Finalne Wyniki AUC (MTL)")
    plot_task_correlation(mtl_model, loaders['test'], dm.tasks, device)
    plot_final_analysis(test_auc_mtl, mtl_betas)

    # Dodatkowe porównanie MTL vs STL (wykres zysku)
    plot_compare_mtl_stl(test_auc_mtl, stl_results)