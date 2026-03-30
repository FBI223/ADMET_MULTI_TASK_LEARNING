import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.args import TrainArgs
from chemprop.models import MoleculeModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from chemprop.features import mol2graph
from chemprop.features import BatchMolGraph
from sklearn.metrics import roc_auc_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
from chemprop.data.utils import scaffold_split
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

RDLogger.DisableLog('rdApp.*')


RUN_ID = str(int(time.time()))

torch.manual_seed(42)
np.random.seed(42)

# Ignorowanie ostrzeżeń o palecie w seaborn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os

BASE_DIR = f"results_{RUN_ID}"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TASK_PLOTS_DIR = os.path.join(PLOTS_DIR, "per_task")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TASK_PLOTS_DIR, exist_ok=True)


'''
SMILES
→ mol2graph(list)
→ BatchMolGraph
→ encoder([batch])
→ embedding
→ concat features
→ multitask heads

'''






def smiles_to_graph(smiles_list):
    return mol2graph(smiles_list)




def plot_auc_per_task(history, tasks):

    epochs = range(len(history["val_auc"]))

    for t_idx, t_name in enumerate(tasks):

        values = history["val_auc_tasks"][t_idx]

        # 🔴 filtr (opcjonalny)
        if np.nanmax(values) < 0.6:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values)

        plt.title(f"AUC: {t_name}")
        plt.xlabel("Epoch")
        plt.ylabel("ROC-AUC")
        plt.grid()

        filename = f"{t_name}.png".replace("/", "_")
        plt.savefig(os.path.join(TASK_PLOTS_DIR, filename))
        plt.close()


def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')

    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
    plt.close()


def plot_global_auc(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history["val_auc"], label="Val AUC")

    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(PLOTS_DIR, "val_auc.png"))
    plt.close()


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
    plt.savefig(os.path.join(PLOTS_DIR, "auc_bar.png"))
    plt.close()



def plot_task_correlation(model, loader, tasks, device):
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in loader:

            batch_graph = batch["graph"]
            rdkit = batch["rdkit"].to(device)
            qc = batch["qc"].to(device)
            qc_mask = batch["qc_mask"].to(device)

            logits = model(batch_graph, rdkit, qc, qc_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)

    # 🔴 shape: (N_samples, T)
    full_preds = np.vstack(all_preds)
    mask = ~np.isnan(full_preds).any(axis=1)
    full_preds = full_preds[mask]

    # 🔴 DataFrame: kolumny = task names
    corr_df = pd.DataFrame(full_preds, columns=tasks)

    # 🔴 korelacja
    corr_matrix = corr_df.corr(method="pearson")

    # 🔴 plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title("Task Prediction Correlation (QW-MTL)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task_correlation.png"))
    plt.close()

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
    plt.savefig(os.path.join(PLOTS_DIR, "final_analysis.png"))
    plt.close()







def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def split_data(data):

    scaffold_dict = defaultdict(list)

    for i, d in enumerate(data):
        scaffold = generate_scaffold(d["smiles"])
        scaffold_dict[scaffold].append(i)

    print("Unique scaffolds:", len(scaffold_dict))


    scaffolds = sorted(scaffold_dict.values(), key=lambda x: -len(x))

    train_idx, val_idx, test_idx = [], [], []

    n = len(data)
    train_cutoff = int(0.8 * n)
    val_cutoff = int(0.9 * n)

    for scaffold in scaffolds:
        if len(train_idx) + len(scaffold) <= train_cutoff:
            train_idx += scaffold
        elif len(val_idx) + len(scaffold) <= (val_cutoff - train_cutoff):
            val_idx += scaffold
        else:
            test_idx += scaffold

    train = [data[i] for i in train_idx]
    val   = [data[i] for i in val_idx]
    test  = [data[i] for i in test_idx]

    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))

    return train, val, test
def split_data_manual(data):

    np.random.seed(42)
    np.random.shuffle(data)

    n = len(data)

    train = data[:int(0.8*n)]
    val   = data[int(0.8*n):int(0.9*n)]
    test  = data[int(0.9*n):]

    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))

    return train, val, test





def load_dataset(path):

    df = pd.read_parquet(path)
    print(df.columns.tolist())

    tasks = sorted(df["task"].unique())
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    grouped = defaultdict(list)

    for _, row in df.iterrows():
        grouped[row["smiles"]].append(row)

    data = []

    for smiles, rows in grouped.items():

        T = len(tasks)

        y = np.zeros(T)
        mask = np.zeros(T)

        for r in rows:
            t = task_to_idx[r["task"]]
            y[t] = r["label"]
            mask[t] = 1

        rdkit_cols = [c for c in df.columns if c not in exclude]
        rdkit = rows[0][rdkit_cols].values.astype(np.float64)
        rdkit = np.clip(rdkit, -1e6, 1e6)  # zabezpieczenie
        rdkit = rdkit.astype(np.float32)
        rdkit = np.nan_to_num(rdkit, nan=0.0, posinf=0.0, neginf=0.0)

        qc = np.array([
            rows[0]["dipole"],
            rows[0]["homo_lumo"],
            rows[0]["electrons"],
            rows[0]["energy"]
        ], dtype=np.float32)

        qc_mask = np.array([
            rows[0]["mask_dipole"],
            rows[0]["mask_homo_lumo"],
            rows[0]["mask_electrons"],
            rows[0]["mask_energy"]
        ], dtype=np.float32)

        # 🔴 USUWAMY split całkowicie
        data.append({
            "smiles": smiles,
            "graph": smiles,
            "y": y,
            "mask": mask,
            "rdkit": rdkit,
            "qc": qc,
            "qc_mask": qc_mask
        })

    print("Total molecules:", len(data))

    return data, tasks



def normalize_features(data, rdkit_scaler=None, qc_scaler=None, fit=False):

    rdkit_all = np.stack([d["rdkit"] for d in data])
    qc_all = np.stack([d["qc"] for d in data])

    rdkit_all = np.nan_to_num(rdkit_all, nan=0.0, posinf=0.0, neginf=0.0)
    qc_all = np.nan_to_num(qc_all, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        rdkit_scaler = StandardScaler().fit(rdkit_all)
        qc_scaler = StandardScaler().fit(qc_all)

    rdkit_all = rdkit_scaler.transform(rdkit_all)
    qc_all = qc_scaler.transform(qc_all)

    for i, d in enumerate(data):
        d["rdkit"] = rdkit_all[i]
        d["qc"] = qc_all[i]

    return data, rdkit_scaler, qc_scaler



class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        return {
            "graph": d["graph"] ,   # mol2graph(smiles) # później Chemprop
            "rdkit": torch.tensor(d["rdkit"], dtype=torch.float32),
            "qc": torch.tensor(d["qc"], dtype=torch.float32),
            "qc_mask": torch.tensor(d["qc_mask"], dtype=torch.float32),
            "y": torch.tensor(d["y"], dtype=torch.float32),
            "mask": torch.tensor(d["mask"], dtype=torch.float32)
        }




def collate_fn(batch):

    smiles = [b["graph"] for b in batch]

    graph = mol2graph(smiles)  # 🔴 KLUCZ

    rdkit = torch.stack([b["rdkit"] for b in batch])
    qc = torch.stack([b["qc"] for b in batch])
    qc_mask = torch.stack([b["qc_mask"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])

    return {
        "graph": graph,
        "rdkit": rdkit,
        "qc": qc,
        "qc_mask": qc_mask,
        "y": y,
        "mask": mask
    }

def build_dataloader(data, batch_size=32, shuffle=True):

    dataset = SimpleDataset(data)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn  # 🔴 DODAJ
    )


def dummy_loader():

    B = 16
    T = 13

    smiles = ["CCO"] * B  # przykładowe molekuły
    graph = smiles_to_graph(smiles)

    for _ in range(10):

        yield {
            "graph": graph,  # 🔴 TERAZ GNN DZIAŁA
            "rdkit": torch.randn(B, 200),
            "qc": torch.randn(B, 4),
            "qc_mask": torch.randint(0, 2, (B, 4)).float(),
            "y": torch.randint(0, 2, (B, T)).float(),
            "mask": torch.ones(B, T)  # 🔴 ważne
        }




def build_encoder():
    args = TrainArgs()

    args.dropout = 0.1
    args.dataset_type = 'classification'
    args.hidden_size = 300
    args.depth = 3
    args.ffn_hidden_size = 300

    model = MoleculeModel(args)

    return model.encoder  # 🔴 tylko encoder


class TaskHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class TaskWeighting(nn.Module):
    def __init__(self, T):
        super().__init__()
        #self.log_beta = nn.Parameter(torch.zeros(T))
        self.log_beta = nn.Parameter(torch.zeros(T) + 0.1)

    def forward(self, losses, n_valid):
        r = n_valid / (n_valid.sum() + 1e-8)
        beta = F.softplus(self.log_beta)
        w = r ** beta
        return (w * losses).sum() / (w.sum() + 1e-8)



class QW_MTL(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()

        self.encoder = build_encoder()

        self.input_dim = 300 + 200 + 4 + 4
        self.num_tasks = num_tasks

        self.norm = nn.LayerNorm(self.input_dim)

        self.heads = nn.ModuleList([
            TaskHead(self.input_dim) for _ in range(num_tasks)
        ])

        self.weighting = TaskWeighting(num_tasks)

    def forward(self, batch, rdkit, qc, qc_mask):

        device = rdkit.device

        rdkit = rdkit.to(device)  # 🔴 DODAJ
        qc = qc.to(device)
        qc_mask = qc_mask.to(device)

        if batch is None:
            z = torch.randn(rdkit.shape[0], 300, device=device)
        else:
            z = self.encoder([batch])

        z = z.to(device)

        x = torch.cat([z, rdkit, qc, qc_mask], dim=1)
        x = self.norm(x)

        out = torch.cat([head(x) for head in self.heads], dim=1)

        return out


def compute_loss(model, preds, targets, mask):

    T = preds.shape[1]

    losses = []
    n_valid = []

    for t in range(T):
        m = mask[:, t] == 1

        if m.sum() == 0:
            losses.append(torch.tensor(0.0, device=preds.device))
            n_valid.append(torch.tensor(0.0, device=preds.device))
            continue

        loss_t = F.binary_cross_entropy_with_logits(
            preds[m, t],
            targets[m, t].clamp(0, 1),
            reduction='mean'
        )

        losses.append(loss_t)
        n_valid.append(m.sum())

    losses = torch.stack(losses)
    n_valid = torch.stack(n_valid).float()

    total_loss = model.weighting(losses, n_valid)

    return total_loss, losses  # 🔴 ZMIANA


def train_step(model, batch, rdkit, qc, qc_mask, targets, mask, opt):

    model.train()

    preds = model(batch, rdkit, qc, qc_mask)

    loss, weights = compute_loss(model, preds, targets, mask)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), weights



def testing():
    B = 16
    T = 13

    model = QW_MTL(T)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # fake data
    rdkit = torch.randn(B, 200)
    qc = torch.randn(B, 4)
    qc_mask = torch.randint(0, 2, (B, 4)).float()
    targets = torch.randint(0, 2, (B, T)).float()
    mask = torch.randint(0, 2, (B, T)).float()

    batch = None  # tutaj idzie chemprop batch

    # forward bez batcha (debug)
    z = torch.randn(B, 300)
    x = torch.cat([z, rdkit, qc, qc_mask], dim=1)


exclude = [
    "smiles", "label", "task", "split",
    "dipole", "homo_lumo", "electrons", "energy",
    "mask_dipole", "mask_homo_lumo", "mask_electrons", "mask_energy",
    "success"
]

def train(model, train_loader, val_loader, optimizer, scheduler, device, num_tasks):

    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_auc_tasks": {i: [] for i in range(num_tasks)}  # 🔴 NOWE
    }

    val_auc_tasks = {}
    best_auc = 0

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for batch in train_loader:

            batch_graph = batch["graph"]
            rdkit = batch["rdkit"].to(device)
            qc = batch["qc"].to(device)
            qc_mask = batch["qc_mask"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)

            preds = model(batch_graph, rdkit, qc, qc_mask)
            loss, _ = compute_loss(model, preds, targets, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        # 🔴 NORMALIZACJA (po pętli!)
        total_loss /= len(train_loader)

        scheduler.step()

        # 🔴 WALIDACJA
        val_loss, val_auc_tasks, val_auc_mean = evaluate_with_metrics(
            model, val_loader, device, num_tasks
        )

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            val_loss = np.nan



        # 🔴 EARLY STOPPING
        if val_auc_mean > best_auc:
            best_auc = val_auc_mean
            torch.save(model.state_dict(), "best_model.pt")

        # 🔴 ZAPIS
        history["train_loss"].append(total_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc_mean)

        # 🔴 PER TASK HISTORY
        for t in range(num_tasks):
            history["val_auc_tasks"][t].append(val_auc_tasks.get(t, np.nan))

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val AUC (mean): {val_auc_mean:.4f}")
        print("-"*40)

    return history, val_auc_tasks



def map_task_names(tasks, auc_dict):
    return {tasks[i]: auc_dict[i] for i in auc_dict}


@torch.no_grad()
def evaluate_with_metrics(model, dataloader, device, num_tasks):
    model.eval()

    all_preds = [[] for _ in range(num_tasks)]
    all_targets = [[] for _ in range(num_tasks)]

    total_loss = 0.0

    for batch in dataloader:
        batch_graph = batch["graph"]
        rdkit = batch["rdkit"].to(device)
        qc = batch["qc"].to(device)
        qc_mask = batch["qc_mask"].to(device)
        targets = batch["y"].to(device)
        mask = batch["mask"].to(device)

        logits = model(batch_graph, rdkit, qc, qc_mask)
        probs = torch.sigmoid(logits)

        loss, _ = compute_loss(model, logits, targets, mask)
        total_loss += loss.item()

        for t in range(num_tasks):
            m = mask[:, t] == 1
            if m.sum() == 0:
                continue

            all_preds[t].extend(probs[m, t].cpu().numpy())
            all_targets[t].extend(targets[m, t].cpu().numpy())

    # ROC-AUC per task
    auc_per_task = {}
    for t in range(num_tasks):
        if len(set(all_targets[t])) < 2 or len(all_targets[t]) < 10:
            auc_per_task[t] = np.nan
        else:
            auc_per_task[t] = roc_auc_score(all_targets[t], all_preds[t])

    mean_auc = np.nanmean(list(auc_per_task.values()))

    return total_loss, auc_per_task, mean_auc


LR = 1e-4
EPOCHS = 50
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
NUM_TASKS = 13


def main():



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔴 1. LOAD DATA
    data, tasks = load_dataset("dataset_raw.parquet")

    pd.DataFrame({
        "lr": [LR],
        "epochs": [EPOCHS],
        "weight_decay": [WEIGHT_DECAY],
        "tasks": [len(tasks)]
    }).to_csv(os.path.join(BASE_DIR, "config.csv"), index=False)

    # 🔴 2. NORMALIZE
    # split FIRST
    train_data, val_data, test_data = split_data(data)

    # fit scaler tylko na train
    train_data, rdkit_scaler, qc_scaler = normalize_features(
        train_data,
        fit=True
    )

    # apply na val/test
    val_data, _, _ = normalize_features(
        val_data,
        rdkit_scaler,
        qc_scaler,
        fit=False
    )

    test_data, _, _ = normalize_features(
        test_data,
        rdkit_scaler,
        qc_scaler,
        fit=False
    )


    # 🔴 4. DATALOADER
    train_loader = build_dataloader(train_data, batch_size=32)
    val_loader = build_dataloader(val_data, batch_size=32, shuffle=False)

    # 🔴 5. MODEL
    model = QW_MTL(num_tasks=len(tasks)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # 🔴 6. TRAIN
    history, auc_tasks = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        len(tasks)
    )

    auc_named = map_task_names(tasks, auc_tasks)


    beta_values = F.softplus(model.weighting.log_beta).detach().cpu().numpy()
    beta_dict = {tasks[i]: beta_values[i] for i in range(len(tasks))}

    pd.DataFrame({
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_auc": history["val_auc"]
    }).to_csv(os.path.join(BASE_DIR, "training_log.csv"), index=False)

    df_auc = pd.DataFrame.from_dict(history["val_auc_tasks"])
    df_auc.columns = tasks
    df_auc.to_csv(os.path.join(BASE_DIR, "auc_per_task.csv"), index=False)

    pd.DataFrame({
        "task": tasks,
        "auc": [auc_named[t] for t in tasks],
        "beta": [beta_dict[t] for t in tasks]
    }).to_csv(os.path.join(BASE_DIR, "final_metrics.csv"), index=False)

    corr = plot_task_correlation(model, val_loader, tasks, device)
    corr.to_csv(os.path.join(BASE_DIR, "task_correlation.csv"))

    test_loader = build_dataloader(test_data, batch_size=32, shuffle=False)

    model.load_state_dict(torch.load("best_model.pt"))

    test_loss, test_auc_tasks, test_auc_mean = evaluate_with_metrics(
        model, test_loader, device, len(tasks)
    )

    pd.DataFrame({
        "task": tasks,
        "test_auc": [test_auc_tasks[i] for i in range(len(tasks))]
    }).to_csv(os.path.join(BASE_DIR, "test_auc.csv"), index=False)


    plot_global_auc(history)
    plot_auc_per_task(history, tasks)
    plot_final_analysis(auc_named, beta_dict)
    plot_training_history(history)
    plot_metrics_per_task(auc_named)






if __name__ == "__main__":
    main()

