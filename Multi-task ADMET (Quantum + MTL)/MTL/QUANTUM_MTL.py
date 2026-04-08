import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from chemprop.nn import BondMessagePassing
from torch.utils.data import Dataset
from chemprop.nn import MeanAggregation
from rdkit import Chem
from chemprop.data import MoleculeDatapoint
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit.Chem import rdFingerprintGenerator
import yaml  # Pamiętaj o eksporcie na górze pliku





# Inicjalizacja featurizera, który zamieni molekuły RDKit na grafy molekularne.
# W Chemprop 2.2.3 to serce D-MPNN (Directed Message Passing)[cite: 122, 424].
featurizer = SimpleMoleculeMolGraphFeaturizer()

# =========================
# CONFIG
# =========================

class CFG:
    """Konfiguracja hiperparametrów treningu zgodnie z badaniami QW-MTL[cite: 168]."""
    data_path = "dataset_raw.parquet"
    batch_size = 64
    lr = 5e-4
    epochs = 10
    patience = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "outputs_mtl"

class FeatureConfig:
    """Panel sterowania modułami wejściowymi."""
    USE_DMPNN = True   # d=300 (Shared Encoder z grafu)
    USE_MORGAN = False  # d=1024 (Fingerprints)
    USE_RDKIT = True
    USE_QUANTUM = True

    @classmethod
    def get_input_dim(cls, rdkit_dim=200):
        dim = 0
        if cls.USE_DMPNN:   dim += 300
        if cls.USE_MORGAN:  dim += 1024
        if cls.USE_RDKIT:   dim += rdkit_dim
        if cls.USE_QUANTUM: dim += 8 # 4 (cechy) + 4 (maska)
        return dim


os.makedirs(CFG.save_dir, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# =========================
# DATASET
# =========================

class SimpleDataset(Dataset):
    """Prosty wrapper PyTorch na listę obiektów QWDatapoint."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class QWDatapoint(MoleculeDatapoint):
    def __init__(self, mol, targets, mask, rdkit, qc, qc_mask, morgan, split):
        super().__init__(mol=mol)

        self.y = targets
        self.mask = mask
        self.rdkit = rdkit
        self.qc = qc
        self.qc_mask = qc_mask
        self.morgan = morgan  # Nowe: Morgan Fingerprint
        self.split = split


def build_dataset(df):
    """
    Zoptymalizowana wersja: modernizuje Fingerprinty, usuwa wąskie gardła
    oraz eliminuje DATA LEAKAGE poprzez skalowanie oparte tylko na zbiorze TRAIN.
    """
    # 1. Identyfikacja zadań i kolumn
    tasks = sorted(df["task"].unique())
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    rdkit_cols = [c for c in df.columns if c not in [
        "smiles", "label", "task", "split", "success",
        "dipole", "homo_lumo", "electrons", "energy",
        "mask_dipole", "mask_homo_lumo", "mask_electrons", "mask_energy"
    ]]

    qc_cols = ["dipole", "homo_lumo", "electrons", "energy"]
    qc_mask_cols = ["mask_dipole", "mask_homo_lumo", "mask_electrons", "mask_energy"]

    # --- NOWA SEKCJA: SKALOWANIE BEZ WYCIEKU DANYCH ---
    # Definiujemy train_df do dopasowania (fit) skalerów
    train_df = df[df['split'] == 'train']

    scaler_rdkit = None
    if FeatureConfig.USE_RDKIT:
        # Przygotowanie danych (logarytm dla Ipc i czyszczenie)
        if 'Ipc' in rdkit_cols:
            df['Ipc'] = np.log1p(df['Ipc'])
        df[rdkit_cols] = df[rdkit_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler_rdkit = StandardScaler()
        # FIT tylko na danych treningowych
        scaler_rdkit.fit(train_df[rdkit_cols].values.astype(np.float64))
        # TRANSFORM na całym DataFrame
        df[rdkit_cols] = scaler_rdkit.transform(df[rdkit_cols].values.astype(np.float64)).astype(np.float32)

    scaler_qc = None
    if FeatureConfig.USE_QUANTUM:
        scaler_qc = StandardScaler()
        # FIT tylko na udanych obliczeniach (success == 1) ze zbioru treningowego
        success_train = train_df[train_df["success"] == 1]
        if not success_train.empty:
            scaler_qc.fit(success_train[qc_cols].values.astype(np.float64))
            # TRANSFORM na wszystkim (uzupełniamy braki zerami przed transformacją)
            df[qc_cols] = scaler_qc.transform(df[qc_cols].fillna(0).values.astype(np.float64)).astype(np.float32)
        else:
            df[qc_cols] = df[qc_cols].fillna(0).astype(np.float32)

    # --- GENERATOR FINGERPRINTÓW I PĘTLA ---
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    print(f"[INFO] Przetwarzanie {len(df)} molekuł (Skalowanie zakończone)...")

    data = []
    for _, row in df.iterrows():
        # Etykiety i maska
        y = np.zeros(len(tasks), dtype=np.float32)
        m = np.zeros(len(tasks), dtype=np.float32)
        t_idx = task_to_idx[row["task"]]
        y[t_idx] = row["label"]
        m[t_idx] = 1

        # Obiekt mol (parsowany tylko raz)
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue

            # Morgan Fingerprint
        morgan_fp = None
        if FeatureConfig.USE_MORGAN:
            morgan_fp = mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32)

        # Tworzenie punktu danych
        data.append(QWDatapoint(
            mol=mol,
            targets=y,
            mask=m,
            rdkit=row[rdkit_cols].values.astype(np.float32) if FeatureConfig.USE_RDKIT else None,
            qc=row[qc_cols].values.astype(np.float32) if FeatureConfig.USE_QUANTUM else None,
            qc_mask=row[qc_mask_cols].values.astype(np.float32) if FeatureConfig.USE_QUANTUM else None,
            morgan=morgan_fp,
            split=row["split"]
        ))

    return data, tasks, rdkit_cols, scaler_rdkit, scaler_qc

def split_data(data):
    """Dzieli dane zgodnie z oficjalnym podziałem 'leaderboard-style scaffold split'[cite: 12, 161]."""
    train = [d for d in data if d.split == "train"]
    val   = [d for d in data if d.split == "valid"]
    test  = [d for d in data if d.split == "test"]

    print(f"[SPLIT] train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def compute_metrics(preds, targets, mask, tasks):
    """Oblicza AUROC/AUPRC zgodnie z leaderboardem TDC."""
    preds = torch.sigmoid(preds).detach().cpu().numpy()
    targets, mask = targets.cpu().numpy(), mask.cpu().numpy()

    auprc_tasks = ["CYP2C9 Substrate", "CYP2D6 Substrate", "CYP3A4 Inhibition", "CYP2C9 Inhibition",
                   "CYP2D6 Inhibition"]
    scores = {}

    for i, t_name in enumerate(tasks):
        valid = mask[:, i] == 1
        if valid.sum() < 2: continue

        if any(x in t_name for x in auprc_tasks):
            scores[t_name] = average_precision_score(targets[valid, i], preds[valid, i])
        else:
            scores[t_name] = roc_auc_score(targets[valid, i], preds[valid, i])

    return scores, np.nanmean(list(scores.values()))


# =========================
# COLLATE
# =========================

def collate_fn(batch):
    """Agreguje dane w batche, przesyłając tylko aktywne cechy."""
    output = {}

    # Zawsze potrzebujemy etykiet i masek zadań
    output["targets"] = torch.from_numpy(np.stack([d.y for d in batch])).float()
    output["mask"] = torch.from_numpy(np.stack([d.mask for d in batch])).float()

    # Moduł Grafowy (D-MPNN)
    if FeatureConfig.USE_DMPNN:
        mol_graphs = [featurizer(d.mol) for d in batch]
        output["bmg"] = BatchMolGraph(mol_graphs)

    # Moduł Morgan Fingerprints
    if FeatureConfig.USE_MORGAN:
        output["morgan"] = torch.from_numpy(np.stack([d.morgan for d in batch])).float()

    # Moduł RDKit (opcjonalnie z cache)
    if FeatureConfig.USE_RDKIT:
        output["rdkit"] = torch.from_numpy(np.stack([d.rdkit for d in batch])).float()

    # Moduł Quantum (opcjonalnie z cache)
    if FeatureConfig.USE_QUANTUM:
        output["qc"] = torch.from_numpy(np.stack([d.qc for d in batch])).float()
        output["qc_mask"] = torch.from_numpy(np.stack([d.qc_mask for d in batch])).float()

    return output

def collate_fn_old(batch):
    """
    Agreguje pojedyncze punkty danych w batche.
    Tutaj zachodzi featuryzacja grafu molekularnego (D-MPNN)[cite: 111, 419].
    """
    mols = [d.mol for d in batch]

    # Sanity check: upewnienie się, że RDKit poprawnie przetworzył SMILES
    for i, m in enumerate(mols[:3]):
        if m is None:
            print(f"[ERROR] mol {i} is None!")

    # Zamiana molekuł na skumulowany graf BatchMolGraph (format wejściowy dla BondMessagePassing)[cite: 122].
    mol_graphs = [featurizer(m) for m in mols]
    bmg = BatchMolGraph(mol_graphs)

    # Konwersja pozostałych cech na tensory PyTorch
    targets = torch.from_numpy(np.stack([d.y for d in batch])).float()
    mask = torch.from_numpy(np.stack([d.mask for d in batch])).float()
    rdkit = torch.from_numpy(np.stack([d.rdkit for d in batch])).float()
    qc = torch.from_numpy(np.stack([d.qc for d in batch])).float()
    qc_mask = torch.from_numpy(np.stack([d.qc_mask for d in batch])).float()
    # Dodajemy zbieranie Morgan Fingerprints
    morgan = torch.from_numpy(np.stack([d.morgan for d in batch])).float()


    return {
        "bmg": bmg,          # Dane dla enkodera D-MPNN (kierunkowe)[cite: 122].
        "targets": targets,  # Etykiety 13 zadań.
        "mask": mask,        # Maska dla wagowania strat[cite: 142].
        "rdkit": rdkit,      # Globalne cechy 2D (200-dim)[cite: 123].
        "qc": qc,            # Cechy kwantowe (4-dim).
        "qc_mask": qc_mask,   # Maska braków QC (4-dim)[cite: 136].
        "morgan": morgan  # Nowy klucz w słowniku
    }



# =========================
# MODEL
# =========================

class QWMTL(nn.Module):
    def __init__(self, num_tasks, rdkit_dim=200):
        super().__init__()

        # 1. Inicjalizacja Enkodera Grafowego
        if FeatureConfig.USE_DMPNN:
            self.message_passing = BondMessagePassing(d_h=300, depth=3)
            self.agg = MeanAggregation()

        # 2. Obliczanie wymiaru wejściowego dla głów (Heads)
        input_dim = FeatureConfig.get_input_dim(rdkit_dim)

        # 3. Wielozadaniowe głowy predykcyjne
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),  # Dodaj Batch Normalization dla stabilności
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(num_tasks)
        ])

        self.log_beta = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, batch, return_embedding=False):
        features = []

        # Ekstrakcja cech grafowych
        if FeatureConfig.USE_DMPNN:
            h_v = self.message_passing(batch["bmg"])
            z_graph = self.agg(h_v, batch["bmg"].batch)
            features.append(z_graph)

        # Ekstrakcja Morgan Fingerprints
        if FeatureConfig.USE_MORGAN:
            features.append(batch["morgan"])

        # Pozostałe moduły (jeśli aktywne)
        if FeatureConfig.USE_RDKIT:
            features.append(batch["rdkit"])
        if FeatureConfig.USE_QUANTUM:
            features.append(batch["qc"])
            features.append(batch["qc_mask"])

        # Łączenie cech (Concatenation)
        combined_x = torch.cat(features, dim=1)

        if return_embedding:
            return None, combined_x

        # Predykcja dla każdego zadania
        return torch.cat([h(combined_x) for h in self.heads], dim=1)


class QWMTL_OLD(nn.Module):
    def __init__(self, num_tasks, rdkit_dim=200):
        super().__init__()
        # Zgodnie z publikacją: d_h=300 dla D-MPNN
        self.message_passing = BondMessagePassing(d_h=300, depth=3)
        self.agg = MeanAggregation()

        # Upewnij się, że suma to dokładnie 508
        self.input_dim = 300 + rdkit_dim + 4 + 4

        # FFN: w artykule wspomniano o warstwach FFN dla każdego zadania [cite: 139, 152]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 512),  # Zwiększenie szerokości dla złożonych ADMET
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1)
            ) for _ in range(num_tasks)
        ])

        # Inicjalizacja log_beta na 0 sprawia, że beta=softplus(0) ok. 0.69 [cite: 187]
        self.log_beta = nn.Parameter(torch.randn(num_tasks) * 0.1)
        #self.log_beta = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, batch):
        # Pobranie grafu molekularnego (BatchMolGraph)
        bmg = batch["bmg"]

        # KROK 1: Generowanie kierunkowych cech ukrytych (D-MPNN)
        h_v = self.message_passing(bmg)

        # KROK 2: Agregacja do postaci wektora (Fingerprint)
        z = self.agg(h_v, bmg.batch)

        # KROK 3: Konkatenacja cech fizykochemicznych i kwantowych
        # Łączenie informacji 2D (RDKit) i 3D (Quantum) w jeden wektor 508-dim.
        x = torch.cat([z, batch["rdkit"], batch["qc"], batch["qc_mask"]], dim=1)

        # KROK 4: Równoległa predykcja dla wszystkich 13 zadań ADMET
        return torch.cat([h(x) for h in self.heads], dim=1)


# =========================
# LOSS
# =========================

def qw_loss(preds, targets, mask, log_beta):
    """Implementacja Adaptive Task Weighting Scheme [cite: 11, 139]"""
    losses = []
    n_t = []

    for t in range(preds.shape[1]):
        valid = mask[:, t] == 1
        if valid.sum() == 0:
            losses.append(torch.tensor(0.0, device=preds.device))
            n_t.append(torch.tensor(0.0, device=preds.device))
            continue

        # Standardowy BCE dla ADMET classification [cite: 151, 156]
        loss_t = F.binary_cross_entropy_with_logits(preds[valid, t], targets[valid, t])
        losses.append(loss_t)
        n_t.append(valid.sum().float()) # Liczba etykiet n_t [cite: 143]

    losses = torch.stack(losses)
    n_t = torch.stack(n_t)

    # Równanie (1): Obliczanie r_t (sample proportion) [cite: 143-146]
    r_t = n_t / (n_t.sum() + 1e-8)

    # Równanie (2): Obliczanie wagi w_t przy użyciu softplus [cite: 149]
    # w_t = r_t^beta_t, gdzie beta_t = softplus(log_beta_t)
    beta = F.softplus(log_beta)
    w_t = r_t ** beta

    # Równanie (3): Suma ważonych strat [cite: 152]
    return (w_t * losses).sum()


# =========================
# METRICS
# =========================

def compute_auc(preds, targets, mask):
    """Oblicza AUROC dla każdego zadania niezależnie, ignorując brakujące etykiety[cite: 166]."""
    preds = torch.sigmoid(preds).detach().cpu().numpy()
    targets = targets.cpu().numpy()
    mask = mask.cpu().numpy()

    aucs = []

    for t in range(preds.shape[1]):
        valid = mask[:, t] == 1
        # Wymagane minimum 10 próbek, aby wynik AUC był statystycznie sensowny.
        if valid.sum() < 10:
            aucs.append(np.nan)
            continue

        try:
            # Obliczanie AUC-ROC dla konkretnego zadania (np. HIA czy BBB)[cite: 170].
            auc = roc_auc_score(targets[valid, t], preds[valid, t])
        except:
            auc = np.nan

        aucs.append(auc)

    # Zwraca listę wyników per-task oraz średnią (Mean AUC).
    return aucs, np.nanmean(aucs)


# =========================
# TRAIN LOOP
# =========================

def run_epoch(model, loader, tasks, optimizer=None):
    preds_all, targets_all, mask_all = [], [], []
    total_loss = 0

    for batch in loader:
        batch["bmg"].to(CFG.device)
        for k in batch:
            if k == "bmg":
                batch[k].to(CFG.device)
            else:
                batch[k] = batch[k].to(CFG.device)

        preds = model(batch)
        loss = qw_loss(preds, batch["targets"], batch["mask"], model.log_beta)

        #if optimizer:
        #    optimizer.zero_grad()
        #    loss.backward()
        #    optimizer.step()

        if optimizer:
            model.train()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            model.eval()

        total_loss += loss.item()
        preds_all.append(preds.detach())
        targets_all.append(batch["targets"])
        mask_all.append(batch["mask"])

    preds = torch.cat(preds_all)
    targets = torch.cat(targets_all)
    mask = torch.cat(mask_all)

    # Obliczanie metryk specyficznych dla każdego zadania (AUROC lub AUPRC) [cite: 208]
    task_scores, mean_score = compute_metrics(preds, targets, mask, tasks)

    return total_loss, mean_score, task_scores


def save_detailed_results(task_scores, mean_score, cfg, feature_cfg, filename_prefix="mtl"):
    """Zapisuje wyniki per-task do CSV i YAML w formacie spójnym z STL."""

    csv_path = os.path.join(cfg.save_dir, f"{filename_prefix}_results.csv")
    yml_path = os.path.join(cfg.save_dir, f"{filename_prefix}_metrics.yml")

    # Lista zadań wymagających AUPRC (zgodnie z protokołem TDC)
    auprc_tasks = ["CYP2C9 Substrate", "CYP2D6 Substrate", "CYP3A4 Inhibition", "CYP2C9 Inhibition",
                   "CYP2D6 Inhibition"]

    # 1. Przygotowanie danych do CSV i słownika YAML
    rows = []
    yaml_tasks = {}

    for task_name, score in task_scores.items():
        metric_name = "AUPRC" if any(x in task_name for x in auprc_tasks) else "ROC-AUC"

        # Dane do tabeli CSV
        rows.append({
            "Task": task_name,
            "Metric": metric_name,
            "Score": float(score)
        })

        # Dane do YAML (klucz 'score' ułatwi porównanie z STL_ROC_AUC/STL_PRC_AUC)
        yaml_tasks[task_name] = {
            "score": float(score),
            "metric": metric_name
        }

    # Zapis CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # 2. Budowa finalnego YAML
    yaml_data = {
        "metadata": {
            "model_type": "QW-MTL (Deep Learning)",
            "mean_performance": float(mean_score),
            "features_used": {
                "DMPNN": feature_cfg.USE_DMPNN,
                "Morgan": feature_cfg.USE_MORGAN,
                "RDKit": feature_cfg.USE_RDKIT,
                "Quantum": feature_cfg.USE_QUANTUM
            },
            "parameters": {
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "epochs_trained": cfg.epochs
            }
        },
        "tasks": yaml_tasks  # Klucz 'tasks' identyczny jak w STL_BASELINE
    }

    with open(yml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"[SAVE] Wyniki MTL zapisane: {yml_path}")

# =========================
# MAIN
# =========================

def main():
    # 1. Ścieżka do cache'u
    cache_path = os.path.join(CFG.save_dir, "preprocessed_data_cache.pt")

    # 2. Ładowanie lub Generowanie danych
    if os.path.exists(cache_path):
        print(f"[CACHE] Wczytywanie wszystkich cech z {cache_path}...")
        cache = torch.load(cache_path)
        data = cache["data"]
        tasks = cache["tasks"]
        rdkit_cols = cache["rdkit_cols"]
        # Skalery są ładowane, aby zachować spójność [cite: 168]
        sc_rdkit = cache.get("sc_rdkit")
        sc_qc = cache.get("sc_qc")
    else:
        print("[CACHE] Brak cache. Przetwarzanie surowych danych...")
        df = pd.read_parquet(CFG.data_path)

        # build_dataset musi zwracać 5 wartości, aby pasować do Twojego unpackingu
        # data, tasks, rdkit_cols, sc_rdkit, sc_qc
        data, tasks, rdkit_cols, sc_rdkit, sc_qc = build_dataset(df)

        print(f"[CACHE] Zapisywanie wszystkich obliczonych cech do {cache_path}...")
        torch.save({
            "data": data,
            "tasks": tasks,
            "rdkit_cols": rdkit_cols,
            "sc_rdkit": sc_rdkit,
            "sc_qc": sc_qc
        }, cache_path)

    # 3. Podział scaffold-based [cite: 12, 161]
    train, val, test = split_data(data)

    # 4. DataLoader (collate_fn musi teraz wybierać cechy na podstawie FeatureConfig)
    train_loader = DataLoader(SimpleDataset(train), batch_size=CFG.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SimpleDataset(val), batch_size=CFG.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # 5. Inicjalizacja modelu (dynamiczny wymiar wejściowy)
    model = QWMTL(len(tasks), rdkit_dim=len(rdkit_cols)).to(CFG.device)

    # 6. Optymalizacja [cite: 168]
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    history = {
        "train_loss": [], "val_loss": [],
        "train_score": [], "val_score": []
    }

    best_score = 0
    patience = 0

    # Główna pętla treningowa: wspólna optymalizacja 13 zadań ADMET [cite: 30, 116]
    for epoch in range(CFG.epochs):
        # run_epoch implementuje dynamiczne wagowanie strat (Adaptive Task Weighting) [cite: 115, 140, 152]
        train_loss, train_score, train_task_scores = run_epoch(model, train_loader, tasks, optimizer)
        val_loss, val_score, val_task_scores = run_epoch(model, val_loader, tasks)

        scheduler.step(val_score)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_score"].append(train_score)
        history["val_score"].append(val_score)

        print(f"[EPOCH {epoch}] Loss: {val_loss:.4f} | Mean Val Score: {val_score:.4f}")

        # Zapis najlepszego modelu na podstawie średniej wydajności [cite: 43, 170]
        if val_score > best_score:
            best_score = val_score
            patience = 0

            torch.save(model.state_dict(), f"{CFG.save_dir}/qw_mtl_best.pt")

            # --- NOWE: Zapis do CSV i YML ---
            save_detailed_results(
                task_scores=val_task_scores,
                mean_score=val_score,
                cfg=CFG,
                feature_cfg=FeatureConfig
            )

            # 7. ZBIERANIE PREDYKCJI (NAPRAWIONY BŁĄD .to()) [cite: 419]
            best_preds, best_targets, best_mask = [], [], []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    # KRYTYCZNA POPRAWKA: .to() dla BatchMolGraph modyfikuje w miejscu [cite: 419]
                    batch["bmg"].to(CFG.device)
                    for k in batch:
                        if k != "bmg":
                            batch[k] = batch[k].to(CFG.device)

                    preds = model(batch)
                    best_preds.append(preds.cpu())
                    best_targets.append(batch["targets"].cpu())
                    best_mask.append(batch["mask"].cpu())

            best_preds = torch.cat(best_preds)
            best_targets = torch.cat(best_targets)
            best_mask = torch.cat(best_mask)

            # 8. ANALIZA I WIZUALIZACJA (PCA/t-SNE/Beta) [cite: 218, 275, 395]
            from PLOTS import (
                plot_task_correlation, plot_roc_combined, plot_pr,
                plot_beta_vs_samples, plot_pca, plot_tsne
            )

            # Wizualizacje korelacji i krzywych wydajności [cite: 170, 366]
            plot_task_correlation(best_preds, best_targets, best_mask, tasks, f"{CFG.save_dir}/task_corr.png")
            plot_roc_combined(best_preds, best_targets, best_mask, tasks, CFG.save_dir)
            plot_pr(best_preds, best_targets, best_mask, tasks, CFG.save_dir)

            # Weryfikacja korelacji beta z rozmiarem danych (cel: r=0.95) [cite: 218]
            plot_beta_vs_samples(model, train, tasks, f"{CFG.save_dir}/beta_vs_samples.png")

            # Analiza przestrzeni ukrytej (508-dim embeddings) [cite: 274, 386]
            embeddings = []
            with torch.no_grad():
                for batch in val_loader:
                    batch["bmg"].to(CFG.device)
                    for k in batch:
                        if k != "bmg":
                            batch[k] = batch[k].to(CFG.device)

                    _, z = model(batch, return_embedding=True)
                    embeddings.append(z.cpu())

            embeddings = torch.cat(embeddings).numpy()
            labels = best_targets.argmax(1).numpy()

            plot_pca(embeddings, labels, tasks, f"{CFG.save_dir}/pca.png")
            plot_tsne(embeddings, labels, tasks, f"{CFG.save_dir}/tsne.png")

        else:
            patience += 1
            if patience >= CFG.patience:
                print(f"[EARLY STOP] Brak poprawy przez {CFG.patience} epok.")
                break

    # 9. FINALNE WYKRESY UCZENIA (Loss i Mean Score) [cite: 116]
    from PLOTS import plot_training
    plot_training(history, f"{CFG.save_dir}/learning_curve.png")

if __name__ == "__main__":
    main()

