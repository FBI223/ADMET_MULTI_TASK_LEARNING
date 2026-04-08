import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader, Subset
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.data import BatchMolGraph, MoleculeDatapoint
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score


# =========================
# 1. KONFIGURACJA (Musi być spójna z MTL)
# =========================

class CFG:
    batch_size = 64
    lr = 1e-4  # STL często wymaga mniejszego LR niż MTL
    epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patience = 2
    save_dir = "outputs_stl"  # Osobny folder na wyniki STL


class FeatureConfig:
    """Panel sterowania - wybierasz te same cechy co w MTL."""
    USE_DMPNN = True
    USE_MORGAN = True
    USE_RDKIT = True
    USE_QUANTUM = True

    @classmethod
    def get_input_dim(cls, rdkit_dim=200):
        dim = 0
        if cls.USE_DMPNN:   dim += 300
        if cls.USE_MORGAN:  dim += 1024
        if cls.USE_RDKIT:   dim += rdkit_dim
        if cls.USE_QUANTUM: dim += 8
        return dim


os.makedirs(CFG.save_dir, exist_ok=True)
featurizer = SimpleMoleculeMolGraphFeaturizer()


# =========================
# 2. DEFINICJA KLASY (Naprawia AttributeError przy torch.load)
# =========================

class QWDatapoint(MoleculeDatapoint):
    def __init__(self, smiles, targets, mask, rdkit, qc, qc_mask, morgan, split):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
        super().__init__(mol=mol)
        self.y = targets
        self.mask = mask
        self.rdkit = rdkit
        self.qc = qc
        self.qc_mask = qc_mask
        self.morgan = morgan
        self.split = split


# =========================
# 3. MODEL STL I POMOCNICY
# =========================

class QWSTL(nn.Module):
    def __init__(self, rdkit_dim=200):
        super().__init__()

        # 1. Enkoder Grafowy - unikalny dla KAŻDEGO zadania w STL
        if FeatureConfig.USE_DMPNN:
            self.message_passing = BondMessagePassing(d_h=300, depth=3)
            self.agg = MeanAggregation()

        # Obliczanie wymiaru wejściowego (identycznie jak w MTL)
        input_dim = FeatureConfig.get_input_dim(rdkit_dim)

        # 2. Głowa modelu - identyczna architektura (512 -> 256 -> 1)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Spójne z nowym MTL
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, batch):
        features = []

        # Ekstrakcja cech grafowych
        if FeatureConfig.USE_DMPNN:
            h_v = self.message_passing(batch["bmg"])
            z = self.agg(h_v, batch["bmg"].batch)
            features.append(z)

        # Morgan Fingerprints
        if FeatureConfig.USE_MORGAN:
            features.append(batch["morgan"])

        # RDKit
        if FeatureConfig.USE_RDKIT:
            features.append(batch["rdkit"])

        # Quantum
        if FeatureConfig.USE_QUANTUM:
            features.append(batch["qc"])
            features.append(batch["qc_mask"])

        # Konkatenacja
        x = torch.cat(features, dim=1)

        return self.head(x)

def collate_fn(batch):
    """Modularne przygotowanie batcha."""
    output = {}
    output["targets"] = torch.from_numpy(np.stack([d.y for d in batch])).float()
    output["mask"] = torch.from_numpy(np.stack([d.mask for d in batch])).float()

    if FeatureConfig.USE_DMPNN:
        output["bmg"] = BatchMolGraph([featurizer(d.mol) for d in batch])
    if FeatureConfig.USE_MORGAN:
        output["morgan"] = torch.from_numpy(np.stack([d.morgan for d in batch])).float()
    if FeatureConfig.USE_RDKIT:  # POPRAWNIE: wszystko dużymi literami
        output["rdkit"] = torch.from_numpy(np.stack([d.rdkit for d in batch])).float()
    if FeatureConfig.USE_QUANTUM:
        output["qc"] = torch.from_numpy(np.stack([d.qc for d in batch])).float()
        output["qc_mask"] = torch.from_numpy(np.stack([d.qc_mask for d in batch])).float()
    return output


def save_stl_results(df_results):
    csv_path = os.path.join(CFG.save_dir, "stl_results.csv")
    yml_path = os.path.join(CFG.save_dir, "stl_metrics.yml")

    # Lista zadań wymagających AUPRC
    auprc_tasks = ["CYP2C9 Substrate", "CYP2D6 Substrate", "CYP3A4 Inhibition", "CYP2C9 Inhibition", "CYP2D6 Inhibition"]

    results_dict = {}
    for _, row in df_results.iterrows():
        # Decydujemy, która metryka jest wiodąca dla danego zadania
        is_auprc = any(x in row['Task'] for x in auprc_tasks)
        metric_name = "AUPRC" if is_auprc else "ROC-AUC"
        final_score = row['STL_PRC_AUC'] if is_auprc else row['STL_ROC_AUC']

        results_dict[row['Task']] = {
            "score": float(final_score),
            "metric": metric_name,
            "samples": int(row['Samples'])
        }

    # Zapis CSV
    df_results.to_csv(csv_path, index=False)

    # Zapis YAML w formacie identycznym jak MTL
    final_yaml = {
        "metadata": {
            "model_type": "STL-Deep-Optimized",
            "mean_performance": float(df_results['STL_ROC_AUC'].mean()), # Tymczasowa średnia
            "features": {
                "DMPNN": FeatureConfig.USE_DMPNN,
                "Morgan": FeatureConfig.USE_MORGAN,
                "Quantum": FeatureConfig.USE_QUANTUM
            }
        },
        "tasks": results_dict
    }

    with open(yml_path, 'w') as f:
        yaml.dump(final_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n[SAVE] Wyniki STL zapisane: {yml_path}")

# =========================
# 4. GŁÓWNA PĘTLA TRENINGOWA
# =========================

def train_one_task(task_idx, task_name, train_data, val_data, rdkit_dim):
    """
    Trenuje pojedynczy model STL dla konkretnego zadania ADMET.
    Naprawiono błąd 0-d array oraz dodano mechanizm Early Stopping.
    """
    print(f"\n[STL] INICJALIZACJA: {task_name}")

    # Wybór indeksów, dla których dany task ma etykietę (mask == 1)
    train_indices = [i for i, d in enumerate(train_data) if d.mask[task_idx] == 1]
    val_indices = [i for i, d in enumerate(val_data) if d.mask[task_idx] == 1]

    if len(train_indices) < 10:
        print(f"   ! POMINIĘTO: {task_name} - niewystarczająca liczba danych ({len(train_indices)})")
        return 0.0, 0.0, len(train_indices), np.zeros(len(val_indices))

    # Przygotowanie loaderów dla podzbioru danych
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=CFG.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=CFG.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Inicjalizacja modelu STL (unikalny enkoder i głowa)
    model = QWSTL(rdkit_dim).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_ap = 0.0
    best_preds = []
    patience_counter = 0

    for epoch in range(CFG.epochs):
        model.train()
        for batch in train_loader:
            for k in batch:
                if k == "bmg":
                    batch[k].to(CFG.device)
                else:
                    batch[k] = batch[k].to(CFG.device)

            optimizer.zero_grad()
            # Używamy .view(-1), aby uniknąć błędów przy batch_size=1
            logits = model(batch).view(-1)
            loss = criterion(logits, batch["targets"][:, task_idx])
            loss.backward()
            optimizer.step()

        # Ewaluacja
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    if k == "bmg":
                        batch[k].to(CFG.device)
                    else:
                        batch[k] = batch[k].to(CFG.device)

                # Bezpieczne wyciąganie predykcji
                logits = model(batch).view(-1)
                y_true.extend(batch["targets"][:, task_idx].cpu().numpy())
                y_pred.extend(torch.sigmoid(logits).cpu().numpy())

        # Obliczanie metryk
        try:
            auc_val = roc_auc_score(y_true, y_pred)
            ap_val = average_precision_score(y_true, y_pred)
        except ValueError:
            auc_val, ap_val = 0.5, 0.0

        # Logika zapisu najlepszego wyniku (Early Stopping)
        if auc_val > best_auc:
            best_auc = auc_val
            best_ap = ap_val
            best_preds = y_pred  # Zapamiętujemy predykcje dla PLOTS
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CFG.patience:
            print(f"   > Early stopping (Epoka {epoch}) | Best AUC: {best_auc:.4f}")
            break

    return best_auc, best_ap, len(train_indices), np.array(best_preds)
# =========================
# 5. CALY MAIN FUNCTION
# =========================

def main():
    # A. Wczytanie Cache
    cache_path = os.path.join("outputs_mtl", "preprocessed_data_cache.pt")
    print(f"[START] Wczytywanie cache: {cache_path}")

    if not os.path.exists(cache_path):
        print(f"[ERROR] Nie znaleziono pliku {cache_path}! Uruchom najpierw MTL.")
        return

    cache = torch.load(cache_path)
    data = cache["data"]
    tasks = cache["tasks"]
    rdkit_dim = len(cache["rdkit_cols"])

    # B. Podział danych
    train_data = [d for d in data if d.split == "train"]
    val_data = [d for d in data if d.split == "valid"]
    print(f"[INFO] Dane: Train={len(train_data)}, Val={len(val_data)}")

    # Macierze do zbierania wyników dla modułu PLOTS
    all_val_preds = np.zeros((len(val_data), len(tasks)))
    all_val_targets = np.zeros((len(val_data), len(tasks)))
    all_val_masks = np.zeros((len(val_data), len(tasks)))

    results = []
    print(f"[RUN] Rozpoczynanie treningu STL dla {len(tasks)} zadań...")

    # C. Pętla po zadaniach
    for i, t_name in enumerate(tasks):
        # Trening modelu dla pojedynczego zadania
        auc, ap, n_samples, task_preds = train_one_task(i, t_name, train_data, val_data, rdkit_dim)

        # Mapowanie predykcji do macierzy zbiorczej (tylko dla valid_indices tego zadania)
        val_indices = [idx for idx, d in enumerate(val_data) if d.mask[i] == 1]
        all_val_preds[val_indices, i] = task_preds
        all_val_targets[val_indices, i] = [val_data[idx].y[i] for idx in val_indices]
        all_val_masks[val_indices, i] = 1

        # Wybór metryki do wyświetlenia (AUPRC dla CYP, AUC dla reszty)
        auprc_tasks = ["CYP2C9 Substrate", "CYP2D6 Substrate", "CYP3A4 Inhibition", "CYP2C9 Inhibition",
                       "CYP2D6 Inhibition"]
        is_auprc = any(x in t_name for x in auprc_tasks)
        score_display = ap if is_auprc else auc
        metric_name = "AUPRC" if is_auprc else "ROC-AUC"

        print(f"   > FINISH {t_name:<20} | {metric_name}: {score_display:.4f} | N: {n_samples}")

        results.append({
            "Task": t_name,
            "STL_ROC_AUC": float(auc),
            "STL_PRC_AUC": float(ap),
            "Samples": int(n_samples)
        })

    # D. Zapis wyników (CSV i YAML)
    df_results = pd.DataFrame(results)
    save_stl_results(df_results)

    # E. Generowanie Wykresów (Moduł PLOTS)
    print(f"\n[PLOTS] Generowanie wizualizacji zbiorczych dla STL...")
    import PLOTS

    # Konwersja na tensory dla kompatybilności z Twoim modułem PLOTS
    t_preds = torch.tensor(all_val_preds)
    t_targets = torch.tensor(all_val_targets)
    t_masks = torch.tensor(all_val_masks)

    PLOTS.plot_roc_combined(t_preds, t_targets, t_masks, tasks, f"{CFG.save_dir}/roc_combined_stl.png")
    PLOTS.plot_task_correlation(t_preds, t_targets, t_masks, tasks, f"{CFG.save_dir}/task_correlation_stl.png")
    PLOTS.plot_pr(t_preds, t_targets, t_masks, tasks, CFG.save_dir)

    print("\n" + "=" * 45)
    print(f"ŚREDNI STL ROC-AUC: {df_results['STL_ROC_AUC'].mean():.4f}")
    print(f"Wszystkie wyniki i wykresy zapisano w: {CFG.save_dir}")
    print("=" * 45)


if __name__ == "__main__":
    main()