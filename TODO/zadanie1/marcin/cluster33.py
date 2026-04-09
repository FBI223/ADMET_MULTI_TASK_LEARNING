import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# Chemia i Bioinformatyka
from tdc.single_pred import ADME, Tox
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# ML i Modele
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset

# =================================================================
# 1. KONFIGURACJA I KATALOGI
# =================================================================
CACHE_DIR = "feature_cache"
OUTPUT_DIR = "../../../Baseline Multi Task Models/MMoE/experiment_results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================
# 2. EKSTRAKCJA CECH (FEATURES)
# =================================================================
print("Ładowanie modeli językowych (ChemBERTa)...")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model_bert = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(DEVICE)
model_bert.eval()


def clean_nan(X):
    """Zamienia NaN i Inf na zera w formacie float32."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def morgan_fp(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol is None: return np.zeros(1024, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32)


def rdkit_top20(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol is None: return np.zeros(20, dtype=np.float32)
    try:
        vals = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
            Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol), rdMolDescriptors.CalcExactMolWt(mol),
            Descriptors.qed(mol), Descriptors.BalabanJ(mol), Descriptors.BertzCT(mol),
            Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol)
        ]
        return clean_nan(np.array(vals))
    except:
        return np.zeros(20, dtype=np.float32)


def chemberta_embeddings(smiles_list, batch_size=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            outputs = model_bert(**inputs)
            # CLS token embedding
            out.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(out).astype(np.float32)


def build_features(df, mode, split_name):
    """Buduje cechy lub ładuje je z cache."""
    path = os.path.join(CACHE_DIR, f"{mode}_{split_name}_{len(df)}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f: return pickle.load(f)

    print(f"Generowanie cech: {mode} dla {split_name}...")
    smiles = df.smiles.tolist()
    if mode == "morgan":
        X = np.array([morgan_fp(s) for s in smiles])
    elif mode == "chemberta":
        X = chemberta_embeddings(smiles)
    elif mode == "rdkit":
        X = np.array([rdkit_top20(s) for s in smiles])

    X = clean_nan(X)
    with open(path, "wb") as f:
        pickle.dump(X, f)
    return X


# =================================================================
# 3. DANE I KORELACJE
# =================================================================
def load_tdc_tasks(tasks):
    all_df = []
    for t in tasks:
        loader = ADME if t.startswith("cyp") else Tox
        data = loader(name=t)
        splits = data.get_split()
        for sp in ["train", "test"]:
            temp = splits[sp].rename(columns={"Drug": "smiles", "Y": "label"})[["smiles", "label"]].dropna()
            temp["task"], temp["split"] = t, sp
            all_df.append(temp)
    return pd.concat(all_df).reset_index(drop=True)


def label_corr(df, tasks):
    mat = pd.DataFrame(index=tasks, columns=tasks)
    for t1 in tasks:
        for t2 in tasks:
            d1 = df[df.task == t1][["smiles", "label"]]
            d2 = df[df.task == t2][["smiles", "label"]]
            merged = d1.merge(d2, on="smiles")
            mat.loc[t1, t2] = merged.label_x.corr(merged.label_y) if len(merged) > 10 else 0
    return mat.astype(float)


# =================================================================
# 4. ARCHITEKTURA MTL
# =================================================================
class MTLNet(nn.Module):
    def __init__(self, in_dim, n_tasks):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
            for _ in range(n_tasks)
        ])
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x, t=None):
        h = self.shared(x)
        if t is not None:
            # Używane podczas treningu (spłaszczony output dla konkretnych zadań)
            out = torch.zeros(x.size(0), device=x.device)
            for i, head in enumerate(self.heads):
                mask = (t == i)
                if mask.any(): out[mask] = head(h[mask]).squeeze()
            return out, h
        else:
            # Używane do korelacji (wszystkie głowice naraz)
            return [torch.sigmoid(head(h)) for head in self.heads], h


# =================================================================
# 5. PĘTLA EKSPERYMENTU
# =================================================================
def run_combo_experiment(df, tasks, combo, cache):
    combo_name = "+".join(combo)
    print(f"\n=== EKSPERYMENT: {combo_name} ===")

    combo_dir = os.path.join(OUTPUT_DIR, combo_name.replace("+", "_"))
    os.makedirs(combo_dir, exist_ok=True)

    train_df = df[df.split == "train"].copy()
    test_df = df[df.split == "test"].copy()
    task_map = {t: i for i, t in enumerate(tasks)}

    # Przygotowanie macierzy cech
    def get_combined_X(target_df):
        f_list = []
        for m in combo:
            X = cache[m][target_df.index.values]
            if m in ["chemberta", "rdkit"]:
                X = StandardScaler().fit_transform(X)
            f_list.append(X)
        return np.concatenate(f_list, axis=1)

    Xtr = get_combined_X(train_df)
    Xte = get_combined_X(test_df)
    ytr = (train_df.label.values > 0).astype(int)
    yte = (test_df.label.values > 0).astype(int)
    ttr = train_df.task.map(task_map).values
    tte = test_df.task.map(task_map).values

    # --- Trening MTL ---
    model = MTLNet(Xtr.shape[1], len(tasks)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    Xtr_t = torch.tensor(Xtr).to(DEVICE)
    ytr_t = torch.tensor(ytr).float().to(DEVICE)
    ttr_t = torch.tensor(ttr).to(DEVICE)

    for epoch in range(20):
        model.train()
        logits, _ = model(Xtr_t, ttr_t)
        loss = F.binary_cross_entropy_with_logits(logits, ytr_t)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        if epoch % 5 == 0: print(f" Epoch {epoch} Loss: {loss.item():.4f}")

    # --- Ewaluacja ---
    model.eval()
    task_stats = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte).to(DEVICE)
        tte_t = torch.tensor(tte).to(DEVICE)
        all_logits, _ = model(Xte_t, tte_t)
        all_preds = torch.sigmoid(all_logits).cpu().numpy()

    for i, t_name in enumerate(tasks):
        mask = (tte == i)
        auc_mtl = roc_auc_score(yte[mask], all_preds[mask])

        # STL (XGBoost) dla porównania
        mask_tr = (ttr == i)
        clf = XGBClassifier(n_estimators=100, tree_method="hist").fit(Xtr[mask_tr], ytr[mask_tr])
        auc_stl = roc_auc_score(yte[mask], clf.predict_proba(Xte[mask])[:, 1])

        task_stats.append({"task": t_name, "MTL_AUC": auc_mtl, "STL_AUC": auc_stl})

    # Zapis CSV i wykresu korelacji przewidywań
    res_df = pd.DataFrame(task_stats)
    res_df.to_csv(os.path.join(combo_dir, "task_metrics.csv"), index=False)

    plt.figure(figsize=(10, 8))
    with torch.no_grad():
        head_outputs, _ = model(Xte_t)
        preds_mat = torch.cat(head_outputs, dim=1).cpu().numpy()

    pc = pd.DataFrame(preds_mat, columns=tasks).corr()
    sns.heatmap(pc, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Prediction Corr: {combo_name}")
    plt.savefig(os.path.join(combo_dir, "pred_corr.png"))
    plt.close()

    return {"features": combo_name, "MTL_mean": res_df.MTL_AUC.mean(), "STL_mean": res_df.STL_AUC.mean()}


# =================================================================
# 6. MAIN
# =================================================================
def main():
    tasks = ["cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith", "cyp2c19_veith", "cyp1a2_veith"]
    combos = [["morgan"], ["chemberta"], ["rdkit"], ["morgan", "chemberta"], ["morgan", "rdkit"]]

    # 1. Dane
    df = load_tdc_tasks(tasks)

    # 2. Cechy (Precompute)
    cache = {}
    for m in ["morgan", "chemberta", "rdkit"]:
        cache[m] = build_features(df, m, "FULL")

    # 3. Pętla
    summary = []
    for c in combos:
        res = run_combo_experiment(df, tasks, c, cache)
        summary.append(res)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 4. Podsumowanie końcowe
    summary_df = pd.DataFrame(summary)
    summary_df["GAIN"] = summary_df["MTL_mean"] - summary_df["STL_mean"]
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "overall_summary.csv"), index=False)

    # Wykres zbiorczy Gain
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x="features", y="GAIN", palette="magma")
    plt.axhline(0, color='black', linewidth=1)
    plt.title("MTL Performance Gain (Mean AUC)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_gain_plot.png"))
    plt.show()

    # Mapa korelacji danych wejściowych
    lc = label_corr(df, tasks)
    plt.figure(figsize=(10, 8))
    sns.heatmap(lc, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Input Label Correlation")
    plt.savefig(os.path.join(OUTPUT_DIR, "label_correlation.png"))
    plt.show()


if __name__ == "__main__":
    main()