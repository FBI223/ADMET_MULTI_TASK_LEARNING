
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
from tdc.single_pred import ADME, Tox
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel

# =================================================================
# 1. KONFIGURACJA I KATALOGI
# =================================================================
CACHE_DIR = "../../TODO/zadanie1/marcin/feature_cache"
OUTPUT_DIR = "experiment_results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {DEVICE}")

# Inicjalizacja modeli ChemBERTa
print("Ładowanie modeli językowych (ChemBERTa)...")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model_bert = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(DEVICE)
model_bert.eval()



TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]

TASKS_TOX = ['hERG_Karim', 'ames', 'dili']

ALL_TASKS = TASKS_ADME + TASKS_TOX


# =================================================================
# 2. FUNKCJE EKSTRAKCJI CECH (FEATURES)
# =================================================================

def clean_nan(X):
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
            out.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
            del inputs, outputs
    return np.vstack(out).astype(np.float32)


def build_features(df, mode, split_name):
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

def load_tdc_tasks(tasks_adme, tasks_tox):
    all_df = []

    # ADME
    for t in tasks_adme:
        data = ADME(name=t)
        splits = data.get_split()

        for sp in ["train", "test"]:
            temp = splits[sp].rename(columns={"Drug": "smiles", "Y": "label"})
            temp = temp[["smiles", "label"]].dropna()
            temp["task"], temp["split"] = t, sp
            all_df.append(temp)

    # TOX
    for t in tasks_tox:
        data = Tox(name=t)
        splits = data.get_split()

        for sp in ["train", "test"]:
            temp = splits[sp].rename(columns={"Drug": "smiles", "Y": "label"})
            temp = temp[["smiles", "label"]].dropna()
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
# 4. ARCHITEKTURA MMOE I STRATA
# =================================================================

class MMoE_MTLNet(nn.Module):
    def __init__(self, in_dim, n_tasks, n_experts=4, expert_dim=256):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_dim),
                nn.ReLU(),
                nn.BatchNorm1d(expert_dim),
                nn.Dropout(0.2),
                nn.Linear(expert_dim, 256),
                nn.ReLU()
            ) for _ in range(n_experts)

        ])

        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, n_experts), nn.Softmax(dim=-1))
            for _ in range(n_tasks)
        ])

        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            for _ in range(n_tasks)
        ])

        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x, t=None):
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)
        if t is not None:
            final_output = torch.zeros(x.size(0), device=x.device)
            for i in range(self.n_tasks):
                mask = (t == i)
                if mask.any():
                    gate_weights = self.gates[i](x[mask]).unsqueeze(1)
                    task_rep = torch.bmm(gate_weights, expert_outputs[mask]).squeeze(1)
                    final_output[mask] = self.heads[i](task_rep).squeeze()
            return final_output, None
        else:
            all_preds = []
            for i in range(self.n_tasks):
                gate_weights = self.gates[i](x).unsqueeze(1)
                task_rep = torch.bmm(gate_weights, expert_outputs).squeeze(1)
                #all_preds.append(torch.sigmoid(self.heads[i](task_rep)))
                all_preds.append(torch.sigmoid(self.heads[i](task_rep)).squeeze(1))

            return all_preds, None


def compute_balanced_loss(model, logits, y, t, n_tasks):
    total_loss = torch.tensor(0.0, device=logits.device)

    for i in range(n_tasks):
        mask = (t == i)
        if not mask.any():
            continue

        y_task = y[mask]

        # POS WEIGHT PER TASK
        n_pos = (y_task == 1).sum()
        n_neg = (y_task == 0).sum()

        if n_pos == 0 or n_neg == 0:
            continue

        pos_weight = (n_neg / n_pos).to(logits.device)

        task_loss = F.binary_cross_entropy_with_logits(
            logits[mask],
            y_task,
            pos_weight=pos_weight
        )

        precision = torch.exp(-model.log_vars[i])
        task_weight = 1.0 / mask.sum().float()
        total_loss += task_weight * (precision * task_loss + model.log_vars[i])

    return total_loss


# =================================================================
# 5. EKSPERYMENT
# =================================================================

def run_mmoe_experiment(df, tasks, combo, cache):
    combo_name = "+".join(combo)
    print(f"\n>>> URUCHAMIANIE: {combo_name}")
    combo_dir = os.path.join(OUTPUT_DIR, combo_name.replace("+", "_"))
    os.makedirs(combo_dir, exist_ok=True)

    train_df = df[df.split == "train"].copy()
    test_df = df[df.split == "test"].copy()
    task_map = {t: i for i, t in enumerate(tasks)}

    # --- POPRAWIONE SKALOWANIE ---
    feats_tr, feats_te = [], []
    for m in combo:
        X_m_tr = cache[m]["train"]
        X_m_te = cache[m]["test"]

        if m in ["chemberta", "rdkit"]:
            scaler = StandardScaler()
            X_m_tr = scaler.fit_transform(X_m_tr)
            X_m_te = scaler.transform(X_m_te)  # Tylko transform!

        feats_tr.append(X_m_tr)
        feats_te.append(X_m_te)

    Xtr = np.concatenate(feats_tr, axis=1).astype(np.float32)
    Xte = np.concatenate(feats_te, axis=1).astype(np.float32)


    # Przygotowanie tensorów
    Xtr_t = torch.tensor(Xtr).to(DEVICE)
    ytr_t = torch.tensor((train_df.label.values > 0).astype(int)).float().to(DEVICE)
    ttr_t = torch.tensor(train_df.task.map(task_map).values).to(DEVICE)

    yte = (test_df.label.values > 0).astype(int)
    tte = test_df.task.map(task_map).values

    # Model
    model = MMoE_MTLNet(Xtr.shape[1], len(tasks)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Trening
    batch_size = 512

    for epoch in range(30):
        model.train()

        perm = torch.randperm(Xtr_t.size(0))

        for i in range(0, Xtr_t.size(0), batch_size):
            idx = perm[i:i + batch_size]

            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            tb = ttr_t[idx]

            logits, _ = model(xb, tb)
            loss = compute_balanced_loss(model, logits, yb, tb, len(tasks))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} done")

    # --- POPRAWIONA EWALUACJA ---
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte).to(DEVICE)
        head_outputs, _ = model(Xte_t)

    task_stats = []
    for i, t_name in enumerate(tasks):
        # Pobieramy przewidywania dla i-tej głowicy (wszystkie próbki)
        preds = head_outputs[i].cpu().numpy().flatten()
        preds = np.clip(preds, 1e-6, 1 - 1e-6)

        # Maska dla próbek należących do tego zadania
        mask = (tte == i)

        if not any(mask): continue  # Omiń, jeśli brak danych testowych dla zadania

        # FIX: Nakładamy maskę na przewidywania, aby pasowały do yte[mask]
        auc_mtl = roc_auc_score(yte[mask], preds[mask])

        # STL Baseline
        mask_tr = (train_df.task.map(task_map).values == i)
        if len(np.unique(ytr_t[mask_tr].cpu().numpy())) > 1:
            pos = (ytr_t[mask_tr] == 1).sum().item()
            neg = (ytr_t[mask_tr] == 0).sum().item()

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                scale_pos_weight=neg / pos if pos > 0 else 1
            )

            clf.fit(Xtr[mask_tr], ytr_t[mask_tr].cpu().numpy())
            auc_stl = roc_auc_score(yte[mask], clf.predict_proba(Xte[mask])[:, 1])
        else:
            auc_stl = 0.5  # Brak możliwości wyliczenia jeśli brak klas

        task_stats.append({"task": t_name, "MTL_AUC": auc_mtl, "STL_AUC": auc_stl})

    res_df = pd.DataFrame(task_stats)
    res_df.to_csv(os.path.join(combo_dir, "metrics.csv"), index=False)
    #return {"features": combo_name, "MTL": res_df.MTL_AUC.mean(), "STL": res_df.STL_AUC.mean()}
    return {
        "features": combo_name,
        "MTL": res_df.MTL_AUC.mean(),
        "STL": res_df.STL_AUC.mean(),
        "model": model,
        "Xte": Xte,
        "yte": yte,
        "tte": tte,
        "combo_dir": combo_dir
    }



def plot_gate_weights(model, X, task_names, save_path):
    model.eval()
    X_t = torch.tensor(X).to(DEVICE)

    gate_means = []

    with torch.no_grad():
        for i in range(model.n_tasks):
            g = model.gates[i](X_t)  # [N, n_experts]
            gate_means.append(g.mean(dim=0).cpu().numpy())

    gate_means = np.array(gate_means)

    plt.figure(figsize=(10, 6))
    sns.heatmap(gate_means, annot=True, cmap="viridis",
                xticklabels=[f"Expert {i}" for i in range(model.n_experts)],
                yticklabels=task_names)
    plt.title("MMoE Gate Weights (średnie)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def plot_gate_entropy(model, X, task_names, save_path):
    model.eval()
    X_t = torch.tensor(X).to(DEVICE)

    entropies = []

    with torch.no_grad():
        for i in range(model.n_tasks):
            g = model.gates[i](X_t) + 1e-8
            entropy = (-g * torch.log(g)).sum(dim=1).mean().item()
            entropies.append(entropy)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=task_names, y=entropies)
    plt.xticks(rotation=45)
    plt.title("Gate Entropy per Task")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()




def plot_prediction_distribution(model, X, y, t, task_names, save_path):
    model.eval()
    X_t = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        preds, _ = model(X_t)

    plt.figure(figsize=(12, 8))

    for i, name in enumerate(task_names):
        mask = (t == i)
        if not mask.any():
            continue

        p = preds[i].cpu().numpy().flatten()
        p = p[mask]              # 🔥 FIX
        y_task = y[mask]

        plt.subplot(4, 4, i + 1)
        sns.histplot(p[y_task == 0], color="blue", label="neg", stat="density", bins=30)
        sns.histplot(p[y_task == 1], color="red", label="pos", stat="density", bins=30)
        plt.title(name)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def plot_prediction_correlation(model, X, task_names, save_path):
    model.eval()
    X_t = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        preds, _ = model(X_t)

    preds_mat = np.vstack([p.cpu().numpy().flatten() for p in preds])

    corr = np.corrcoef(preds_mat)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm",
                xticklabels=task_names, yticklabels=task_names)
    plt.title("Prediction Correlation (MMoE)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


from sklearn.metrics import roc_curve, auc

def plot_roc_curves(model, X, y, t, task_names, save_path):
    model.eval()
    X_t = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        preds, _ = model(X_t)

    plt.figure(figsize=(10, 8))

    for i, name in enumerate(task_names):
        mask = (t == i)
        if not mask.any():
            continue

        p = preds[i].cpu().numpy().flatten()[mask]
        y_task = y[mask]

        if len(np.unique(y_task)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_task, p)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title("ROC Curves")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def plot_final_summary(summary_df, df, tasks, output_dir):
    # --- GAIN ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x="features", y="GAIN", palette="coolwarm")
    plt.axhline(0, color='black', lw=1)
    plt.title("MMoE MTL Gain over STL (Mean AUC)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_gain_plot.png"))
    plt.show()

    # --- LABEL CORRELATION ---
    lc = label_corr(df, tasks)

    plt.figure(figsize=(12, 10))
    sns.heatmap(lc, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Input Label Correlation (ADME + TOX)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_correlation.png"))
    plt.show()



def analyze_model(res, tasks):
    model = res["model"]
    model.eval()
    Xte = res["Xte"]
    yte = res["yte"]
    tte = res["tte"]
    combo_dir = res["combo_dir"]

    plot_gate_weights(
        model, Xte, tasks,
        os.path.join(combo_dir, "gate_weights.png")
    )

    plot_gate_entropy(
        model, Xte, tasks,
        os.path.join(combo_dir, "gate_entropy.png")
    )

    plot_prediction_distribution(
        model, Xte, yte, tte, tasks,
        os.path.join(combo_dir, "pred_dist.png")
    )

    plot_prediction_correlation(
        model, Xte, tasks,
        os.path.join(combo_dir, "pred_corr.png")
    )

    plot_roc_curves(
        model, Xte, yte, tte, tasks,
        os.path.join(combo_dir, "roc_curves.png")
    )

# =================================================================
# 6. MAIN
# =================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    combos = [
        ["morgan"],
        ["chemberta"],
        ["rdkit"],
        ["morgan", "chemberta"],
        ["morgan", "rdkit"]
    ]

    # --- LOAD ---
    df = load_tdc_tasks(TASKS_ADME, TASKS_TOX)

    # usuń taski z jedną klasą
    df = df.groupby("task").filter(lambda x: x.label.nunique() > 1)

    # KLUCZOWE: aktualna lista tasków
    tasks = sorted(df.task.unique())
    print(f"Final tasks: {tasks}")

    # --- SPLIT ---
    train_df = df[df.split == "train"].copy()
    test_df = df[df.split == "test"].copy()

    # --- FEATURE CACHE (BEZ LEAKAGE) ---
    cache = {}
    for m in ["morgan", "chemberta", "rdkit"]:
        cache[m] = {
            "train": build_features(train_df, m, "train"),
            "test": build_features(test_df, m, "test")
        }

    results = []

    for c in combos:
        print(f"\n=== COMBO: {c} ===")

        res = run_mmoe_experiment(df, tasks, c, cache)

        analyze_model(res, tasks)

        results.append({
            "features": res["features"],
            "MTL": res["MTL"],
            "STL": res["STL"]
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- SUMMARY ---
    summary_df = pd.DataFrame(results)
    summary_df["GAIN"] = summary_df["MTL"] - summary_df["STL"]

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "mmoe_overall_summary.csv"),
        index=False
    )

    # wykresy końcowe
    plot_final_summary(summary_df, df, tasks, OUTPUT_DIR)



if __name__ == "__main__":
    main()