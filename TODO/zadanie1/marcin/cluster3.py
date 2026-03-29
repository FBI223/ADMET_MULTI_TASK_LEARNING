# =========================
# 1. IMPORTS
# =========================
import numpy as np
import pandas as pd
import torch.nn as nn
from tdc.single_pred import ADME, Tox
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler

CACHE_DIR = "feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(mode, split, size):
    return os.path.join(CACHE_DIR, f"{mode}_{split}_{size}.pkl")

def save_cache(X, path):
    with open(path, "wb") as f:
        pickle.dump(X, f)


def load_cache(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None



tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model_bert = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

model_bert.eval()



def chemberta_embeddings(smiles_list, batch_size=32):
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]

            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

            outputs = model_bert(**inputs)

            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

            embeddings.append(cls_emb.numpy())

    return np.vstack(embeddings)

# =========================
# 2. DATA
# =========================
def load_tdc_tasks(tasks):
    data_dict = {}
    splits = []

    for t in tasks:

        if t.startswith("cyp") or t in ["pgp_broccatelli", "hia_hou", "bbb_martins", "bioavailability_ma"]:
            loader = ADME
        else:
            loader = Tox


        d = loader(name=t)
        s = d.get_split()

        for split_name in ["train", "valid", "test"]:
            df = s[split_name].rename(columns={"Drug": "smiles", "Y": "label"})
            df = df[["smiles", "label"]].dropna()
            df["task"] = t
            df["split"] = split_name
            splits.append(df)

        data_dict[t] = s

    df_all = pd.concat(splits).reset_index(drop=True)

    return df_all, data_dict






# =========================
# 3. FEATURES
# =========================

def clean_nan(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def make_global_split(df):
    smiles_unique = df.smiles.unique()

    np.random.seed(42)
    np.random.shuffle(smiles_unique)

    n = len(smiles_unique)

    train_sm = set(smiles_unique[:int(0.7*n)])
    val_sm   = set(smiles_unique[int(0.7*n):int(0.85*n)])
    test_sm  = set(smiles_unique[int(0.85*n):])

    df["split"] = df.smiles.map(
        lambda x: "train" if x in train_sm else ("valid" if x in val_sm else "test")
    )

    return df


def check_leakage(df):
    train_sm = set(df[df.split=="train"].smiles)
    val_sm   = set(df[df.split=="valid"].smiles)
    test_sm  = set(df[df.split=="test"].smiles)

    print("Leakage check:")
    print("train ∩ test:", len(train_sm & test_sm))
    print("train ∩ val:", len(train_sm & val_sm))
    print("val ∩ test:", len(val_sm & test_sm))


def morgan_fp(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol is None:
        return np.zeros(1024)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))


def rdkit_desc_all(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol is None:
        return np.zeros(len(Descriptors.descList))

    vals = []
    for _, f in Descriptors.descList:
        try:
            v = f(mol)
            if np.isnan(v) or np.isinf(v):
                v = 0.0
        except:
            v = 0.0
        vals.append(v)

    return np.array(vals)


def rdkit_desc(sm):
    mol = Chem.MolFromSmiles(sm)

    if mol is None:
        return np.zeros(16)

    try:
        vals = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),

            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),

            Descriptors.NumRotatableBonds(mol),

            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),

            Descriptors.HeavyAtomCount(mol),

            Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol),

            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcExactMolWt(mol),

            Descriptors.qed(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol)
        ]

        vals = np.array(vals, dtype=float)

        # safety
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        return vals

    except:
        return np.zeros(16)





def build_features(df, mode, split_name):
    cache_path = get_cache_path(mode, split_name, len(df))

    X_cached = load_cache(cache_path)
    if X_cached is not None:
        print(f"[CACHE HIT] {mode} {split_name}")
        return X_cached

    print(f"[COMPUTE] {mode} {split_name}")

    smiles = df["smiles"].tolist()

    X = []

    if mode == "chemberta":
        X = chemberta_embeddings(smiles)

    elif mode == "morgan":
        X = [morgan_fp(sm) for sm in smiles]

    elif mode == "rdkit":
        X = [rdkit_desc(sm) for sm in smiles]

    X = np.vstack(X)
    X = clean_nan(X)

    save_cache(X, cache_path)

    return X

def scale_data(Xtr, Xva, Xte):
    scaler = StandardScaler()
    scaler.fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)


# =========================
# 4. MODELS
# =========================
class MTLNet(nn.Module):
    def __init__(self, in_dim, n_tasks):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x, t):
        h = self.shared(x)

        out = torch.zeros(x.size(0), device=x.device)

        for i, head in enumerate(self.heads):
            mask = (t == i)
            if mask.any():
                out[mask] = head(h[mask]).squeeze()

        return out, h


def compute_mtl_loss(model, logits, h, yb, tb, n_tasks):
    loss = 0.0
    active_tasks = 0

    for i in range(n_tasks):
        mask = (tb == i)

        if mask.any():
            y_task = yb[mask]
            logits_task = logits[mask]

            logits_task = torch.clamp(logits_task, -10, 10)

            pos = (y_task == 1).sum()
            neg = (y_task == 0).sum()

            if pos == 0:
                continue

            pos_weight = neg / (pos + 1e-8)

            l_i = F.binary_cross_entropy_with_logits(
                logits_task,
                y_task,
                pos_weight=pos_weight
            )

            # 🔥 stabilne ważenie tasków
            w_i = 1.0 / torch.sqrt(pos + neg + 1e-8)

            loss += w_i * l_i
            active_tasks += 1

    if active_tasks > 0:
        loss = loss / active_tasks

    # =========================
    # 🔥 REGULARIZATION (STABILNA)
    # =========================
    reg = 0.0

    for i in range(n_tasks):
        mask_i = (tb == i)
        if not mask_i.any():
            continue

        p_i = torch.sigmoid(model.heads[i](h[mask_i])).mean()

        for j in range(i + 1, n_tasks):
            mask_j = (tb == j)
            if not mask_j.any():
                continue

            p_j = torch.sigmoid(model.heads[j](h[mask_j])).mean()

            reg += (p_i - p_j) ** 2

    loss = loss + 0.001 * reg

    return loss


def train_xgb(X, y):
    m = XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8
)
    m.fit(X, y)
    return m

def plot_main_results(results_list):
    df = pd.DataFrame(results_list)

    sns.barplot(data=df.melt(id_vars="mode"),
                x="mode", y="value", hue="variable")

    plt.title("MTL vs STL across feature types")
    plt.show()

def plot_feature_impact(results_list):
    df = pd.DataFrame(results_list)

    plt.plot(df["mode"], df["MTL"], label="MTL", marker="o")
    plt.plot(df["mode"], df["XGB_STL"], label="STL", marker="o")

    plt.title("Feature impact on AUROC")
    plt.legend()
    plt.show()

def plot_mtl_gain(results_list):
    df = pd.DataFrame(results_list)

    df["GAIN"] = df["MTL"] - df["XGB_STL"]

    sns.barplot(x="mode", y="GAIN", data=df)

    plt.axhline(0, color='red', linestyle='--')
    plt.title("MTL Gain over STL")
    plt.show()


def per_task_auc(model, X, y, t, tasks):
    model.eval()

    X = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(t)

    with torch.no_grad():
        logits, _ = model(X, t)
        preds = torch.sigmoid(logits).numpy()

    df = pd.DataFrame({
        "y": y,
        "pred": preds,
        "task": t.numpy()
    })

    for i, task in enumerate(tasks):
        sub = df[df.task == i]
        if len(np.unique(sub.y)) > 1:
            auc = roc_auc_score(sub.y, sub.pred)
            print(f"{task}: {auc:.4f}")


def compute_overlap(df_all, tasks):
    print("\n=== OVERLAP ===")
    for t1 in tasks:
        for t2 in tasks:
            d1 = df_all[df_all.task==t1]
            d2 = df_all[df_all.task==t2]

            overlap = len(set(d1.smiles) & set(d2.smiles))
            print(f"{t1} vs {t2}: {overlap}")

# =========================
# 5. EVALUATION
# =========================
def eval_mtl(model, X, y, t):

    model.eval()   # 🔥 KLUCZOWE

    X = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(t)

    with torch.no_grad():
        logits, _ = model(X, t)
        logits = torch.clamp(logits, -20, 20)
        preds = torch.sigmoid(logits).numpy()

    return safe_auc(y, preds)


def train_mtl(X, y, t_idx, n_tasks, epochs=50, batch_size=256):

    model = MTLNet(X.shape[1], n_tasks)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t_idx = torch.tensor(t_idx)

    # =========================
    # BALANCED SAMPLER
    # =========================
    task_counts = np.bincount(t_idx.numpy())
    weights = 1.0 / (task_counts + 1e-8)
    sample_weights = weights[t_idx.numpy()]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loader = DataLoader(
        TensorDataset(X, y, t_idx),
        batch_size=batch_size,
        sampler=sampler
    )

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):

        model.train()  # 🔥 ważne przy każdej epoce

        for xb, yb, tb in loader:
            opt.zero_grad()

            logits, h = model(xb, tb)

            logits = torch.clamp(logits, -20, 20)

            loss = compute_mtl_loss(model, logits, h, yb, tb, n_tasks)

            loss.backward()
            opt.step()

        if epoch % 10 == 0:
            print(f"[epoch {epoch}] loss={loss.item():.4f}")

    return model


def eval_clf(model, X, y):
    return safe_auc(y, model.predict_proba(X)[:,1])


def safe_auc(y, preds):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, preds)

# =========================
# 6. MAIN EXPERIMENT
# =========================
def run_experiment(tasks, feature_mode="morgan"):
    df_all, data_dict = load_tdc_tasks(tasks)

    check_leakage(df_all)

    train = df_all[df_all.split == "train"]
    val   = df_all[df_all.split == "valid"]
    test  = df_all[df_all.split == "test"]

    task_map = {t:i for i,t in enumerate(tasks)}

    # =========================
    # FEATURES
    # =========================
    if feature_mode == "morgan":
        Xtr = build_features(train, "morgan", "train")
        Xva = build_features(val,   "morgan", "valid")
        Xte = build_features(test,  "morgan", "test")

    elif feature_mode == "chemberta":
        Xtr = build_features(train, "chemberta", "train")
        Xva = build_features(val,   "chemberta", "valid")
        Xte = build_features(test,  "chemberta", "test")

        # scaling OK dla chemberta
        Xtr, Xva, Xte = scale_data(Xtr, Xva, Xte)

    elif feature_mode == "fused":
        # morgan (NO scaling)
        Xtr_m = build_features(train, "morgan", "train")
        Xva_m = build_features(val,   "morgan", "valid")
        Xte_m = build_features(test,  "morgan", "test")

        # chemberta (scaling)
        Xtr_c = build_features(train, "chemberta", "train")
        Xva_c = build_features(val,   "chemberta", "valid")
        Xte_c = build_features(test,  "chemberta", "test")

        scaler = StandardScaler()
        Xtr_c = scaler.fit_transform(Xtr_c)
        Xva_c = scaler.transform(Xva_c)
        Xte_c = scaler.transform(Xte_c)

        Xtr = np.concatenate([Xtr_m, Xtr_c], axis=1)
        Xva = np.concatenate([Xva_m, Xva_c], axis=1)
        Xte = np.concatenate([Xte_m, Xte_c], axis=1)

    else:
        raise ValueError()

    Xtr = clean_nan(Xtr)
    Xva = clean_nan(Xva)
    Xte = clean_nan(Xte)

    # =========================
    # LABELS
    # =========================
    ytr = (train["label"].values > 0).astype(int)
    yte = (test["label"].values > 0).astype(int)

    ttr = train["task"].map(task_map).values
    tte = test["task"].map(task_map).values

    # =========================
    # MTL
    # =========================
    mtl = train_mtl(Xtr, ytr, ttr, len(tasks))
    auc_mtl = eval_mtl(mtl, Xte, yte, tte)

    # =========================
    # STL (XGB)
    # =========================
    aucs_xgb = []

    for t in tasks:
        tr = train[train.task == t]
        te = test[test.task == t]

        if feature_mode == "fused":
            Xtr_m = build_features(tr, "morgan", "train_stl")
            Xte_m = build_features(te, "morgan", "test_stl")

            Xtr_c = build_features(tr, "chemberta", "train_stl")
            Xte_c = build_features(te, "chemberta", "test_stl")

            scaler = StandardScaler()
            Xtr_c = scaler.fit_transform(Xtr_c)
            Xte_c = scaler.transform(Xte_c)

            Xtr_t = np.concatenate([Xtr_m, Xtr_c], axis=1)
            Xte_t = np.concatenate([Xte_m, Xte_c], axis=1)

        else:
            Xtr_t = build_features(tr, feature_mode, "train_stl")
            Xte_t = build_features(te, feature_mode, "test_stl")

            if feature_mode == "chemberta":
                Xtr_t, _, Xte_t = scale_data(Xtr_t, Xtr_t, Xte_t)

        Xtr_t = clean_nan(Xtr_t)
        Xte_t = clean_nan(Xte_t)

        tr_y = (tr.label.values > 0).astype(int)
        te_y = (te.label.values > 0).astype(int)

        xgb = train_xgb(Xtr_t, tr_y)
        aucs_xgb.append(eval_clf(xgb, Xte_t, te_y))

    print("MTL:", auc_mtl)
    print("STL:", np.mean(aucs_xgb))

    return {
        "mode": feature_mode,
        "MTL": auc_mtl,
        "XGB_STL": np.mean(aucs_xgb),
    }, mtl, Xte, data_dict, df_all


# =========================
# 7. CORRELATIONS
# =========================
def compute_label_corr(df_all, tasks):
    mat = pd.DataFrame(index=tasks, columns=tasks)

    for t1 in tasks:
        for t2 in tasks:
            d1 = df_all[df_all.task==t1][["smiles","label"]]
            d2 = df_all[df_all.task==t2][["smiles","label"]]

            merged = d1.merge(d2, on="smiles")

            if len(merged) > 20:
                mat.loc[t1, t2] = merged["label_x"].corr(merged["label_y"])

    return mat.astype(float)

def compute_pred_corr(model, X):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        h = model.shared(X)
        preds = []
        for head in model.heads:
            preds.append(torch.sigmoid(head(h)).numpy().flatten())
    return pd.DataFrame(preds).T.corr()


# =========================
# 8. PLOTS
# =========================
def plot_results(df):
    sns.barplot(data=df.melt(id_vars="features"),
                x="features", y="value", hue="variable")
    plt.title("MTL vs STL vs Features")
    plt.show()


def plot_corr(mat, title):
    sns.heatmap(mat, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()


# =========================
# 9. RUN ALL
# =========================
def run_full_pipeline():
    tasks = [
        "cyp2d6_veith",
        "cyp3a4_veith",
        "cyp2c9_veith",
        "herg_karim",
        "pgp_broccatelli"
    ]

    results_list = []

    # =========================
    # RUNS (TYLKO RAZ!)
    # =========================
    res_morgan, mtl_morgan, X_morgan, _, df_all = run_experiment(tasks, "morgan")
    results_list.append(res_morgan)

    res_chem, mtl_chem, X_chem, _, _ = run_experiment(tasks, "chemberta")
    results_list.append(res_chem)

    res_fused, mtl_fused, X_fused, _, _ = run_experiment(tasks, "fused")
    results_list.append(res_fused)

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n=== RESULTS ===")
    for r in results_list:
        print(r)

    # =========================
    # 📊 MAIN PLOTS
    # =========================
    plot_main_results(results_list)
    plot_feature_impact(results_list)
    plot_mtl_gain(results_list)

    # =========================
    # 📊 CORRELATIONS
    # =========================
    label_corr = compute_label_corr(df_all, tasks)
    plot_corr(label_corr, "Label Correlation")

    pred_corr = compute_pred_corr(mtl_fused, X_fused)
    plot_corr(pred_corr, "Prediction Correlation")

    print("Mean label corr:", label_corr.values.mean())
    print("Mean pred corr:", pred_corr.values.mean())

    # =========================
    # 📊 OVERLAP (ważne!)
    # =========================
    compute_overlap(df_all, tasks)

    # =========================
    # 📊 PER-TASK AUROC
    # =========================
    print("\n=== PER TASK AUC (MTL FUSED) ===")
    train = df_all[df_all.split == "train"]
    test = df_all[df_all.split == "test"]

    yte = (test["label"].values > 0).astype(int)
    tte = test["task"].map({t:i for i,t in enumerate(tasks)}).values

    per_task_auc(mtl_fused, X_fused, yte, tte, tasks)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run_full_pipeline()


