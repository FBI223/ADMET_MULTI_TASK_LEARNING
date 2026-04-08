import os
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tdc.single_pred import ADME
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "feature_cache")
RESULTS_DIR = os.path.join(BASE_DIR, "final_results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model_bert = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model_bert.eval()

# (Funkcje pomocnicze do wczytywania danych pozostają identyczne jak w regresji)
def get_cache_path(mode, split, size): return os.path.join(CACHE_DIR, f"{mode}_{split}_{size}.pkl")
def save_cache(X, path): 
    with open(path, "wb") as f: pickle.dump(X, f)
def load_cache(path):
    if os.path.exists(path):
        with open(path, "rb") as f: return pickle.load(f)
    return None

def chemberta_embeddings(smiles_list, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = model_bert(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.numpy())
    return np.vstack(embeddings)

def load_tdc_tasks(tasks):
    splits = []
    for t in tasks:
        d = ADME(name=t)
        s = d.get_split()
        for split_name in ["train", "valid", "test"]:
            df = s[split_name].rename(columns={"Drug": "smiles", "Y": "label"})
            df = df[["smiles", "label"]].dropna()
            df["task"] = t
            df["split"] = split_name
            splits.append(df)
    return pd.concat(splits).reset_index(drop=True)

def clean_nan(X): return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def morgan_fp(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol is None: return np.zeros(1024)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

def rdkit_desc_20(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol is None: return np.zeros(20)
    try:
        vals = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol), Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol), rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol), rdMolDescriptors.CalcNumHeteroatoms(mol), rdMolDescriptors.CalcLabuteASA(mol),
            Descriptors.BalabanJ(mol), Descriptors.BertzCT(mol), rdMolDescriptors.CalcExactMolWt(mol), Descriptors.qed(mol)
        ]
        return np.nan_to_num(np.array(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.zeros(20)

def build_features(df, mode, split_name):
    if mode == "all":
        X_m = build_features(df, "morgan", split_name)
        X_r = build_features(df, "rdkit", split_name)
        X_c = build_features(df, "chemberta", split_name)
        return np.hstack([X_m, X_r, X_c])

    cache_path = get_cache_path(mode, split_name, len(df))
    X_cached = load_cache(cache_path)
    if X_cached is not None: return X_cached

    print(f"[COMPUTE] {mode} {split_name}")
    smiles = df["smiles"].tolist()
    if mode == "chemberta": X = chemberta_embeddings(smiles)
    elif mode == "morgan": X = [morgan_fp(sm) for sm in smiles]
    elif mode == "rdkit": X = [rdkit_desc_20(sm) for sm in smiles]
    else: raise ValueError(f"Unknown mode: {mode}")
    
    X = clean_nan(np.vstack(X))
    save_cache(X, cache_path)
    return X

def scale_data(Xtr, Xva, Xte):
    scaler = StandardScaler()
    scaler.fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

# --- NEURAL NETWORK (CLASSIFICATION) ---
class MTLNet(nn.Module):
    def __init__(self, in_dim, n_tasks):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_tasks)
        ])
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x, t_idx):
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h1))) + h1
        h2 = self.dropout(h2)
        h3 = F.relu(self.ln3(self.fc3(h2)))
        out = torch.zeros(x.size(0), device=x.device)
        for i, head in enumerate(self.heads):
            mask = t_idx == i
            if mask.any():
                out[mask] = head(h3[mask]).squeeze()
        return out, h3

def compute_mtl_loss(model, logits, yb, tb):
    loss = 0.0
    for i in torch.unique(tb):
        mask = tb == i
        if mask.sum() == 0: continue
        
        y_task = yb[mask]
        logits_task = torch.clamp(logits[mask], -10, 10)
        pos = (y_task == 1).sum()
        neg = (y_task == 0).sum()
        pos_weight = (neg / (pos + 1e-8)) if pos > 0 else torch.tensor(1.0, device=yb.device)
        
        l_i = F.binary_cross_entropy_with_logits(logits_task, y_task, pos_weight=pos_weight)
        log_var = model.log_vars[i]
        loss += (torch.exp(-log_var) * l_i + log_var)
    return loss

def train_mtl(X_tr, y_tr, t_tr, X_va, y_va, t_va, n_tasks, epochs=300, batch_size=256):
    model = MTLNet(X_tr.shape[1], n_tasks)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15)
    
    X_tr_t, y_tr_t, t_tr_t = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32), torch.tensor(t_tr, dtype=torch.long)
    X_va_t, y_va_t, t_va_t = torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32), torch.tensor(t_va, dtype=torch.long)
    
    weights = 1.0 / (np.bincount(t_tr) + 1e-8)
    sampler = WeightedRandomSampler(weights[t_tr], num_samples=len(t_tr), replacement=True)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t, t_tr_t), batch_size=batch_size, sampler=sampler)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb, tb in loader:
            opt.zero_grad()
            logits, _ = model(xb, tb)
            loss = compute_mtl_loss(model, logits, yb, tb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_va_t, t_va_t)
            val_loss = compute_mtl_loss(model, val_logits, y_va_t, t_va_t).item()
            
        history['train_loss'].append(ep_loss / len(loader))
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model_wts)
    return model, history

# --- EVALUATION & PLOTS ---
def eval_classification_metrics(y_true, preds_proba):
    if len(np.unique(y_true)) < 2: return np.nan, np.nan, np.nan
    auc = roc_auc_score(y_true, preds_proba)
    auprc = average_precision_score(y_true, preds_proba)
    f1 = f1_score(y_true, (preds_proba >= 0.5).astype(int), zero_division=0)
    return auc, auprc, f1

def save_plot(filename):
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curve(history, task_prefix, mode):
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"MTL Learning Curve - Classification ({mode})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_plot(f"learning_curve_{task_prefix}_{mode}.png")

def save_csv(results, filename):
    df = pd.DataFrame([item for sublist in results for item in sublist])
    df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)

def plot_smart_results(results_list, metric, filename):
    df_raw = pd.DataFrame([item for sublist in results_list for item in sublist])
    metrics_cols = [c for c in df_raw.columns if metric in c and "Gain" not in c]
    df_melt = df_raw.melt(id_vars=["mode", "task"], value_vars=metrics_cols, var_name="model", value_name=metric)
    df_melt["model"] = df_melt["model"].str.replace(f"_{metric}", "")
    
    g = sns.catplot(data=df_melt, x="mode", y=metric, hue="model", col="task", kind="bar", height=5, aspect=1.2, sharey=False)
    g.fig.suptitle(f"Classification: {metric} comparison", y=1.08)
    save_plot(filename)

# --- EXPERIMENT ---
def run_experiment(tasks, feature_mode):
    df_all = load_tdc_tasks(tasks)
    train, val, test = df_all[df_all.split == "train"], df_all[df_all.split == "valid"], df_all[df_all.split == "test"]
    task_map = {t: i for i, t in enumerate(tasks)}
    
    Xtr = build_features(train, feature_mode, "train")
    Xva = build_features(val, feature_mode, "valid")
    Xte = build_features(test, feature_mode, "test")
    
    if feature_mode in ["chemberta", "rdkit", "all"]:
        Xtr, Xva, Xte = scale_data(Xtr, Xva, Xte)
        
    ytr, yva, yte = (train["label"].values > 0).astype(int), (val["label"].values > 0).astype(int), (test["label"].values > 0).astype(int)
    ttr, tva, tte = train["task"].map(task_map).values, val["task"].map(task_map).values, test["task"].map(task_map).values
    
    mtl, history = train_mtl(Xtr, ytr, ttr, Xva, yva, tva, len(tasks))
    plot_training_curve(history, "classification", feature_mode)
    
    mtl.eval()
    with torch.no_grad():
        mtl_logits, _ = mtl(torch.tensor(Xte, dtype=torch.float32), torch.tensor(tte, dtype=torch.long))
    mtl_preds = torch.sigmoid(mtl_logits).numpy()
    
    task_results = []
    for t in tasks:
        idx_t = task_map[t]
        mask_te = tte == idx_t
        te_y_true, te_preds_mtl = yte[mask_te], mtl_preds[mask_te]
        
        mtl_auc, mtl_auprc, mtl_f1 = eval_classification_metrics(te_y_true, te_preds_mtl)
        
        tr, te = train[train.task == t], test[test.task == t]
        task_label = f"{t}\n(N={len(tr)})"
        
        Xtr_t, Xte_t = build_features(tr, feature_mode, "train_stl"), build_features(te, feature_mode, "test_stl")
        if feature_mode in ["chemberta", "rdkit", "all"]:
            Xtr_t, _, Xte_t = scale_data(Xtr_t, Xtr_t, Xte_t)
            
        tr_y, te_y = (tr.label.values > 0).astype(int), (te.label.values > 0).astype(int)
        
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, eval_metric="logloss").fit(Xtr_t, tr_y)
        xgb_preds_proba = xgb.predict_proba(Xte_t)[:, 1]
        xgb_auc, xgb_auprc, xgb_f1 = eval_classification_metrics(te_y, xgb_preds_proba)
        
        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42).fit(Xtr_t, tr_y)
        rf_preds_proba = rf.predict_proba(Xte_t)[:, 1]
        rf_auc, rf_auprc, rf_f1 = eval_classification_metrics(te_y, rf_preds_proba)
        
        gain_auc = mtl_auc - max(xgb_auc, rf_auc)
        gain_auprc = mtl_auprc - max(xgb_auprc, rf_auprc)
        gain_f1 = mtl_f1 - max(xgb_f1, rf_f1)
        
        task_results.append({
            "mode": feature_mode, "task": task_label,
            "MTL_AUROC": mtl_auc, "XGB_AUROC": xgb_auc, "RF_AUROC": rf_auc, "MTL_Gain_AUROC": gain_auc,
            "MTL_AUPRC": mtl_auprc, "XGB_AUPRC": xgb_auprc, "RF_AUPRC": rf_auprc, "MTL_Gain_AUPRC": gain_auprc,
            "MTL_F1": mtl_f1, "XGB_F1": xgb_f1, "RF_F1": rf_f1, "MTL_Gain_F1": gain_f1
        })
    return task_results

if __name__ == "__main__":
    tasks = ["cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith"]
    modes = ["morgan", "rdkit", "chemberta", "all"]
    all_results = [run_experiment(tasks, m) for m in modes]
    
    save_csv(all_results, "results_classification.csv")
    plot_smart_results(all_results, "AUROC", "plot_classification_auroc.png")
    plot_smart_results(all_results, "AUPRC", "plot_classification_auprc.png")
    plot_smart_results(all_results, "F1", "plot_classification_f1.png")