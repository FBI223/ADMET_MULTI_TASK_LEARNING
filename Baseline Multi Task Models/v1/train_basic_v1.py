# =========================
# INSTALL
# pip install torch pandas numpy scikit-learn matplotlib seaborn PyTDC transformers
# =========================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix

from tdc.single_pred import ADME, Tox
from transformers import AutoTokenizer, AutoModel





# =========================
# CONFIG
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tasks_reg = ["sol", "caco2", "ppb", "clearance"]
tasks_clf = ["herg", "dili", "bbb", "hia"]

# =========================
# DATA
# =========================
def load_data():
    datasets = {
        "sol": ADME(name="Solubility_AqSolDB").get_data(),
        "caco2": ADME(name="Caco2_Wang").get_data(),
        "ppb": ADME(name="PPBR_AZ").get_data(),
        "clearance": ADME(name="Clearance_Microsome_AZ").get_data(),
        "bbb": ADME(name="BBB_Martins").get_data(),
        "hia": ADME(name="HIA_Hou").get_data(),
        "herg": Tox(name="hERG").get_data(),
        "dili": Tox(name="DILI").get_data()
    }

    dfs = []
    for name, d in datasets.items():
        d = d.rename(columns={"Drug": "smiles", "Y": name})
        d = d[["smiles", name]]
        dfs.append(d)

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="smiles", how="outer")

    return df.reset_index(drop=True)

# =========================
# EMBEDDINGS (ChemBERTa)
# =========================
def load_chemberta():
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="hf_cache")
    model = AutoModel.from_pretrained(model_name, cache_dir="hf_cache").to(device)
    model.eval()
    return tokenizer, model

def compute_descriptors(smiles_list):
    feats = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            feats.append([0]*6)
            continue

        feats.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol)
        ])

    return np.array(feats)


def inverse_transform(pr, yr, tasks):
    pr_new = pr.copy()
    yr_new = yr.copy()

    for i, t in enumerate(tasks):
        if t in ["ppb", "clearance"]:
            pr_new[:, i] = np.exp(pr[:, i])
            yr_new[:, i] = np.exp(yr[:, i])

    return pr_new, yr_new


def embed_smiles(smiles, tokenizer, model, batch_size=32):
    embeddings = []

    for i in range(0, len(smiles), batch_size):
        batch = smiles[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            out = model(**tokens)

        emb = out.last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)

# =========================
# PREPROCESS
# =========================
def preprocess(df):
    y_reg = df[tasks_reg].values
    y_clf = df[tasks_clf].values

    mask_reg = ~np.isnan(y_reg)
    mask_clf = ~np.isnan(y_clf)

    y_reg = np.nan_to_num(y_reg)
    y_clf = np.nan_to_num(y_clf)

    # log transform
    y_reg = transform_regression(y_reg, tasks_reg)

    # normalize
    mean = y_reg.mean(0)
    std = y_reg.std(0) + 1e-8
    y_reg = (y_reg - mean) / std

    return y_reg, y_clf, mask_reg, mask_clf, mean, std

# =========================
# DATASET
# =========================
class DS(torch.utils.data.Dataset):
    def __init__(self, X, yr, yc, mr, mc):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.yr = torch.tensor(yr, dtype=torch.float32)
        self.yc = torch.tensor(yc, dtype=torch.float32)
        self.mr = torch.tensor(mr, dtype=torch.float32)
        self.mc = torch.tensor(mc, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.yr[i], self.yc[i], self.mr[i], self.mc[i]

# =========================
# MODEL
# =========================
class MTL(nn.Module):
    def __init__(self, in_dim, n_reg, n_clf):
        super().__init__()

        # shared trunk
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.reg_tower = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.clf_tower = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # heads
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(n_reg)
        ])

        self.clf_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(n_clf)
        ])

    def forward(self, x):
        h = self.shared(x)

        h_reg = self.reg_tower(h)
        h_clf = self.clf_tower(h)

        reg_out = torch.cat([head(h_reg) for head in self.reg_heads], dim=1)
        clf_out = torch.cat([head(h_clf) for head in self.clf_heads], dim=1)

        return reg_out, clf_out


# =========================
# LOSS
# =========================
mse = nn.MSELoss(reduction="none")
bce = nn.BCEWithLogitsLoss(reduction="none")



def compute_total_loss(pr, pc, yr, yc, mr, mc, reg_weight=0.8, clf_weight=1.0):
    losses = []

    # regresja
    reg_losses = []
    for i in range(pr.shape[1]):
        l = mse(pr[:, i], yr[:, i])
        l = (l * mr[:, i]).sum() / (mr[:, i].sum() + 1e-8)
        reg_losses.append(l)

    # klasyfikacja
    clf_losses = []
    for i in range(pc.shape[1]):
        l = bce(pc[:, i], yc[:, i])
        l = (l * mc[:, i]).sum() / (mc[:, i].sum() + 1e-8)
        clf_losses.append(l)

    # 🔥 balance
    reg_loss = sum(reg_losses) / len(reg_losses)
    clf_loss = sum(clf_losses) / len(clf_losses)

    total = reg_weight * reg_loss + clf_weight * clf_loss

    return total


def compute_losses(pr, pc, yr, yc, mr, mc):
    losses = []

    for i in range(pr.shape[1]):
        l = mse(pr[:, i], yr[:, i])
        l = (l * mr[:, i]).sum() / (mr[:, i].sum() + 1e-8)
        losses.append(l)

    for i in range(pc.shape[1]):
        l = bce(pc[:, i], yc[:, i])
        l = (l * mc[:, i]).sum() / (mc[:, i].sum() + 1e-8)
        losses.append(l)

    return losses

def transform_regression(y, tasks):
    y_new = y.copy()

    for i, t in enumerate(tasks):
        col = y[:, i]

        if t in ["ppb", "clearance"]:
            col = np.log(np.clip(col, a_min=1e-6, a_max=None))

        # sol i caco2 → NIE logujemy (mają wartości ujemne)

        y_new[:, i] = col

    return y_new

# =========================
# TRAIN
# =========================
def train_model(model, loader, epochs=30):
    opt = torch.optim.Adam(
        list(model.parameters()),
        lr=3e-4,
        weight_decay=1e-5
    )
    hist = []

    for epoch in range(epochs):
        model.train()
        total = 0

        for xb, yrb, ycb, mrb, mcb in loader:
            xb, yrb, ycb, mrb, mcb = xb.to(device), yrb.to(device), ycb.to(device), mrb.to(device), mcb.to(device)

            opt.zero_grad()
            pr, pc = model(xb)
            loss = compute_total_loss(pr, pc, yrb, ycb, mrb, mcb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


            total += loss.item()

        hist.append(total / len(loader))
        print(f"Epoch {epoch}: {hist[-1]:.4f}")

    return hist

# =========================
# EVAL
# =========================
def evaluate(model, X_te, yr_te, yc_te, mean, std):
    model.eval()
    X_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    with torch.no_grad():
        pr, pc = model(X_t)

    pr = pr.cpu().numpy()
    pc = torch.sigmoid(pc).cpu().numpy()

    pr = pr * std + mean
    yr = yr_te * std + mean


    # inverse log tylko dla wybranych tasków
    pr = np.clip(pr, -10, 10)
    yr = np.clip(yr, -10, 10)

    pr, yr = inverse_transform(pr, yr, tasks_reg)

    for i, t in enumerate(tasks_reg):
        rmse = np.sqrt(mean_squared_error(yr[:, i], pr[:, i]))
        print(t, "RMSE:", rmse)

    for i, t in enumerate(tasks_clf):
        auc = roc_auc_score(yc_te[:, i], pc[:, i])
        print(t, "AUROC:", auc)

    return pr, pc, yr

# =========================
# PLOTS
# =========================
def plot_all(loss_hist, pr, yr, pc, yc_te):
    plt.plot(loss_hist)
    plt.title("Loss")
    plt.show()

    for i, t in enumerate(tasks_reg):
        plt.scatter(pr[:, i], yr[:, i])
        plt.title(t)
        plt.show()

    for i, t in enumerate(tasks_clf):
        preds = (pc[:, i] > 0.5).astype(int)
        cm = confusion_matrix(yc_te[:, i], preds)

        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(t)
        plt.show()

# =========================
# MAIN
# =========================
def main():
    df = load_data()

    tokenizer, chemberta = load_chemberta()
    X_chem = embed_smiles(df["smiles"].tolist(), tokenizer, chemberta)
    X_desc = compute_descriptors(df["smiles"].tolist())
    scaler_chem = StandardScaler()
    scaler_desc = StandardScaler()

    X_chem = scaler_chem.fit_transform(X_chem)
    X_desc = scaler_desc.fit_transform(X_desc)

    X = np.concatenate([X_chem, X_desc], axis=1)

    df["ppb"] = df["ppb"].clip(upper=df["ppb"].quantile(0.99))
    df["clearance"] = df["clearance"].clip(upper=df["clearance"].quantile(0.99))

    y_reg, y_clf, mr, mc, mean, std = preprocess(df)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, mr_tr, mr_te, mc_tr, mc_te = train_test_split(
        X, y_reg, y_clf, mr, mc, test_size=0.2, random_state=42
    )

    train_ds = DS(X_tr, yr_tr, yc_tr, mr_tr, mc_tr)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    model = MTL(X.shape[1], len(tasks_reg), len(tasks_clf)).to(device)


    loss_hist = train_model(model, loader)

    torch.save({
        "model": model.state_dict(),
        "scaler_chem": scaler_chem,
        "scaler_desc": scaler_desc
    }, "mtl_chemberta.pt")

    pr, pc, yr = evaluate(model, X_te, yr_te, yc_te, mean, std)

    plot_all(loss_hist, pr, yr, pc, yc_te)

if __name__ == "__main__":
    main()