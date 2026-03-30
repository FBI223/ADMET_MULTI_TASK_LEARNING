import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import deepchem as dc

from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

from tdc.single_pred import ADME


import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

RDLogger.DisableLog('rdApp.*')
os.makedirs("results", exist_ok=True)

# =========================
# CONFIG
# =========================
TASK_CONFIG = {
    'Caco2_Wang': 'reg',
    'Lipophilicity_AstraZeneca': 'reg',
    'Solubility_AqSolDB': 'reg',
    'BBB_Martins': 'clf'
}
TASK_NAMES = list(TASK_CONFIG.keys())

# =========================
# FEATURIZER
# =========================
class GNNFeaturizer:
    def __init__(self):
        self.feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    def transform(self, smiles_list):
        X = []
        FIXED_DIM = 96

        for i, sm in enumerate(smiles_list):
            try:
                g = self.feat.featurize([sm])[0]

                if g is None or isinstance(g, np.ndarray):
                    X.append(np.zeros(FIXED_DIM))
                    continue

                nf = g.node_features

                if nf.shape[0] < 2:
                    X.append(np.zeros(FIXED_DIM))
                    continue

                vec = np.concatenate([
                    nf.mean(0),
                    nf.std(0),
                    nf.max(0)
                ])

                if len(vec) > FIXED_DIM:
                    vec = vec[:FIXED_DIM]
                else:
                    vec = np.pad(vec, (0, FIXED_DIM - len(vec)))

                X.append(vec)

            except Exception as e:
                print(f"[FEAT ERROR] {i}: {e}")
                X.append(np.zeros(FIXED_DIM))

        return np.vstack(X)

# =========================
# MODEL
# =========================
class MTLModel(nn.Module):
    def __init__(self, dim, n_tasks):
        super().__init__()
        h = 512

        self.input = nn.Linear(dim, h)

        self.shared = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h, h),
            nn.ReLU()
        )

        self.heads = nn.ModuleList([nn.Linear(h, 1) for _ in range(n_tasks)])
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.shared(x)
        return [h(x) for h in self.heads]

# =========================
# LOSS
# =========================
def loss_fn(outputs, targets, model):
    total = 0

    for i in range(targets.shape[1]):
        mask = ~torch.isnan(targets[:, i])
        if not mask.any(): continue

        out = outputs[i].squeeze()[mask]
        tgt = targets[mask, i]

        if TASK_CONFIG[TASK_NAMES[i]] == 'reg':
            loss = nn.MSELoss()(out, tgt)
        else:
            loss = nn.BCEWithLogitsLoss()(out, tgt)

        precision = torch.exp(-model.log_vars[i])
        total += precision * loss + model.log_vars[i]

    return total

# =========================
# DATA CLEAN
# =========================
def clean_smiles(df):
    valid = []
    for s in df['smiles']:
        mol = Chem.MolFromSmiles(s)
        valid.append(mol is not None and mol.GetNumAtoms() >= 2)
    return df[np.array(valid)].reset_index(drop=True)

# =========================
# ANALYSIS
# =========================
def analyze_data(X, Y):
    print("\n=== GLOBAL STATS ===")
    print("Shape:", X.shape)
    print("NaNs:", np.isnan(X).sum())

    stats = {
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "skew": float(skew(X.flatten())),
        "kurtosis": float(kurtosis(X.flatten()))
    }

    with open("results/global_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], s=5)
    plt.title("PCA projection")
    plt.savefig("results/pca.png")
    plt.close()

    # Feature hist
    plt.figure()
    sns.histplot(X.flatten(), bins=100)
    plt.savefig("results/feature_hist.png")
    plt.close()

# =========================
# TRAIN
# =========================
def train(model, X, Y):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    losses = []

    for ep in range(40):
        model.train()
        opt.zero_grad()

        out = model(X)
        loss = loss_fn(out, Y, model)

        loss.backward()
        opt.step()

        losses.append(loss.item())
        print(f"[E{ep}] loss={loss.item():.4f}")

    plt.figure()
    plt.plot(losses)
    plt.savefig("results/loss.png")
    plt.close()

# =========================
# METRICS
# =========================
def metrics(y, p, task):
    if TASK_CONFIG[task] == 'reg':
        rmse = np.sqrt(mean_squared_error(y, p))
        r2 = r2_score(y, p)
        return rmse, r2
    else:
        prob = 1/(1+np.exp(-p))
        auc = roc_auc_score(y, prob)
        return auc, 0

# =========================
# PLOTS
# =========================
def plot_all(y, p, task, fold):
    if TASK_CONFIG[task] == 'reg':
        plt.figure()
        plt.scatter(y, p, alpha=0.4)
        plt.savefig(f"results/{task}_parity_{fold}.png")
        plt.close()

        res = y - p
        plt.figure()
        sns.histplot(res)
        plt.savefig(f"results/{task}_res_{fold}.png")
        plt.close()

    else:
        from sklearn.metrics import roc_curve
        prob = 1/(1+np.exp(-p))
        fpr, tpr, _ = roc_curve(y, prob)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.savefig(f"results/{task}_roc_{fold}.png")
        plt.close()



def multitask_gain_analysis(res_mtl, res_stl):
    rows = []

    for t in TASK_NAMES:
        mtl = np.array(res_mtl[t])
        stl = np.array(res_stl[t])

        mean_mtl = mtl.mean()
        mean_stl = stl.mean()

        # dla RMSE: niżej = lepiej → odwracamy znak
        if TASK_CONFIG[t] == 'reg':
            gain = mean_stl - mean_mtl
            perc = 100 * (mean_stl - mean_mtl) / abs(mean_stl)
        else:
            gain = mean_mtl - mean_stl
            perc = 100 * (mean_mtl - mean_stl) / abs(mean_stl)

        rows.append({
            "task": t,
            "MTL_mean": mean_mtl,
            "STL_mean": mean_stl,
            "gain_abs": gain,
            "gain_%": perc
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/multitask_gain.csv", index=False)

    # ===== BAR PLOT =====
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="task", y="gain_%")
    plt.axhline(0, color='black')
    plt.title("MTL Gain (%) vs STL")
    plt.xticks(rotation=30)
    plt.savefig("results/multitask_gain_bar.png")
    plt.close()

    # ===== ABS GAIN =====
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="task", y="gain_abs")
    plt.axhline(0, color='black')
    plt.title("MTL Absolute Gain vs STL")
    plt.savefig("results/multitask_gain_abs.png")
    plt.close()

    return df

def task_correlation_analysis(Y, preds_all=None):
    # ===== TRUE TARGET CORRELATION =====
    dfY = pd.DataFrame(Y, columns=TASK_NAMES)

    corr = dfY.corr()

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Target Correlation")
    plt.savefig("results/task_corr_targets.png")
    plt.close()

    corr.to_csv("results/task_corr_targets.csv")

    # ===== PREDICTION CORRELATION =====
    if preds_all is not None:
        dfP = pd.DataFrame(preds_all, columns=TASK_NAMES)
        corr_p = dfP.corr()

        plt.figure(figsize=(6,5))
        sns.heatmap(corr_p, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Prediction Correlation (MTL)")
        plt.savefig("results/task_corr_preds.png")
        plt.close()

        corr_p.to_csv("results/task_corr_preds.csv")


# =========================
# EXPERIMENT
# =========================
def run(X, Y):
    kf = KFold(5, shuffle=True, random_state=42)
    logs = []

    for f,(tr,te) in enumerate(kf.split(X)):
        print(f"\n==== FOLD {f} ====")

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])

        Ytr, Yte = Y[tr], Y[te]

        model = MTLModel(X.shape[1], len(TASK_NAMES))
        train(model, Xtr, Ytr)

        pred = model(torch.tensor(Xte, dtype=torch.float32))
        pred = torch.cat(pred,1).detach().numpy()

        print("log_vars:", model.log_vars.data)

        for i,t in enumerate(TASK_NAMES):
            mask = ~np.isnan(Yte[:,i])
            if not mask.any(): continue

            m1, m2 = metrics(Yte[mask,i], pred[mask,i], t)

            logs.append({
                "fold":f,
                "task":t,
                "metric1":m1,
                "metric2":m2
            })

            plot_all(Yte[mask,i], pred[mask,i], t, f)

    pd.DataFrame(logs).to_csv("results/logs.csv", index=False)





# =========================
# PREDICT
# =========================
def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32)
        out = model(X)
        return torch.cat(out, dim=1).numpy()


# =========================
# TARGET PLOTS
# =========================
def plot_targets(Y):
    for i, t in enumerate(TASK_NAMES):
        y = Y[:, i]
        y = y[~np.isnan(y)]

        if len(y) == 0:
            continue

        plt.figure()
        sns.histplot(y, kde=True)
        plt.title(f"{t} distribution")
        plt.savefig(f"results/{t}_dist.png")
        plt.close()

        # boxplot (outliers)
        plt.figure()
        sns.boxplot(x=y)
        plt.title(f"{t} boxplot")
        plt.savefig(f"results/{t}_box.png")
        plt.close()


# =========================
# FEATURE ANALYSIS
# =========================
def analyze_features(X, Y, df):
    print("\n=== FEATURE ANALYSIS ===")

    print("Shape:", X.shape)
    print("NaNs:", np.isnan(X).sum())
    print("Mean:", np.mean(X))
    print("Std:", np.std(X))

    stats = pd.DataFrame({
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0),
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0)
    })

    stats.to_csv("results/feature_stats.csv")

    # correlation (subset żeby nie zabić RAM)
    dfX = pd.DataFrame(X[:, :50])
    corr = dfX.corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Feature correlation (subset)")
    plt.savefig("results/feature_corr.png")
    plt.close()

    # missing per task
    for i, t in enumerate(TASK_NAMES):
        print(f"{t} missing:", np.isnan(Y[:, i]).sum())


# =========================
# SIGNIFICANCE TEST
# =========================


def significance(res_mtl, res_stl):
    from scipy.stats import ttest_rel

    print("\n=== STATISTICAL SIGNIFICANCE ===")

    for t in TASK_NAMES:
        if len(res_mtl[t]) == 0 or len(res_stl[t]) == 0:
            print(f"{t}: skipped (no data)")
            continue

        mtl = np.array(res_mtl[t], dtype=float)
        stl = np.array(res_stl[t], dtype=float)

        if len(mtl) != len(stl):
            print(f"{t}: skipped (unequal lengths)")
            continue

        stat, p = ttest_rel(mtl, stl)
        print(f"{t} p-value = {float(p):.6f}")

def run_experiment(X, Y):
    import xgboost as xgb

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    res_mtl = {t: [] for t in TASK_NAMES}
    res_stl = {t: [] for t in TASK_NAMES}

    preds_all = []
    logs = []

    for f, (tr, te) in enumerate(kf.split(X)):
        print(f"\n=== FOLD {f} ===")

        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        # ===== MTL =====
        model = MTLModel(X_tr.shape[1], len(TASK_NAMES))
        train(model, X_tr, Y_tr)
        preds = predict(model, X_te)

        preds_all.append(preds)

        for i, t in enumerate(TASK_NAMES):
            mask = ~np.isnan(Y_te[:, i])
            if not mask.any():
                continue

            y_true = Y_te[mask, i]
            y_pred = preds[mask, i]

            try:
                m1, m2 = metrics(y_true, y_pred, t)
                score = float(m1)
            except Exception as e:
                print(f"[MTL ERROR] {t}: {e}")
                continue

            res_mtl[t].append(score)

            logs.append({
                "fold": f,
                "task": t,
                "model": "MTL",
                "score": score
            })

            plot_all(y_true, y_pred, t, f)

        # ===== STL =====
        for i, t in enumerate(TASK_NAMES):
            mtr = ~np.isnan(Y_tr[:, i])
            mte = ~np.isnan(Y_te[:, i])

            if not mtr.any() or not mte.any():
                continue

            y_tr = Y_tr[mtr, i]
            y_te = Y_te[mte, i]

            try:
                if TASK_CONFIG[t] == 'reg':
                    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=8)
                    mdl.fit(X_tr[mtr], y_tr)
                    pred = mdl.predict(X_te[mte])

                else:
                    # 🔥 FIX: AUC crash
                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                        print(f"[SKIP STL] {t} only one class")
                        continue

                    mdl = xgb.XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric="logloss")
                    mdl.fit(X_tr[mtr], y_tr)
                    pred = mdl.predict_proba(X_te[mte])[:, 1]

                m1, m2 = metrics(y_te, pred, t)
                score = float(m1)

            except Exception as e:
                print(f"[STL ERROR] {t}: {e}")
                continue

            res_stl[t].append(score)

            logs.append({
                "fold": f,
                "task": t,
                "model": "STL",
                "score": score
            })

    # ===== SAVE LOGS =====
    logs_df = pd.DataFrame(logs)

    print("\n=== LOGS ===")
    print("Records:", len(logs_df))

    if len(logs_df) > 0:
        logs_df.to_csv("results/logs.csv", index=False)
    else:
        print("⚠️ WARNING: logs empty")

    # ===== STACK PREDS =====
    if len(preds_all) > 0:
        preds_all = np.vstack(preds_all)
    else:
        preds_all = None

    return res_mtl, res_stl, preds_all


# =========================
# SUMMARY PLOTS
# =========================
def summary_plots():
    df = pd.read_csv("results/logs.csv")

    for t in TASK_NAMES:
        d = df[df.task == t]

        if len(d) == 0:
            continue

        # boxplot
        plt.figure()
        sns.boxplot(data=d, x="model", y="score")
        plt.title(f"{t} comparison")
        plt.savefig(f"results/{t}_boxplot.png")
        plt.close()

        # violin
        plt.figure()
        sns.violinplot(data=d, x="model", y="score")
        plt.title(f"{t} violin")
        plt.savefig(f"results/{t}_violin.png")
        plt.close()

        # mean bar
        plt.figure()
        sns.barplot(data=d, x="model", y="score")
        plt.title(f"{t} mean performance")
        plt.savefig(f"results/{t}_bar.png")
        plt.close()




# =========================
# MAIN
# =========================
def main():
    dfs = []
    for t in TASK_NAMES:
        d = ADME(name=t).get_data()
        dfs.append(d[['Drug','Y']].rename(columns={'Drug':'smiles','Y':t}))

    df = dfs[0]
    for d in dfs[1:]:
        df = pd.merge(df, d, on='smiles', how='outer')

    df = df.drop_duplicates('smiles')
    df = clean_smiles(df)

    Y = df[TASK_NAMES].values.astype(float)

    feat = GNNFeaturizer()
    X = feat.transform(df['smiles'])

    analyze_features(X, Y, df)
    plot_targets(Y)

    res_mtl, res_stl, preds_all = run_experiment(X, Y)

    # 🔥 NOWE ANALIZY
    task_correlation_analysis(Y, preds_all)
    gain_df = multitask_gain_analysis(res_mtl, res_stl)

    significance(res_mtl, res_stl)
    summary_plots()

    print("\n=== MULTITASK GAIN ===")
    print(gain_df)


if __name__ == "__main__":
    main()