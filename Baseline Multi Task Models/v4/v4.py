
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import deepchem as dc

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.stats import ttest_rel

import matplotlib.pyplot as plt
import seaborn as sns

from tdc.single_pred import ADME
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
os.makedirs("results", exist_ok=True)


import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
class Featurizer:
    def __init__(self):
        self.graph = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    def morgan(self, s):
        m = Chem.MolFromSmiles(s)
        if m is None: return np.zeros(1024)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024))

    def rdkit(self, s):
        m = Chem.MolFromSmiles(s)
        if m is None: return np.zeros(5)
        return np.array([
            Descriptors.MolLogP(m),
            Descriptors.MolWt(m),
            Descriptors.NumHDonors(m),
            Descriptors.TPSA(m),
            Descriptors.NumRotatableBonds(m)
        ])

    def graph_feat(self, s):
        try:
            g = self.graph.featurize([s])[0]
            if g is None or isinstance(g, np.ndarray):
                return np.zeros(128)

            nf = g.node_features

            vec = np.concatenate([
                nf.mean(0),
                nf.std(0),
                nf.max(0),
                nf.min(0)
            ])

            # FIX DIMENSION
            if vec.shape[0] < 128:
                vec = np.pad(vec, (0, 128 - vec.shape[0]))
            else:
                vec = vec[:128]

            return vec

        except:
            return np.zeros(128)

    def transform(self, smiles, mode):
        X = []
        for s in tqdm(smiles):
            feats = []
            if 'A' in mode: feats.append(self.morgan(s))
            if 'B' in mode: feats.append(self.rdkit(s))
            if 'C' in mode: feats.append(self.graph_feat(s))
            X.append(np.concatenate(feats))
        return np.array(X)




# =========================
# MODEL
# =========================
class MTLModel(nn.Module):
    def __init__(self, dim, n_tasks):
        super().__init__()
        h = 512

        self.input = nn.Linear(dim, h)

        self.shared = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(3)
        ])

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x):
        x = torch.relu(self.input(x))
        for l in self.shared:
            x = l(x) + x
        return [h(x) for h in self.heads]

# =========================
# LOSS
# =========================
def loss_fn(outputs, targets):
    total = 0
    for i in range(targets.shape[1]):
        mask = ~torch.isnan(targets[:, i])
        if not mask.any(): continue

        out = outputs[i].squeeze()[mask]
        tgt = targets[mask, i]

        if TASK_CONFIG[TASK_NAMES[i]] == 'reg':
            loss_i = nn.MSELoss()(out, tgt)
            total += loss_i / (tgt.std() + 1e-6)
        else:
            loss_i = nn.BCEWithLogitsLoss()(out, tgt)
            total += loss_i

    return total

# =========================
# TRAIN
# =========================
def train(model, X, Y, epochs=30):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        out = model(X)
        loss = loss_fn(out, Y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 5 == 0:
            print(f"Epoch {ep} loss {loss.item():.4f}")

# =========================
# PREDICT
# =========================
def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
        return torch.cat(out, dim=1).numpy()

# =========================
# METRICS
# =========================
def metrics(y, p, task):
    if TASK_CONFIG[task] == 'reg':
        return {
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'R2': r2_score(y, p)
        }
    else:
        prob = 1/(1+np.exp(-p))
        return {'AUROC': roc_auc_score(y, prob)}

# =========================
# EXPERIMENT
# =========================
def run_experiment(X, Y, mode):
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    res_mtl = {t: [] for t in TASK_NAMES}
    res_stl = {t: [] for t in TASK_NAMES}

    for fold, (tr, te) in enumerate(kf.split(X)):
        print(f"FOLD {fold} MODE {mode}")

        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]

        Y_tr_scaled = Y_tr.copy()
        Y_te_scaled = Y_te.copy()

        y_scalers = {}

        for i, t in enumerate(TASK_NAMES):
            if TASK_CONFIG[t] == 'reg':
                mask = ~np.isnan(Y_tr[:, i])
                sc_y = StandardScaler()
                Y_tr_scaled[mask, i] = sc_y.fit_transform(Y_tr[mask, i].reshape(-1, 1)).flatten()
                Y_te_scaled[:, i] = sc_y.transform(Y_te[:, i].reshape(-1, 1)).flatten()
                y_scalers[i] = sc_y

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        # MTL
        model = MTLModel(X_tr.shape[1], len(TASK_NAMES))

        train(model, X_tr, Y_tr_scaled)
        preds = predict(model, X_te)

        for i, t in enumerate(TASK_NAMES):
            if TASK_CONFIG[t] == 'reg' and i in y_scalers:
                preds[:, i] = y_scalers[i].inverse_transform(preds[:, i].reshape(-1, 1)).flatten()

        for i, t in enumerate(TASK_NAMES):
            mask = ~np.isnan(Y_te[:, i])
            if not mask.any(): continue
            res_mtl[t].append(metrics(Y_te[mask, i], preds[mask, i], t))

        # STL
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        for i, t in enumerate(TASK_NAMES):
            mtr = ~np.isnan(Y_tr[:, i])
            mte = ~np.isnan(Y_te[:, i])
            if not mtr.any(): continue

            if TASK_CONFIG[t] == 'reg':
                mdl = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
            else:
                mdl = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

            mdl.fit(X_tr[mtr], Y_tr[mtr, i])
            if TASK_CONFIG[t] == 'reg':
                pred = mdl.predict(X_te[mte])
            else:
                pred = mdl.predict_proba(X_te[mte])[:, 1]

            res_stl[t].append(metrics(Y_te[mte, i], pred, t))

    return res_mtl, res_stl

# =========================
# SIGNIFICANCE + SAVE
# =========================
def significance(res_mtl, res_stl, mode):
    rows = []

    for t in TASK_NAMES:
        for metric in res_mtl[t][0]:
            mtl_vals = [r[metric] for r in res_mtl[t]]
            stl_vals = [r[metric] for r in res_stl[t]]

            stat, p = ttest_rel(mtl_vals, stl_vals)

            rows.append({
                'Mode': mode,
                'Task': t,
                'Metric': metric,
                'MTL_mean': np.mean(mtl_vals),
                'STL_mean': np.mean(stl_vals),
                'p_value': p
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/stats_{mode}.csv", index=False)
    return df

# =========================
# PLOTS
# =========================
def plot_results(res_mtl, res_stl, mode):
    rows = []

    for t in TASK_NAMES:
        for metric in res_mtl[t][0]:
            for v in res_mtl[t]:
                rows.append({'Task': t, 'Metric': metric, 'Model': 'MTL', 'Value': v[metric]})
            for v in res_stl[t]:
                rows.append({'Task': t, 'Metric': metric, 'Model': 'STL', 'Value': v[metric]})

    df = pd.DataFrame(rows)

    for metric in df['Metric'].unique():
        plt.figure()
        sns.boxplot(data=df[df['Metric']==metric], x='Task', y='Value', hue='Model')
        plt.savefig(f"results/{mode}_box_{metric}.png")
        plt.close()

    # heatmap
    heat = []
    for t in TASK_NAMES:
        for metric in res_mtl[t][0]:
            mtl = np.mean([r[metric] for r in res_mtl[t]])
            stl = np.mean([r[metric] for r in res_stl[t]])

            if metric == 'RMSE':
                gain = (stl - mtl)/stl
            else:
                gain = (mtl - stl)/stl

            heat.append([t, metric, gain])

    heat_df = pd.DataFrame(heat, columns=['Task','Metric','Gain'])
    pivot = heat_df.pivot(index='Task', columns='Metric', values='Gain')

    plt.figure()
    sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=0)
    plt.savefig(f"results/{mode}_heatmap.png")
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

    df = df.drop_duplicates('smiles').reset_index(drop=True)

    Y = df[TASK_NAMES].values.astype(float)

    feat = Featurizer()
    modes = ['A','B','C','AB','AC','ABC']

    all_stats = []

    for mode in modes:
        print(f"MODE {mode}")

        X = feat.transform(df['smiles'], mode)

        res_mtl, res_stl = run_experiment(X, Y, mode)

        stats_df = significance(res_mtl, res_stl, mode)
        plot_results(res_mtl, res_stl, mode)

        all_stats.append(stats_df)

    final = pd.concat(all_stats)
    final.to_csv("results/final_stats.csv", index=False)

if __name__ == "__main__":
    main()
