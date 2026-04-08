import os
import torch
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# CONFIG
# =========================

class CFG:
    save_dir = "outputs_random"
    random_state = 42

os.makedirs(CFG.save_dir, exist_ok=True)

# =========================
# FEATURE CONFIG
# =========================

class FeatureConfig:
    USE_MORGAN = True
    USE_RDKIT = True
    USE_QUANTUM = True

# =========================
# FEATURE BUILDER
# =========================


from chemprop.data import MoleculeDatapoint
from rdkit import Chem

class QWDatapoint(MoleculeDatapoint):
    def __init__(self, smiles, targets, mask, rdkit, qc, qc_mask, morgan, split):
        mol = Chem.MolFromSmiles(smiles)
        super().__init__(mol=mol)
        self.y = targets
        self.mask = mask
        self.rdkit = rdkit
        self.qc = qc
        self.qc_mask = qc_mask
        self.morgan = morgan
        self.split = split

def build_features(data, task_idx):
    X, y = [], []

    for d in data:
        if d.mask[task_idx] != 1:
            continue

        features = []

        if FeatureConfig.USE_MORGAN:
            features.append(d.morgan)

        if FeatureConfig.USE_RDKIT:
            features.append(d.rdkit)

        if FeatureConfig.USE_QUANTUM:
            features.append(d.qc)
            features.append(d.qc_mask)

        x = np.concatenate(features)
        X.append(x)
        y.append(d.y[task_idx])

    return np.array(X), np.array(y)

# =========================
# TRAIN RF PER TASK
# =========================

def train_one_task_rf(task_idx, task_name, train_data, val_data):
    print(f"\n[RF] TASK: {task_name}")

    X_train, y_train = build_features(train_data, task_idx)
    X_val, y_val = build_features(val_data, task_idx)

    if len(X_train) < 10:
        print("   ! Za mało danych")
        return 0.0, 0.0, len(X_train), np.zeros(len(X_val))

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=CFG.random_state
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]

    try:
        auc = roc_auc_score(y_val, probs)
        ap = average_precision_score(y_val, probs)
    except:
        auc, ap = 0.5, 0.0

    return auc, ap, len(X_train), probs

# =========================
# SAVE RESULTS
# =========================

def save_results(df_results):
    csv_path = os.path.join(CFG.save_dir, "rf_results.csv")
    yml_path = os.path.join(CFG.save_dir, "rf_metrics.yml")

    results_dict = {}
    for _, row in df_results.iterrows():
        results_dict[row['Task']] = {
            "score": float(row['ROC_AUC']),
            "metric": "ROC-AUC",
            "samples": int(row['Samples'])
        }

    df_results.to_csv(csv_path, index=False)

    final_yaml = {
        "metadata": {
            "model_type": "RandomForest-STL",
            "mean_performance": float(df_results['ROC_AUC'].mean()),
            "features": {
                "Morgan": FeatureConfig.USE_MORGAN,
                "RDKit": FeatureConfig.USE_RDKIT,
                "Quantum": FeatureConfig.USE_QUANTUM
            }
        },
        "tasks": results_dict
    }

    with open(yml_path, 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)

    print(f"[SAVE] {yml_path}")

# =========================
# MAIN
# =========================

def main():
    # 🔥 używasz cache z MTL
    cache_path = os.path.join("outputs_mtl", "preprocessed_data_cache.pt")

    print(f"[LOAD] {cache_path}")

    if not os.path.exists(cache_path):
        print("❌ brak cache z MTL")
        return

    cache = torch.load(cache_path)

    data = cache["data"]
    tasks = cache["tasks"]

    # split
    train_data = [d for d in data if d.split == "train"]
    val_data = [d for d in data if d.split == "valid"]

    print(f"[DATA] Train={len(train_data)} Val={len(val_data)}")

    results = []

    # =========================
    # LOOP TASKS
    # =========================

    for i, t_name in enumerate(tasks):
        auc, ap, n_samples, _ = train_one_task_rf(i, t_name, train_data, val_data)

        print(f"   > {t_name:<25} | AUC: {auc:.4f} | N: {n_samples}")

        results.append({
            "Task": t_name,
            "ROC_AUC": float(auc),
            "PR_AUC": float(ap),
            "Samples": int(n_samples)
        })

    df_results = pd.DataFrame(results)

    save_results(df_results)

    print("\n" + "="*40)
    print(f"RF MEAN ROC-AUC: {df_results['ROC_AUC'].mean():.4f}")
    print(f"Wyniki zapisane w: {CFG.save_dir}")
    print("="*40)


if __name__ == "__main__":
    main()