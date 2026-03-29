import numpy as np
import pandas as pd
from rdkit import Chem
from tdc.single_pred import ADME, Tox

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



# -------------------------
# 1. BASIC INFO
# -------------------------
def dataset_overview(df):
    print("TOTAL SAMPLES:", len(df))
    print("TASKS:", df.task.unique())
    print("SPLITS:", df.split.unique())
    print()

    print(df.groupby(["task","split"]).size())
    print()

# -------------------------
# 2. LABEL DISTRIBUTION
# -------------------------
def label_stats(df):
    print("\n=== LABEL DISTRIBUTION ===")
    for t in df.task.unique():
        sub = df[df.task==t]
        y = (sub.label.values > 0).astype(int)

        print(f"\n{t}")
        print("total:", len(y))
        print("pos:", y.sum())
        print("neg:", (y==0).sum())
        print("ratio:", y.mean())

# -------------------------
# 3. UNIQUE MOLECULES
# -------------------------
def unique_molecules(df):
    print("\n=== UNIQUE MOLECULES ===")
    print("total smiles:", len(df))
    print("unique smiles:", df.smiles.nunique())

    for split in ["train","valid","test"]:
        sub = df[df.split==split]
        print(f"{split}: {sub.smiles.nunique()}")

# -------------------------
# 4. SPLIT LEAKAGE
# -------------------------
def check_split_leakage(df):
    print("\n=== SPLIT LEAKAGE ===")

    train_sm = set(df[df.split=="train"].smiles)
    val_sm   = set(df[df.split=="valid"].smiles)
    test_sm  = set(df[df.split=="test"].smiles)

    print("train ∩ val:", len(train_sm & val_sm))
    print("train ∩ test:", len(train_sm & test_sm))
    print("val ∩ test:", len(val_sm & test_sm))

# -------------------------
# 5. TASK OVERLAP
# -------------------------
def task_overlap(df):
    print("\n=== TASK OVERLAP ===")

    tasks = df.task.unique()

    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]

            s1 = set(df[df.task==t1].smiles)
            s2 = set(df[df.task==t2].smiles)

            overlap = len(s1 & s2)

            print(f"{t1} vs {t2}: {overlap}")

# -------------------------
# 6. INVALID SMILES
# -------------------------
def check_invalid_smiles(df):
    print("\n=== INVALID SMILES ===")

    bad = 0
    for sm in df.smiles:
        if Chem.MolFromSmiles(sm) is None:
            bad += 1

    print("invalid:", bad)

# -------------------------
# 7. DUPLICATES
# -------------------------
def check_duplicates(df):
    print("\n=== DUPLICATES ===")

    dup = df.duplicated(subset=["smiles","task"]).sum()
    print("duplicates (same task):", dup)

    dup_all = df.duplicated(subset=["smiles"]).sum()
    print("duplicates (global):", dup_all)

# -------------------------
# 8. FEATURE NAN CHECK
# -------------------------
def check_feature_nan(df, build_features, feat="rdkit", sample=1000):
    print("\n=== FEATURE NaN CHECK ===")

    df_sub = df.sample(min(sample, len(df)))
    X = build_features(df_sub, feat)

    print("NaN:", np.isnan(X).any())
    print("Inf:", np.isinf(X).any())


import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# 1. LABEL HISTOGRAM (per task)
# -------------------------
def plot_label_distribution(df):
    df_plot = df.copy()
    df_plot["label_bin"] = (df_plot["label"] > 0).astype(int)

    plt.figure(figsize=(10,5))
    sns.countplot(data=df_plot, x="task", hue="label_bin")
    plt.title("Label distribution per task")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------
# 2. SPLIT SIZE HISTOGRAM
# -------------------------
def plot_split_sizes(df):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="split", hue="task")
    plt.title("Split sizes per task")
    plt.tight_layout()
    plt.show()


# -------------------------
# 3. MOLECULE LENGTH (SMILES)
# -------------------------
def plot_smiles_length(df):
    df_plot = df.copy()
    df_plot["len"] = df_plot["smiles"].apply(len)

    plt.figure(figsize=(8,5))
    sns.histplot(df_plot["len"], bins=50)
    plt.title("SMILES length distribution")
    plt.show()



# -------------------------
# 9. SPLIT BALANCE PER TASK
# -------------------------
def split_balance(df):
    print("\n=== SPLIT BALANCE PER TASK ===")

    for t in df.task.unique():
        sub = df[df.task==t]

        print(f"\n{t}")
        for s in ["train","valid","test"]:
            n = len(sub[sub.split==s])
            print(s, n)


# -------------------------
# 8. FEATURE DISTRIBUTION (RDKit example)
# -------------------------


# -------------------------
# 7. TASK OVERLAP HEATMAP
# -------------------------
def compute_label_correlation(df):
    print("\n=== LABEL CORRELATION BETWEEN TASKS ===")

    tasks = df.task.unique()
    mat = pd.DataFrame(index=tasks, columns=tasks)

    # tylko TRAIN (bez leakage)
    df = df[df.split=="train"]

    for t1 in tasks:
        for t2 in tasks:

            d1 = df[df.task==t1][["smiles","label"]]
            d2 = df[df.task==t2][["smiles","label"]]

            merged = d1.merge(d2, on="smiles")

            if len(merged) > 20:
                y1 = (merged.label_x.values > 0).astype(int)
                y2 = (merged.label_y.values > 0).astype(int)

                mat.loc[t1, t2] = np.corrcoef(y1, y2)[0,1]

    mat = mat.astype(float)
    print(mat)

    return mat

def plot_label_corr(mat):
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Label correlation between tasks")
    plt.show()

def plot_task_overlap_heatmap(df):
    tasks = df.task.unique()
    mat = pd.DataFrame(0, index=tasks, columns=tasks)

    for t1 in tasks:
        s1 = set(df[df.task==t1].smiles)
        for t2 in tasks:
            s2 = set(df[df.task==t2].smiles)
            mat.loc[t1,t2] = len(s1 & s2)

    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, cmap="viridis")
    plt.title("Task overlap (shared molecules)")
    plt.show()

def plot_feature_distribution(df, build_features):
    X = build_features(df.sample(min(500, len(df))), "rdkit")

    plt.figure(figsize=(8,5))
    sns.histplot(X.flatten(), bins=100)
    plt.title("Feature value distribution")
    plt.show()

def run_data_audit():
    tasks = [
        "cyp2d6_veith",
        "cyp3a4_veith",
        "cyp2c9_veith",
        "herg_karim",
        "pgp_broccatelli"
    ]

    df_all, _ = load_tdc_tasks(tasks)

    mat = compute_label_correlation(df_all)
    plot_label_corr(mat)

    # tekstowe
    dataset_overview(df_all)
    label_stats(df_all)
    check_split_leakage(df_all)
    task_overlap(df_all)

    # wizualizacje
    plot_label_distribution(df_all)
    plot_split_sizes(df_all)
    plot_smiles_length(df_all)


    plot_task_overlap_heatmap(df_all)




if __name__ == "__main__":
    run_data_audit()
