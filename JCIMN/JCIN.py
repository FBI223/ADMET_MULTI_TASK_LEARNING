import deepchem as dc
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import subprocess
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from deepchem.splits import ScaffoldSplitter

import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("data", exist_ok=True)
os.makedirs("cache", exist_ok=True)

def load_dataset(name):
    if name == "tox21":
        return dc.molnet.load_tox21()
    elif name == "pcba":
        return dc.molnet.load_pcba()
    elif name == "muv":
        return dc.molnet.load_muv()
    elif name == "toxcast":
        return dc.molnet.load_toxcast()
    elif name == "sider":
        return dc.molnet.load_sider()

    # regression
    elif name == "esol":
        return dc.molnet.load_delaney()
    elif name == "freesolv":
        return dc.molnet.load_sampl()
    elif name == "lipophilicity":
        return dc.molnet.load_lipo()
    elif name == "qm9":
        return dc.molnet.load_qm9()

    # single-task classification
    elif name == "hiv":
        return dc.molnet.load_hiv()
    elif name == "bace":
        return dc.molnet.load_bace_classification()
    elif name == "bbbp":
        return dc.molnet.load_bbbp()
    elif name == "clintox":
        return dc.molnet.load_clintox()

    else:
        raise ValueError(name)




def clean_features(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # clipping (ważne!)
    X = np.clip(X, -1e6, 1e6)

    return X

def cdf_normalize(X):
    X = clean_features(X)

    qt = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=min(1000, X.shape[0])
    )

    return qt.fit_transform(X)


def cdf_fit_transform(X_train, X_test):
    qt = QuantileTransformer(output_distribution="uniform")
    X_train = qt.fit_transform(X_train)
    X_test  = qt.transform(X_test)
    return X_train, X_test


def dc_to_df(dataset, tasks):
    df = pd.DataFrame({"smiles": dataset.ids})
    y = dataset.y

    for i, t in enumerate(tasks):
        df[t] = y[:, i]

    return df


def scaffold_split(dataset):
    splitter = ScaffoldSplitter()
    return splitter.train_valid_test_split(dataset)

def smiles_to_morgan(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None   # ❗ zamiast zeros

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array(fp)


def smiles_to_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None   # ❗

    desc = []
    for name, func in Descriptors.descList:
        try:
            desc.append(func(mol))
        except:
            desc.append(0)
    return np.array(desc)

def remove_bad_columns(X):
    mask = np.isfinite(X).all(axis=0)
    return X[:, mask]


def build_features(smiles_list, cache_name=None):
    if cache_name and os.path.exists(cache_name):
        data = np.load(cache_name, allow_pickle=True).item()
        return data["X"], data["mask"], data["idx"]

    X = []
    valid_idx = []

    for i, s in enumerate(smiles_list):
        morgan = smiles_to_morgan(s)
        rdkit  = smiles_to_rdkit(s)

        if morgan is None or rdkit is None:
            continue

        X.append(np.concatenate([morgan, rdkit]))
        valid_idx.append(i)

    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    mask = np.isfinite(X).all(axis=0)
    X = X[:, mask]

    if cache_name:
        np.save(cache_name, {
            "X": X,
            "mask": mask,
            "idx": valid_idx,
        })



    return X, mask, valid_idx








def save_chemprop_csv(df, path):
    df.to_csv(path, index=False)



def train_rf(X_train, y_train, task_type):
    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            class_weight="balanced"
        ).fit(X_train, y_train)
    else:
        return RandomForestRegressor(
            n_estimators=500,
            n_jobs=-1
        ).fit(X_train, y_train)



def evaluate(model, X, y, task_type):
    preds = model.predict_proba(X)[:,1] if task_type=="classification" else model.predict(X)

    mask = ~np.isnan(y)

    if task_type == "classification":
        return roc_auc_score(y[mask], preds[mask])
    else:
        return mean_squared_error(y[mask], preds[mask], squared=False)






def train_chemprop_multitask(train_csv, valid_csv, test_csv, dataset_name):

    subprocess.run([
        "chemprop_train",
        "--data_path", train_csv,
        "--separate_val_path", valid_csv,
        "--separate_test_path", test_csv,
        "--dataset_type", "classification",
        #"--split_type", "scaffold_balanced",

        # ✔ RDKit features (paper)
        "--features_generator", "rdkit_2d_normalized",
        "--no_features_scaling",

        # ✔ większy model
        "--hidden_size", "300",
        "--depth", "3",

        # ✔ ensemble
        "--ensemble_size", "5",

        "--epochs", "30",
        "--metric", "auc",
        "--save_dir", f"ckpt_{dataset_name}"
    ] , check=True )





def predict_chemprop(test_csv, dataset_name):
    out = f"preds_{dataset_name}.csv"

    subprocess.run([
        "chemprop_predict",
        "--test_path", test_csv,
        "--checkpoint_dir", f"ckpt_{dataset_name}",
        "--preds_path", out,

        # 🔥 MUSI BYĆ IDENTYCZNE JAK W TRAIN
        "--features_generator", "rdkit_2d_normalized",
        "--no_features_scaling"
    ], check=True)

    if not os.path.exists(out):
        raise RuntimeError("Chemprop failed → no predictions file")

    return pd.read_csv(out)

def eval_multitask(df_true, df_pred):
    scores = []

    for col in df_true.columns:
        if col == "smiles":
            continue

        if col in df_pred.columns:
            mask = ~df_true[col].isna()

            if mask.sum() > 0:
                score = roc_auc_score(df_true[col][mask], df_pred[col][mask])
                scores.append(score)

    return np.mean(scores)


def run_single_task(df_train, df_test,
                    X_train_all, X_test_all,
                    idx_train_all, idx_test_all,
                    dataset_name):

    results = []
    tasks = list(df_train.columns)
    tasks.remove("smiles")

    for t in tasks:
        print(f"{dataset_name} - {t}")

        y_train = df_train[t].values[idx_train_all]
        y_test  = df_test[t].values[idx_test_all]

        X_train = X_train_all
        X_test  = X_test_all

        task_type = "classification" if set(np.unique(y_train[~np.isnan(y_train)])) <= {0,1} else "regression"

        model = train_rf(X_train, y_train, task_type)
        score = evaluate(model, X_test, y_test, task_type)

        results.append(score)

    return np.mean(results)


def run_all_models():
    datasets = ["tox21", "bbbp"]
    results = []





    for name in datasets:
        print(f"\n=== {name} ===")

        tasks, (dataset, _, _), _ = load_dataset(name)

        train, valid, test = scaffold_split(dataset)

        df_train = dc_to_df(train, tasks)
        df_valid = dc_to_df(valid, tasks)
        df_test  = dc_to_df(test, tasks)

        # ===== FEATURE CACHE (RAZ NA DATASET) =====
        X_train_all, mask_all, idx_train_all = build_features(
            df_train["smiles"],
            cache_name=f"cache/{name}_train.npy"
        )

        X_test_all, _, idx_test_all = build_features(
            df_test["smiles"],
            cache_name=f"cache/{name}_test.npy"
        )

        X_test_all = X_test_all[:, mask_all]

        # ===== GLOBAL CDF =====
        qt = QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=min(1000, X_train_all.shape[0])
        )

        qt.fit(X_train_all)


        np.save(f"cache/{name}_qt.npy", qt)

        X_train_all = qt.transform(X_train_all)
        X_test_all = qt.transform(X_test_all)

        # RF
        rf_score = run_single_task(
            df_train, df_test,
            X_train_all, X_test_all,
            idx_train_all, idx_test_all,
            name
        )

        # Chemprop
        train_csv = f"data/{name}_train.csv"
        valid_csv = f"data/{name}_valid.csv"
        test_csv  = f"data/{name}_test.csv"

        save_chemprop_csv(df_train, train_csv)
        save_chemprop_csv(df_valid, valid_csv)
        save_chemprop_csv(df_test, test_csv)

        train_chemprop_multitask(train_csv, valid_csv, test_csv, name)
        preds = predict_chemprop(test_csv, name)

        mt_score = eval_multitask(df_test, preds)

        results.append({
            "dataset": name,
            "RF_single": rf_score,
            "DMPNN_multitask": mt_score
        })

    return pd.DataFrame(results)

def visualize_embeddings(smiles_list):

    X, _, _ = build_features(smiles_list)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], s=5)
    plt.title("PCA embeddings")
    plt.show()

    # TSNE
    tsne = TSNE(n_components=2, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(X_tsne[:,0], X_tsne[:,1], s=5)
    plt.title("t-SNE embeddings")
    plt.show()

def plot_results(results):
    df = pd.DataFrame(list(results.items()), columns=["dataset","score"])

    plt.figure()
    sns.barplot(data=df, x="dataset", y="score")
    plt.xticks(rotation=45)
    plt.title("Model performance per dataset")
    plt.show()



def plot_benchmark(df):
    df_melt = df.melt(id_vars="dataset")

    plt.figure(figsize=(10,5))
    sns.barplot(data=df_melt, x="dataset", y="value", hue="variable")
    plt.xticks(rotation=45)
    plt.title("RF vs D-MPNN multitask")
    plt.show()


def plot_feature_corr(smiles_list):
    X, _, _ = build_features(smiles_list)

    corr = np.corrcoef(X.T)

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature correlation")
    plt.show()


if __name__ == "__main__":
    df = run_all_models()

    print(df)

    plot_benchmark(df)

    # embedding analysis na jednym dataset
    tasks, (train, _, _), _ = load_dataset("tox21")
    smiles = train.ids[:1000]

    visualize_embeddings(smiles)
    plot_feature_corr(smiles)
