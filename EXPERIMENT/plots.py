import pandas as pd
import seaborn as sns
import torch
import torch_geometric
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore


def plot_stl_correlation(all_preds, config):
    plot_task_correlation(all_preds, config.tasks)

def plot_feature_importance_xgboost(xgb_model, feature_names, top_n=20):
    """Pokazuje, które cechy (np. konkretne deskryptory RDKit) były kluczowe."""
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} najważniejszych cech (XGBoost)")
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def remove_outliers(X, Y, z_thresh=5.0):


    # 🔥 stabilizacja danych
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -100, 100)

    z = np.abs(zscore(X, axis=0))

    # 🔥 NIE .all()
    mask = np.mean(z < z_thresh, axis=1) > 0.99

    return X[mask], Y[mask]



def plot_tsne(X, Y, task_idx, save_path, title):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    if X.shape[0] == 0:
        print("Brak danych:", title)
        return

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -100, 100)

    X = StandardScaler().fit_transform(X)

    X_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    y = Y[:, task_idx]
    mask = ~np.isnan(y)

    y = y[mask]
    X_2d = X_2d[mask]

    # 🔥 binarizacja jeśli nie binary
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(int)

    print("classes:", np.unique(y))

    plt.figure(figsize=(6,5))

    for cls in np.unique(y):
        idx = y == cls
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=f"class {cls}",
            s=10
        )

    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne_scatter(X, Y, task_idx, save_path, title):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    if X.shape[0] == 0:
        print("Brak danych:", title)
        return

    # 🔥 KLUCZOWE
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -100, 100)

    X = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    y = Y[:, task_idx]
    mask = ~np.isnan(y)

    plt.figure(figsize=(6,5))
    plt.scatter(X_2d[mask,0], X_2d[mask,1], c=y[mask], cmap="coolwarm", s=10)
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_morgan(dataset):
    X, Y = [], []
    for d in dataset:
        if hasattr(d, "morgan"):
            X.append(d.morgan.numpy())
            Y.append(d.y.numpy())

    if len(X) == 0:
        raise ValueError("Morgan: brak próbek")

    return np.vstack(X), np.vstack(Y)

def extract_rdkit(dataset):
    X, Y = [], []
    for d in dataset:
        if hasattr(d, "rdkit") and d.rdkit is not None:
            X.append(d.rdkit.numpy())
            Y.append(d.y.numpy())

    if len(X) == 0:
        raise ValueError("RDKit: brak próbek")

    return np.vstack(X), np.vstack(Y)

def tsne_morgan(dataset, task_idx, save_path):
    X, Y = extract_morgan(dataset)
    plot_tsne(X, Y, task_idx, save_path, "t-SNE Morgan")

def tsne_rdkit(dataset, task_idx, save_path):
    X, Y = extract_rdkit(dataset)

    # 🔥 usuń outliery
    print("RDKit BEFORE:", X.shape)
    X2, Y2 = remove_outliers(X, Y)
    print("RDKit AFTER:", X2.shape)

    plot_tsne(X2, Y2, task_idx, save_path, "t-SNE RDKit")


def tsne_gnn(model, loader, config, task_idx, save_path):
    X, Y = extract_gnn_embeddings(model, loader, config)
    plot_tsne(X, Y, task_idx, save_path, "t-SNE GNN embeddings")


def extract_gnn_embeddings(model, loader, config):
    model.eval()
    all_emb = []
    all_y = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)

            # zakładamy że model ma encoder / backbone
            x = batch.x
            edge_index = batch.edge_index

            for conv in model.gin_backbone:
                x = conv(x, edge_index)

            # global pooling
            emb = torch_geometric.nn.global_mean_pool(x, batch.batch)

            all_emb.append(emb.cpu().numpy())
            all_y.append(batch.y.cpu().numpy())

    return np.vstack(all_emb), np.vstack(all_y)


def plot_rho_vs_delta(corr_path="korelacje.csv",
                      results_path="final_results.csv",
                      save_path="rho_vs_delta.png"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # --- dane ---
    corr = pd.read_csv(corr_path, index_col=0)
    results = pd.read_csv(results_path)

    tasks = corr.columns.tolist()

    # --- rho ---
    rho = {t: corr.loc[t].drop(t).mean() for t in tasks}

    # --- delta ---
    delta = {
        row["Task"]: row["MTL_GNN"] - row["STL_GNN"]
        for _, row in results.iterrows()
    }

    # --- wektory ---
    rho_vals = [rho[t] for t in tasks if t in delta]
    delta_vals = [delta[t] for t in tasks if t in delta]
    labels = [t for t in tasks if t in delta]

    # --- plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(rho_vals, delta_vals)

    for i, t in enumerate(labels):
        plt.text(rho_vals[i], delta_vals[i], t, fontsize=8)

    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")

    # --- trend ---
    if len(rho_vals) > 1:
        z = np.polyfit(rho_vals, delta_vals, 1)
        p = np.poly1d(z)
        plt.plot(rho_vals, p(rho_vals), linestyle="--")

    plt.xlabel("rho")
    plt.ylabel("delta")

    # --- zapis ---
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()




def plot_performance_vs_size(results_dict, dataset_sizes):
    """
    results_dict: {task_name: delta_auroc} (różnica między MTL a STL)
    dataset_sizes: {task_name: n_samples}
    """
    tasks = list(results_dict.keys())
    deltas = [results_dict[t] for t in tasks]
    sizes = [dataset_sizes[t] for t in tasks]

    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, deltas, s=100, color='purple', alpha=0.6)

    for i, task in enumerate(tasks):
        plt.annotate(task, (sizes[i], deltas[i]), xytext=(5, 5), textcoords='offset points')

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel("Liczba próbek w zbiorze (Log Scale)")
    plt.xscale('log')
    plt.ylabel("Zysk z MTL (AUROC MTL - AUROC STL)")
    plt.title("Zależność: Czy MTL pomaga najbardziej małym zbiorom?")
    plt.show()






def plot_label_correlations(df, tasks):
    """
    Sprawdza, czy w danych surowych istnieje korelacja między etykietami.
    Wskazuje na biologiczne powiązania, które model może wykorzystać.
    """
    plt.figure(figsize=(10, 8))
    # Używamy korelacji Spearmana, bo etykiety są binarne/skokowe
    corr = df[tasks].corr(method='spearman')

    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
    plt.title("Biologiczna korelacja między zadaniami ADMET (Dane surowe)")
    plt.show()


def plot_all_roc_curves(y_true, y_pred_probs, tasks):
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(tasks))

    for i, task in enumerate(tasks):
        # Maskowanie obu: tam gdzie etykieta nie jest NaN ORAZ gdzie predykcja nie jest NaN
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred_probs[:, i])

        if mask.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true[mask, i], y_pred_probs[mask, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{task} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywe ROC dla wszystkich zadań ADMET')
    plt.legend(loc="lower right")
    plt.show()

def plot_task_correlation(y_pred_probs, tasks):
    """Pokazuje korelację między predykcjami modelu dla różnych zadań."""
    preds_df = pd.DataFrame(y_pred_probs, columns=tasks)
    corr = preds_df.corr()

    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelacja predykcji między zadaniami\n(Wykrywanie synergii w Multi-Task)")
    plt.show()


def plot_multi_task_confusion_matrices(y_true, y_pred_probs, tasks, threshold=0.5):
    """
    Rysuje siatkę macierzy pomyłek dla wszystkich zadań.
    """
    n_tasks = len(tasks)
    cols = 3
    rows = (n_tasks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        mask = ~np.isnan(y_true[:, i])
        true = y_true[mask, i]
        pred = (y_pred_probs[mask, i] > threshold).astype(int)

        cm = confusion_matrix(true, pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
        axes[i].set_title(f"Confusion Matrix: {task}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Ukrywamy puste osie
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_df):
    """
    results_df: DataFrame z kolumnami ['Task', 'AUROC', 'Model_Type']
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(x='Task', y='AUROC', hue='Model_Type', data=results_df, palette='viridis')
    plt.axhline(0.5, ls='--', color='red', alpha=0.5, label='Random')
    plt.title("Porównanie skuteczności modeli dla poszczególnych zadań")
    plt.ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_data_sparsity(df, tasks):
    """Pokazuje, które cząsteczki mają etykiety dla danych zadań."""
    plt.figure(figsize=(10, 6))
    # 1 oznacza obecność danych, 0 oznacza NaN
    sparsity_matrix = df[tasks].notnull().astype(int)
    sns.heatmap(sparsity_matrix, cmap="YlGnBu", cbar=False, yticklabels=False)
    plt.title("Macierz obecności etykiet (Białe = Brak danych)")
    plt.xlabel("Zadania ADMET")
    plt.ylabel("Cząsteczki")
    plt.show()

def plot_training_results(history, title="Training History"):
    """Rysuje wykresy straty i AUROC."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax[0].plot(history['train_loss'], label='Train Loss')
    ax[0].set_title(f'{title} - Loss')
    ax[0].legend()

    # AUROC plot
    ax[1].plot(history['val_auroc'], label='Val AUROC', color='orange')
    ax[1].set_title(f'{title} - Mean AUROC')
    ax[1].legend()

    plt.show()


def evaluate_per_task(y_true, y_pred, tasks):
    """
    y_true: macierz numpy [N, num_tasks] z wartościami NaN
    y_pred: macierz numpy [N, num_tasks] (lub lista list) z predykcjami (0-1)
    """
    scores = {}
    # Jeśli y_pred to lista, zamień na macierz numpy dla łatwego indeksowania
    y_pred = np.array(y_pred)
    if y_pred.shape[0] == len(tasks):  # obsługa Twojego formatu z main()
        y_pred = y_pred.T

    for i, task in enumerate(tasks):
        # 1. Stwórz maskę tam, gdzie mamy prawdziwe dane
        mask = ~np.isnan(y_true[:, i])

        y_t = y_true[mask, i]
        y_p = y_pred[mask, i]

        # 2. Sprawdź, czy mamy oba typy klas (wymagane dla ROC AUC)
        if len(np.unique(y_t)) > 1:
            # Upewnij się, że y_p nie zawiera NaN (błąd modelu)
            if np.isnan(y_p).any():
                print(f"Warning: Model wygenerował NaN dla zadania {task}!")
                scores[task] = 0.5
            else:
                scores[task] = roc_auc_score(y_t, y_p)
        else:
            print(f"Skipping {task}: tylko jedna klasa w secie testowym.")
            scores[task] = 0.5

    return scores



def evaluate_gnn_simple(model, loader, config):
    """Pomocnicza funkcja do walidacji w pętli treningowej"""
    model.eval()
    y_true, y_pred = [], [[] for _ in config.tasks]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            out = model(batch)
            y_true.append(batch.y.cpu().numpy())
            for i in range(len(config.tasks)):
                y_pred[i].extend(torch.sigmoid(out[i]).cpu().numpy().flatten())
    return evaluate_per_task(np.vstack(y_true), y_pred, config.tasks)


def plot_experiment_comparison(df, save_path):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Sortowanie od najlepszego wyniku
    df_plot = df.sort_values("Avg_AUROC", ascending=False)

    ax = sns.barplot(x="Experiment", y="Avg_AUROC", data=df_plot, palette="viridis")

    # Dodanie etykiet z wartościami
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold')

    plt.title("Porównanie 6 kombinacji reprezentacji molekularnych (MTL)", fontsize=15)
    plt.ylabel("Średni AUROC na zbiorze testowym", fontsize=12)
    plt.xlabel("Kombinacja (G: Graph, R: RDKit, M: Morgan)", fontsize=12)
    plt.ylim(0.5, 1.0)  # Skala AUROC od losowego do idealnego

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "final_comparison_chart.png"), dpi=300)
    plt.show()

def plot_model_comparison_simple(df):
    """Proste porównanie MTL vs XGBoost"""
    df_melted = df.melt(id_vars='Task', var_name='Model', value_name='AUROC')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Task', y='AUROC', hue='Model')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    plt.show()


def plot_single_roc(y_true, y_pred, task_name, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)

    if len(set(y_true)) < 2:
        print(f"[WARN] ROC nie może być policzone dla {task_name} (1 klasa)")
        return

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {task_name}")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_dir, f"roc_{task_name}.png"))
    plt.close()

def plot_all_stl_roc(stl_preds_dict):
    plt.figure()

    for task, (y_true, y_pred) in stl_preds_dict.items():
        if len(set(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{task} ({roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.title("STL ROC curves")

    plt.savefig("results/stl_all_roc.png")
    plt.close()

def evaluate_single_task(model, loader, task_idx, config):
    """
    Ewaluacja modelu GNN na konkretnym jednym zadaniu (dla STL).
    """
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            # Wybieramy tylko wyjście z głowicy o indeksie task_idx
            preds = torch.sigmoid(model(batch)[task_idx]).cpu().numpy().flatten()
            targets = batch.y[:, task_idx].cpu().numpy().flatten()

            mask = ~np.isnan(targets)
            y_true_all.extend(targets[mask])
            y_pred_all.extend(preds[mask])

    if len(np.unique(y_true_all)) > 1:
        return roc_auc_score(y_true_all, y_pred_all)
    return 0.5