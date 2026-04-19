import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc


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