


import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def plot_pca(embeddings, labels, tasks, save_path):
    pca = PCA(n_components=2)
    X = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    # Używamy palety tab20, aby rozróżnić 13 zadań ADMET
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20', s=15, alpha=0.6)

    # Dodanie legendy z nazwami zadań
    legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=tasks,
                         title="ADMET Tasks", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(legend1)

    plt.title("PCA of Molecular Representations (QW-MTL)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne(embeddings, labels, tasks, save_path):
    # Perplexity 30-50 jest optymalne dla wizualizacji klastrów molekularnych [cite: 278, 395]
    tsne = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto')
    X = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20', s=15, alpha=0.6)

    plt.legend(handles=scatter.legend_elements()[0], labels=tasks,
               title="ADMET Tasks", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE Visualization of Shared Latent Space")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()





def plot_pr(preds, targets, mask, tasks, save_dir):
    preds = torch.sigmoid(preds).cpu().numpy()
    targets = targets.cpu().numpy()
    mask = mask.cpu().numpy()

    for i, t in enumerate(tasks):
        valid = mask[:,i] == 1
        if valid.sum() < 10: continue

        p, r, _ = precision_recall_curve(targets[valid,i], preds[valid,i])

        plt.plot(r, p)
        plt.title(t)
        plt.savefig(f"{save_dir}/pr_{t}.png")
        plt.close()





def plot_training(history, save_path):
    """Generuje wykresy strat (Loss) oraz metryk (Mean Score)[cite: 116]."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Wykres Loss (Train vs Val)
    ax[0].plot(history["train_loss"], label="Train Loss")
    ax[0].plot(history["val_loss"], label="Val Loss")
    ax[0].set_title("Multi-Task Loss Convergence")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # Wykres Mean Score (AUROC/AUPRC)
    ax[1].plot(history["train_score"], label="Train Mean Score")
    ax[1].plot(history["val_score"], label="Val Mean Score")
    ax[1].set_title("Mean Performance across 13 Tasks")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Score")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_task_correlation(preds, targets, mask, tasks, save_path):
    """Poprawiona wersja: liczy korelację na wszystkich predykcjach modelu."""
    # Obliczamy sigmoidę, aby przejść do prawdopodobieństw [cite: 102]
    preds_np = torch.sigmoid(preds).detach().cpu().numpy()

    # Liczymy korelację Spearmana między kolumnami predykcji
    from scipy.stats import spearmanr
    corr_matrix, _ = spearmanr(preds_np)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, xticklabels=tasks, yticklabels=tasks,
                annot=True, fmt=".2f", cmap='RdBu_r', vmin=-1, vmax=1, center=0)
    plt.title("Inter-Task Prediction Correlation (Shared Latent Knowledge)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_beta_vs_samples(model, train_data, tasks, save_path):
    """Wizualizuje korelację wykładnika beta z rozmiarem danych (Figure 3)[cite: 221, 271]."""
    # Zliczanie próbek na zadanie
    counts = np.zeros(len(tasks))
    for d in train_data:
        counts += d.mask

    betas = torch.nn.functional.softplus(model.log_beta).detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(counts, betas)

    # Linia trendu (Pearson r ~ 0.95 w artykule) [cite: 218]
    if len(counts) > 1:
        m, b = np.polyfit(counts, betas, 1)
        plt.plot(counts, m * counts + b, color='red', alpha=0.5, label=f'Fit (r={pearsonr(counts, betas)[0]:.2f})')

    plt.title(r"Correlation between Task Sample Size and Learned $\beta_t$")
    plt.xlabel("Task Sample Size")
    plt.ylabel(r"Learned $\beta$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()


def plot_roc(preds, targets, mask, tasks, save_dir):
    """Rysuje krzywe ROC dla wszystkich zadań klasyfikacji ADMET[cite: 170]."""
    plt.figure(figsize=(10, 8))
    preds_np = torch.sigmoid(preds).numpy()
    targets_np = targets.numpy()

    for i, task in enumerate(tasks):
        valid = mask[:, i] == 1
        if valid.sum() > 0:
            fpr, tpr, _ = roc_curve(targets_np[valid, i], preds_np[valid, i])
            plt.plot(fpr, tpr, label=f'{task} (AUC={auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Task ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png")
    plt.close()


def plot_roc_combined(preds, targets, mask, tasks, save_path):
    plt.figure(figsize=(10, 8))
    preds_np = torch.sigmoid(preds).numpy()
    targets_np = targets.numpy()

    for i, task in enumerate(tasks):
        valid = mask[:, i] == 1
        if valid.sum() > 10:
            fpr, tpr, _ = roc_curve(targets_np[valid, i], preds_np[valid, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{task} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves for all ADMET Tasks')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
