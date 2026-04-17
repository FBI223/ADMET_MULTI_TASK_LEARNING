"""
EXPERIMENT_CROSS_UNCERTAINTY — Cross-Task Uncertainty-Weighted Multi-Task Learning

Rozszerzenie EXPERIMENT_UNCERTAINTY: zamiast 13 skalarów sigma (jeden per task),
uczona jest macierz 13×13 gdzie sigma[i,j] = niepewność taska j z perspektywy taska i.

Wzór:
    L_total = sum_i sum_j [ L_j / exp(log_sigma_sq[i,j]) + log_sigma_sq[i,j] / 2 ]

Efektywna waga gradientu taska j = suma kolumny j: sum_i 1/sigma[i,j]^2

Uruchomienie:
    python -m EXPERIMENT.train_cross_uncertainty
"""

import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from EXPERIMENT.config import Config
from EXPERIMENT.model import ADMET_Hybrid_Model
from EXPERIMENT.data import get_full_data
from EXPERIMENT.plots import (
    evaluate_per_task,
    evaluate_gnn_simple,
    plot_training_results,
    plot_all_roc_curves,
    plot_data_sparsity,
    plot_label_correlations,
    plot_model_comparison_simple,
)


# ============================================================
# PlotSaver
# ============================================================

class PlotSaver:
    def __init__(self, plots_dir: str, timestamp: str = None):
        self.plots_dir = plots_dir
        self.timestamp = timestamp or datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.counter = 0
        self._original_show = None
        os.makedirs(plots_dir, exist_ok=True)

    def __enter__(self):
        self._original_show = plt.show
        saver = self

        def _save_and_close():
            path = os.path.join(
                saver.plots_dir,
                f"plot_{saver.timestamp}_{saver.counter}.png"
            )
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saver.counter += 1

        plt.show = _save_and_close
        return self

    def __exit__(self, *_):
        plt.show = self._original_show


# ============================================================
# Cross-task uncertainty loss
# ============================================================

class CrossTaskUncertaintyLoss(nn.Module):
    """
    Macierz 13×13 uczalnych parametrów log_sigma_sq[i,j].
    log_sigma_sq[i,j] = log(sigma[i,j]^2), inicjalizowany na 0 (sigma=1).

    Dla każdej pary (i,j) wkład do straty:
        L_j / exp(log_sigma_sq[i,j]) + log_sigma_sq[i,j] / 2

    Clamping log_sigma_sq do [-4, 4] zapobiega kolapsowi (sigma w [0.14, 7.4]).
    """

    LOG_SIGMA_CLAMP = 4.0

    def __init__(self, n_tasks: int):
        super().__init__()
        self.n_tasks = n_tasks
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_tasks, n_tasks))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: list, targets: torch.Tensor):
        # Oblicz L_j dla każdego taska j (ze skalarem per task)
        task_losses = []
        for j, pred in enumerate(preds):
            mask = ~torch.isnan(targets[:, j])
            if mask.any():
                y_j = targets[mask, j].unsqueeze(1)
                p_j = pred[mask]
                task_losses.append(self.bce(p_j, y_j).mean())
            else:
                task_losses.append(torch.tensor(0.0, device=targets.device))

        # Suma po całej macierzy
        log_sq = torch.clamp(self.log_sigma_sq, -self.LOG_SIGMA_CLAMP, self.LOG_SIGMA_CLAMP)
        total_loss = torch.tensor(0.0, device=targets.device)

        for i in range(self.n_tasks):
            for j, l_j in enumerate(task_losses):
                precision = torch.exp(-log_sq[i, j])
                total_loss = total_loss + precision * l_j + log_sq[i, j] / 2.0

        return total_loss

    def get_sigma_matrix(self) -> np.ndarray:
        with torch.no_grad():
            log_sq = torch.clamp(self.log_sigma_sq, -self.LOG_SIGMA_CLAMP, self.LOG_SIGMA_CLAMP)
            return torch.exp(log_sq / 2.0).cpu().numpy()

    def get_effective_weights(self) -> np.ndarray:
        """Suma kolumnowa 1/sigma^2 — efektywna waga każdego taska."""
        sigma_mat = self.get_sigma_matrix()
        return (1.0 / (sigma_mat ** 2)).sum(axis=0)


# ============================================================
# Pomocnicze
# ============================================================

def _sync_dims(loader, cfg):
    batch = next(iter(loader))
    if cfg.use_graph and hasattr(batch, 'x'):
        cfg.node_dim = batch.x.shape[1]
    if cfg.use_morgan and hasattr(batch, 'morgan'):
        cfg.morgan_dim = batch.morgan.shape[1]
    if cfg.use_rdkit and hasattr(batch, 'rdkit'):
        cfg.rdkit_dim = batch.rdkit.shape[1]


def _save_plot(save_path: str, plot_fn):
    original_show = plt.show

    def _save_and_close():
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    plt.show = _save_and_close
    try:
        plot_fn()
    finally:
        plt.show = original_show


def _plot_sigma_heatmap(sigma_matrix: np.ndarray, tasks: list, save_path: str):
    _, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        sigma_matrix,
        xticklabels=tasks,
        yticklabels=tasks,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=1.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel('Task j (źródło straty)')
    ax.set_ylabel('Task i (perspektywa)')
    ax.set_title('Cross-Task Sigma Matrix\n'
                 'Zielony < 1.0 = task j pomaga taskowi i | Czerwony > 1.0 = koliduje')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# MTL z cross-task uncertainty
# ============================================================

def train_mtl(train_loader, val_loader, test_loader, config,
              results_dir: str, timestamp: str):
    model = ADMET_Hybrid_Model(config).to(config.device)
    criterion = CrossTaskUncertaintyLoss(n_tasks=len(config.tasks)).to(config.device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    history = {'train_loss': [], 'val_auroc': []}
    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    model_path = os.path.join(results_dir, 'best_mtl_model.pt')

    print(f"\n>>> Start treningu MTL z Cross-Task Uncertainty ({config.epochs} epok max)...")

    for epoch in range(config.epochs):
        model.train()
        criterion.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(config.device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        criterion.eval()
        val_scores = evaluate_gnn_simple(model, val_loader, config)
        avg_val_auc = np.mean(list(val_scores.values()))
        scheduler.step(avg_val_auc)

        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_auroc'].append(avg_val_auc)
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(train_loader):.4f} "
              f"| Val AUC: {avg_val_auc:.4f}")

        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'val_auc': best_val_auc,
            }, model_path)
            print(f"  >>> Nowy najlepszy model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  !!! Early Stopping po {epoch+1} epokach.")
                break

    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    print(f"\n>>> Wczytano model z epoki {checkpoint['epoch']+1} "
          f"(Val AUC: {checkpoint['val_auc']:.4f})")

    model.eval()
    y_true_parts, y_pred_parts = [], [[] for _ in config.tasks]
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.device)
            out = model(batch)
            y_true_parts.append(batch.y.cpu().numpy())
            for i in range(len(config.tasks)):
                y_pred_parts[i].extend(
                    torch.sigmoid(out[i]).cpu().numpy().flatten()
                )

    y_true_test = np.vstack(y_true_parts)
    sigma_matrix = criterion.get_sigma_matrix()
    effective_weights = criterion.get_effective_weights()
    return model, y_true_test, y_pred_parts, history, sigma_matrix, effective_weights


# ============================================================
# Jeden eksperyment (jedna kombinacja wejść)
# ============================================================

def _load_main_baselines(name: str, tasks: list) -> dict:
    """Wczytuje MTL_GNN, STL_GNN i XGBoost z EXPERIMENT_MAIN jeśli istnieje."""
    main_csv = os.path.join(
        os.path.dirname(__file__), 'EXPERIMENT_MAIN', name, 'final_results.csv'
    )
    if not os.path.exists(main_csv):
        return {}
    df = pd.read_csv(main_csv).set_index('Task')
    result = {}
    for col in ('MTL_GNN', 'STL_GNN', 'XGBoost'):
        if col in df.columns:
            result[col] = {t: df.loc[t, col] if t in df.index else float('nan') for t in tasks}
    return result


def run_single_experiment(cfg: Config, results_dir: str, name: str):
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'results'), exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_path = os.path.join(results_dir, 'run_stats.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    class Tee:
        def __init__(self, *s): self.s = s
        def write(self, m):
            for x in self.s: x.write(m); x.flush()
        def flush(self):
            for x in self.s: x.flush()

    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_file)

    try:
        print(f"Zadania ({len(cfg.tasks)}): {cfg.tasks}")
        print(f"Wejścia: graph={cfg.use_graph}, rdkit={cfg.use_rdkit}, morgan={cfg.use_morgan}")
        print(f"Device: {cfg.device}\n")

        train_loader, val_loader, test_loader, _, _, raw_train_df = get_full_data(cfg)
        _sync_dims(train_loader, cfg)
        print(f"Wymiary: node={cfg.node_dim}, morgan={cfg.morgan_dim}, rdkit={cfg.rdkit_dim}\n")

        with PlotSaver(os.path.join(results_dir, 'plots'), timestamp) as saver:
            plot_data_sparsity(raw_train_df, cfg.tasks)
            plot_label_correlations(raw_train_df, cfg.tasks)

            print(">>> MTL GNN + Cross-Task Uncertainty Weighting...")
            _, y_true_test, y_pred_test, mtl_history, sigma_matrix, eff_weights = train_mtl(
                train_loader, val_loader, test_loader, cfg, results_dir, timestamp
            )
            mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

            plot_all_roc_curves(y_true_test, np.array(y_pred_test).T, cfg.tasks)
            plot_training_results(mtl_history, title='MTL GNN Cross-Task Uncertainty')

        plot_count = saver.counter

        # Zapis macierzy sigma
        sigma_df = pd.DataFrame(sigma_matrix, index=cfg.tasks, columns=cfg.tasks)
        sigma_df.to_csv(os.path.join(results_dir, 'sigma_matrix_final.csv'))

        # Efektywne wagi per task (suma kolumn)
        colsum_df = pd.DataFrame({
            'Task':             cfg.tasks,
            'effective_weight': eff_weights,
        })
        colsum_df.to_csv(os.path.join(results_dir, 'sigma_colsum.csv'), index=False)

        print("\n>>> Efektywne wagi per task (suma kolumn 1/sigma^2):")
        for _, row in colsum_df.iterrows():
            print(f"  {row['Task']:<35} eff_weight = {row['effective_weight']:.4f}")

        _plot_sigma_heatmap(sigma_matrix, cfg.tasks,
                            os.path.join(results_dir, 'sigma_heatmap.png'))

        # Baselines z EXPERIMENT_MAIN
        baselines = _load_main_baselines(name, cfg.tasks)
        main_mtl   = baselines.get('MTL_GNN', {})
        stl_scores = baselines.get('STL_GNN', {})
        xgb_scores = baselines.get('XGBoost', {})
        if baselines:
            print(f"\n>>> Wczytano baseline z EXPERIMENT_MAIN/{name}/final_results.csv")
        else:
            print(f"\n>>> Brak baseline EXPERIMENT_MAIN/{name} — kolumny MTL_GNN, STL_GNN i XGBoost będą puste.")

        results_df = pd.DataFrame({
            'Task':               cfg.tasks,
            'MTL_GNN_CROSS':      [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
            'MTL_GNN':            [main_mtl.get(t,   float('nan')) for t in cfg.tasks],
            'STL_GNN':            [stl_scores.get(t, float('nan')) for t in cfg.tasks],
            'XGBoost':            [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
        })
        csv_path = os.path.join(results_dir, 'final_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n>>> Wyniki:\n{results_df.to_string(index=False)}")

        comparison_path = os.path.join(results_dir, f"plot_{timestamp}_{plot_count}.png")
        _save_plot(comparison_path, lambda: plot_model_comparison_simple(results_df))
        print(f"\nGOTOWE. Wyniki: {csv_path}")

    finally:
        sys.stdout = orig_stdout
        log_file.close()


# ============================================================
# MAIN — 4 kombinacje jak w EXPERIMENT_MAIN
# ============================================================

COMBINATIONS = [
    (True,  False, False, 'GNN'),
    (True,  False, True,  'GNN_MORGAN'),
    (True,  True,  False, 'GNN_RDKIT'),
    (True,  True,  True,  'GNN_RDKIT_MORGAN'),
]

BASE_DIR = os.path.join(os.path.dirname(__file__), 'EXPERIMENT_CROSS_UNCERTAINTY')


def main():
    print(">>> Przygotowanie Master Cache...")
    cache_cfg = Config()
    cache_cfg.use_graph = cache_cfg.use_rdkit = cache_cfg.use_morgan = True
    get_full_data(cache_cfg)
    print(">>> Cache gotowy.\n")

    for use_graph, use_rdkit, use_morgan, name in COMBINATIONS:
        print(f"\n{'='*60}\n  EKSPERYMENT: {name}\n{'='*60}")

        cfg = Config()
        cfg.tasks = cfg.tasks_old
        cfg.use_graph = use_graph
        cfg.use_rdkit = use_rdkit
        cfg.use_morgan = use_morgan

        results_dir = os.path.join(BASE_DIR, name)
        run_single_experiment(cfg, results_dir, name)


if __name__ == '__main__':
    main()
