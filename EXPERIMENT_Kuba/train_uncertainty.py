"""
EXPERIMENT_UNCERTAINTY — Homoscedastic Uncertainty-Weighted Multi-Task Learning

Jedyna zmiana względem EXPERIMENT_MAIN: funkcja kosztu.
Zamiast równego ważenia zadań, uczony jest parametr log_sigma_sq_t per zadanie.

Wzór (Kendall et al., CVPR 2018):
    L = sum_t [ L_t / exp(log_sigma_sq_t) + log_sigma_sq_t / 2 ]

Uruchomienie:
    python -m EXPERIMENT.train_uncertainty
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
# Uncertainty loss
# ============================================================

class UncertaintyWeightedMaskedLoss(nn.Module):
    """
    Uczalne wagi per zadanie oparte na homoscedastic uncertainty.
    log_sigma_sq_t inicjalizowany na 0 → sigma=1 → równe ważenie na starcie.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_tasks))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: list, targets: torch.Tensor):
        total_loss = torch.tensor(0.0, device=targets.device)

        for t, pred in enumerate(preds):
            mask = ~torch.isnan(targets[:, t])
            if not mask.any():
                continue

            y_t = targets[mask, t].unsqueeze(1)
            p_t = pred[mask]

            raw_loss = self.bce(p_t, y_t).mean()
            precision = torch.exp(-self.log_sigma_sq[t])
            total_loss = total_loss + precision * raw_loss + self.log_sigma_sq[t] / 2.0

        return total_loss

    def get_sigmas(self) -> list:
        return [torch.exp(s / 2.0).item() for s in self.log_sigma_sq]


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


def _plot_sigma_final(tasks: list, sigmas: list, save_path: str):
    colors = ['#e74c3c' if s > 1.1 else '#2ecc71' for s in sigmas]
    _, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tasks, sigmas, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='sigma=1 (start)')
    ax.set_xlabel('Zadanie')
    ax.set_ylabel('sigma (wyuczona)')
    ax.set_title('Uncertainty Weighting — finalna sigma per zadanie\n'
                 'Czerwony = większa niepewność = mniejsza waga')
    ax.legend()
    for bar, val in zip(bars, sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_sigma_evolution(sigma_df: pd.DataFrame, save_path: str):
    import seaborn as sns
    _, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette('tab20', n_colors=len(sigma_df.columns))
    for i, task in enumerate(sigma_df.columns):
        ax.plot(sigma_df.index + 1, sigma_df[task], label=task,
                color=palette[i], linewidth=1.8)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.4, label='sigma=1 (start)')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('sigma_t')
    ax.set_title('Ewolucja sigma per zadanie w trakcie treningu')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# MTL z uncertainty weighting
# ============================================================

def train_mtl(train_loader, val_loader, test_loader, config,
              results_dir: str, timestamp: str):
    model = ADMET_Hybrid_Model(config).to(config.device)
    criterion = UncertaintyWeightedMaskedLoss(n_tasks=len(config.tasks)).to(config.device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    history = {'train_loss': [], 'val_auroc': []}
    sigma_history = []
    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    model_path = os.path.join(results_dir, 'best_mtl_model.pt')

    print(f"\n>>> Start treningu MTL z Uncertainty Weighting ({config.epochs} epok max)...")

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

        current_sigmas = criterion.get_sigmas()
        sigma_history.append(dict(zip(config.tasks, current_sigmas)))

        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_auroc'].append(avg_val_auc)

        sigma_str = ' | '.join(f"{t[:6]}:{s:.3f}" for t, s in zip(config.tasks, current_sigmas))
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(train_loader):.4f} "
              f"| Val AUC: {avg_val_auc:.4f} | sigma: [{sigma_str}]")

        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'val_auc': best_val_auc,
                'sigmas': current_sigmas,
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
    final_sigmas = checkpoint['sigmas']
    print(f"\n>>> Wczytano model z epoki {checkpoint['epoch']+1} "
          f"(Val AUC: {checkpoint['val_auc']:.4f})")

    # Ewaluacja testowa
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
    sigma_df = pd.DataFrame(sigma_history)
    return model, y_true_test, y_pred_parts, history, final_sigmas, sigma_df


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

            print(">>> MTL GNN + Uncertainty Weighting...")
            _, y_true_test, y_pred_test, mtl_history, final_sigmas, sigma_df = train_mtl(
                train_loader, val_loader, test_loader, cfg, results_dir, timestamp
            )
            mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

            plot_all_roc_curves(y_true_test, np.array(y_pred_test).T, cfg.tasks)
            plot_training_results(mtl_history, title='MTL GNN Uncertainty Training')

        plot_count = saver.counter

        # Wagi per zadanie — sigma i weight = 1/sigma^2
        weights = [1.0 / (s ** 2) for s in final_sigmas]
        sigma_weights_df = pd.DataFrame({
            'Task':         cfg.tasks,
            'sigma_final':  final_sigmas,
            'weight_final': weights,
        })
        sigma_weights_df.to_csv(os.path.join(results_dir, 'sigma_weights.csv'), index=False)
        sigma_df.to_csv(os.path.join(results_dir, 'sigma_evolution.csv'), index=False)

        print("\n>>> Nauczone wagi (sigma_final | weight = 1/sigma^2):")
        for _, row in sigma_weights_df.iterrows():
            print(f"  {row['Task']:<35} sigma = {row['sigma_final']:.4f}  "
                  f"weight = {row['weight_final']:.4f}")

        _plot_sigma_final(cfg.tasks, final_sigmas,
                          os.path.join(results_dir, 'sigma_final.png'))
        _plot_sigma_evolution(sigma_df,
                              os.path.join(results_dir, 'sigma_evolution.png'))

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
            'Task':                  cfg.tasks,
            'MTL_GNN_UNCERTAINTY':   [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
            'MTL_GNN':               [main_mtl.get(t,   float('nan')) for t in cfg.tasks],
            'STL_GNN':               [stl_scores.get(t, float('nan')) for t in cfg.tasks],
            'XGBoost':               [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
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

BASE_DIR = os.path.join(os.path.dirname(__file__), 'EXPERIMENT_UNCERTAINTY')


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
