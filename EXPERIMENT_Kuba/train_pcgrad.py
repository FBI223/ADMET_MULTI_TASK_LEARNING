"""
EXPERIMENT_PCGRAD — Gradient Surgery (PCGrad) Multi-Task Learning

Jedyna zmiana względem EXPERIMENT_MAIN: pętla treningowa.
Gradienty per-task są projektowane na wzajemne prostopadłe gdy kolidują (dot < 0).

Źródło: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

Uruchomienie:
    python -m EXPERIMENT.train_pcgrad
"""

import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, plots_dir, timestamp=None):
        self.plots_dir = plots_dir
        self.timestamp = timestamp or datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.counter = 0
        self._original_show = None
        os.makedirs(plots_dir, exist_ok=True)

    def __enter__(self):
        self._original_show = plt.show
        saver = self
        def _save_and_close():
            path = os.path.join(saver.plots_dir, f"plot_{saver.timestamp}_{saver.counter}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saver.counter += 1
        plt.show = _save_and_close
        return self

    def __exit__(self, *_):
        plt.show = self._original_show


# ============================================================
# PCGrad
# ============================================================

def _pcgrad_step(model, preds, targets, optimizer, n_tasks):
    """
    Gradient Surgery na wspólnych parametrach backbone'u.
    Głowice task-specific dostają zwykły gradient.
    """
    bce = nn.BCEWithLogitsLoss()

    # Identyfikacja parametrów: backbone (wspólne) vs głowice (task-specific)
    shared_params = [p for n, p in model.named_parameters()
                     if 'head' not in n and p.requires_grad]
    head_params_per_task = [list(model.heads[t].parameters()) for t in range(n_tasks)]

    # 1. Per-task loss i gradient na shared params
    task_losses = []
    task_shared_grads = []

    for t in range(n_tasks):
        mask = ~torch.isnan(targets[:, t])
        if not mask.any():
            task_losses.append(None)
            task_shared_grads.append(None)
            continue
        loss_t = bce(preds[t][mask], targets[mask, t].unsqueeze(1))
        task_losses.append(loss_t)
        grads = torch.autograd.grad(
            loss_t, shared_params,
            retain_graph=True, allow_unused=True
        )
        task_shared_grads.append([
            g.clone() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, shared_params)
        ])

    # 2. PCGrad: projekcja kolidujących gradientów
    projected = [list(g) if g is not None else None for g in task_shared_grads]

    for i in range(n_tasks):
        if projected[i] is None:
            continue
        for j in range(n_tasks):
            if i == j or task_shared_grads[j] is None:
                continue
            gi = torch.cat([g.flatten() for g in projected[i]])
            gj = torch.cat([g.flatten() for g in task_shared_grads[j]])
            dot = (gi * gj).sum()
            if dot < 0:
                coef = dot / (gj.dot(gj) + 1e-12)
                projected[i] = [
                    projected[i][k] - coef * task_shared_grads[j][k]
                    for k in range(len(projected[i]))
                ]

    # 3. Ustaw gradienty shared params = średnia z projected
    optimizer.zero_grad()
    valid_projected = [g for g in projected if g is not None]
    if valid_projected:
        for k, p in enumerate(shared_params):
            p.grad = torch.stack([vp[k] for vp in valid_projected]).mean(0)

    # 4. Gradienty głowic — zwykłe (bez projekcji)
    for t in range(n_tasks):
        if task_losses[t] is None:
            continue
        mask = ~torch.isnan(targets[:, t])
        if not mask.any():
            continue
        loss_t = bce(preds[t][mask], targets[mask, t].unsqueeze(1))
        head_grads = torch.autograd.grad(
            loss_t, head_params_per_task[t],
            retain_graph=True, allow_unused=True
        )
        for p, g in zip(head_params_per_task[t], head_grads):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g

    optimizer.step()

    valid = [l.item() for l in task_losses if l is not None]
    return sum(valid) / len(valid) if valid else 0.0


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


def _save_plot(save_path, plot_fn):
    original_show = plt.show
    def _save_and_close():
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    plt.show = _save_and_close
    try:
        plot_fn()
    finally:
        plt.show = original_show


def _load_main_baselines(name, tasks):
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


# ============================================================
# MTL z PCGrad
# ============================================================

def train_mtl(train_loader, val_loader, test_loader, config, results_dir, timestamp):
    model = ADMET_Hybrid_Model(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    n_tasks = len(config.tasks)
    history = {'train_loss': [], 'val_auroc': []}
    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    model_path = os.path.join(results_dir, 'best_mtl_model.pt')

    print(f"\n>>> Start treningu MTL z PCGrad ({config.epochs} epok max)...")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(config.device)
            preds = model(batch)
            loss_val = _pcgrad_step(model, preds, batch.y, optimizer, n_tasks)
            epoch_loss += loss_val

        model.eval()
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
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_auc': best_val_auc}, model_path)
            print(f"  >>> Nowy najlepszy model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  !!! Early Stopping po {epoch+1} epokach.")
                break

    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n>>> Wczytano model z epoki {checkpoint['epoch']+1} "
          f"(Val AUC: {checkpoint['val_auc']:.4f})")

    model.eval()
    y_true_parts, y_pred_parts = [], [[] for _ in config.tasks]
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.device)
            out = model(batch)
            y_true_parts.append(batch.y.cpu().numpy())
            for i in range(n_tasks):
                y_pred_parts[i].extend(torch.sigmoid(out[i]).cpu().numpy().flatten())

    return model, np.vstack(y_true_parts), y_pred_parts, history


# ============================================================
# Jeden eksperyment
# ============================================================

def run_single_experiment(cfg, results_dir, name):
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'results'), exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_file = open(os.path.join(results_dir, 'run_stats.txt'), 'w', encoding='utf-8')
    orig_stdout = sys.stdout

    class Tee:
        def __init__(self, *s): self.s = s
        def write(self, m):
            for x in self.s: x.write(m); x.flush()
        def flush(self):
            for x in self.s: x.flush()

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

            print(">>> MTL GNN + PCGrad...")
            _, y_true_test, y_pred_test, mtl_history = train_mtl(
                train_loader, val_loader, test_loader, cfg, results_dir, timestamp
            )
            mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

            plot_all_roc_curves(y_true_test, np.array(y_pred_test).T, cfg.tasks)
            plot_training_results(mtl_history, title='MTL GNN PCGrad Training')

        plot_count = saver.counter

        baselines = _load_main_baselines(name, cfg.tasks)
        main_mtl   = baselines.get('MTL_GNN', {})
        stl_scores = baselines.get('STL_GNN', {})
        xgb_scores = baselines.get('XGBoost', {})
        if baselines:
            print(f"\n>>> Wczytano baseline z EXPERIMENT_MAIN/{name}/final_results.csv")
        else:
            print(f"\n>>> Brak baseline EXPERIMENT_MAIN/{name}.")

        results_df = pd.DataFrame({
            'Task':           cfg.tasks,
            'MTL_GNN_PCGRAD': [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
            'MTL_GNN':        [main_mtl.get(t,   float('nan')) for t in cfg.tasks],
            'STL_GNN':        [stl_scores.get(t, float('nan')) for t in cfg.tasks],
            'XGBoost':        [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
        })
        csv_path = os.path.join(results_dir, 'final_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n>>> Wyniki:\n{results_df.to_string(index=False)}")

        _save_plot(
            os.path.join(results_dir, f"plot_{timestamp}_{plot_count}.png"),
            lambda: plot_model_comparison_simple(results_df)
        )
        print(f"\nGOTOWE. Wyniki: {csv_path}")

    finally:
        sys.stdout = orig_stdout
        log_file.close()


# ============================================================
# MAIN
# ============================================================

COMBINATIONS = [
    (True, False, False, 'GNN'),
    (True, False, True,  'GNN_MORGAN'),
    (True, True,  False, 'GNN_RDKIT'),
    (True, True,  True,  'GNN_RDKIT_MORGAN'),
]

BASE_DIR = os.path.join(os.path.dirname(__file__), 'EXPERIMENT_PCGRAD')


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
        run_single_experiment(cfg, os.path.join(BASE_DIR, name), name)


if __name__ == '__main__':
    main()
