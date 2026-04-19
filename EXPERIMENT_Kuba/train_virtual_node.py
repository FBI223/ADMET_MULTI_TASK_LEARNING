"""
EXPERIMENT_VIRTUAL_NODE — Wirtualny węzeł dla globalnego przepływu informacji

Zmiana względem EXPERIMENT_MAIN: każdy graf molekularny wzbogacony o węzeł wirtualny
połączony ze wszystkimi atomami (obie strony). Umożliwia globalną komunikację w 1 przeskoku.

Model, konfiguracja, funkcja kosztu — bez zmian.

Uruchomienie:
    python -m EXPERIMENT.train_virtual_node
"""

import os
import sys
import datetime
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader as GNNLoader

from EXPERIMENT.config import Config
from EXPERIMENT.model import ADMET_Hybrid_Model, MaskedBCELoss
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
# Virtual Node transform
# ============================================================

def add_virtual_node(data):
    n = data.x.size(0)
    vn_feat = torch.zeros(1, data.x.size(1), dtype=data.x.dtype)
    data.x = torch.cat([data.x, vn_feat], dim=0)

    vn_idx = n
    src = torch.arange(n, dtype=torch.long)
    dst = torch.full((n,), vn_idx, dtype=torch.long)

    new_edges = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ], dim=0)
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return data


def get_full_data_vn(config):
    task_hash = hashlib.md5("".join(sorted(config.tasks)).encode()).hexdigest()
    cache_base = "data_cache"
    os.makedirs(cache_base, exist_ok=True)
    vn_cache_path  = os.path.join(cache_base, f"vn_cache_{task_hash}.pt")
    df_cache_path  = os.path.join(cache_base, f"master_df_{task_hash}.pkl")
    master_cache_path = os.path.join(cache_base, f"master_cache_{task_hash}.pt")

    if os.path.exists(vn_cache_path) and os.path.exists(df_cache_path):
        print(f"\n>>> Wczytywanie VN Cache: {task_hash[:8]}...")
        all_data = torch.load(vn_cache_path, weights_only=False)
        raw_train_df = pd.read_pickle(df_cache_path)
    else:
        if not os.path.exists(master_cache_path):
            get_full_data(config)

        print(f"\n>>> Dodawanie Virtual Node do Master Cache: {task_hash[:8]}...")
        all_data = torch.load(master_cache_path, weights_only=False)
        raw_train_df = pd.read_pickle(df_cache_path)

        for s in ['train', 'valid', 'test']:
            all_data[s] = [add_virtual_node(obj) for obj in all_data[s]]

        print(">>> Zapisywanie VN Cache...")
        torch.save(all_data, vn_cache_path)

    for s in ['train', 'valid', 'test']:
        for obj in all_data[s]:
            if not config.use_rdkit  and hasattr(obj, 'rdkit'):  delattr(obj, 'rdkit')
            if not config.use_morgan and hasattr(obj, 'morgan'): delattr(obj, 'morgan')

    train_loader = GNNLoader(all_data['train'], batch_size=config.batch_size, shuffle=True)
    val_loader   = GNNLoader(all_data['valid'], batch_size=config.batch_size)
    test_loader  = GNNLoader(all_data['test'],  batch_size=config.batch_size)
    return train_loader, val_loader, test_loader, all_data['train'], all_data['test'], raw_train_df


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
# MTL z Virtual Node
# ============================================================

def train_mtl(train_loader, val_loader, test_loader, config, results_dir, timestamp):
    model = ADMET_Hybrid_Model(config).to(config.device)
    criterion = MaskedBCELoss().to(config.device)
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

    print(f"\n>>> Start treningu MTL z Virtual Node ({config.epochs} epok max)...")

    for epoch in range(config.epochs):
        model.train()
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

        train_loader, val_loader, test_loader, _, _, raw_train_df = get_full_data_vn(cfg)
        _sync_dims(train_loader, cfg)
        print(f"Wymiary: node={cfg.node_dim}, morgan={cfg.morgan_dim}, rdkit={cfg.rdkit_dim}\n")

        with PlotSaver(os.path.join(results_dir, 'plots'), timestamp) as saver:
            plot_data_sparsity(raw_train_df, cfg.tasks)
            plot_label_correlations(raw_train_df, cfg.tasks)

            print(">>> MTL GNN + Virtual Node...")
            _, y_true_test, y_pred_test, mtl_history = train_mtl(
                train_loader, val_loader, test_loader, cfg, results_dir, timestamp
            )
            mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

            plot_all_roc_curves(y_true_test, np.array(y_pred_test).T, cfg.tasks)
            plot_training_results(mtl_history, title='MTL GNN Virtual Node Training')

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
            'Task':       cfg.tasks,
            'MTL_GNN_VN': [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
            'MTL_GNN':    [main_mtl.get(t,   float('nan')) for t in cfg.tasks],
            'STL_GNN':    [stl_scores.get(t, float('nan')) for t in cfg.tasks],
            'XGBoost':    [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
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

BASE_DIR = os.path.join(os.path.dirname(__file__), 'EXPERIMENT_VIRTUAL_NODE')


def main():
    print(">>> Przygotowanie VN Cache...")
    cache_cfg = Config()
    cache_cfg.tasks = cache_cfg.tasks_old
    cache_cfg.use_graph = cache_cfg.use_rdkit = cache_cfg.use_morgan = True
    get_full_data_vn(cache_cfg)
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
