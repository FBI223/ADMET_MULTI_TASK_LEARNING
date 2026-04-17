"""
EXPERIMENT_GROUPING — Task-Grouped Multi-Task Learning

Zmiana względem EXPERIMENT_MAIN: dwa osobne backbone'y GIN.
gin_cyp  → 5 tasków CYP (metabolizm)
gin_other → 8 pozostałych tasków (transport, toksyczność)

Motywacja: gradienty klastru CYP i herg/dili/skin_reaction kolidują na wspólnym backbone'ie.
Osobne backbone'y eliminują konflikt bez rezygnacji z MTL na głowicach.

Uruchomienie:
    python -m EXPERIMENT.train_grouping
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
from torch_geometric.nn import GINConv, global_add_pool
from EXPERIMENT.config import Config
from EXPERIMENT.model import MaskedBCELoss
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


CYP_TASK_NAMES = frozenset({
    'cyp2c19_veith', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp1a2_veith', 'cyp2c9_veith'
})


# ============================================================
# Grouped Model
# ============================================================

class ADMET_Grouped_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._cyp_idx = [i for i, t in enumerate(config.tasks) if t in CYP_TASK_NAMES]
        self._other_idx = [i for i, t in enumerate(config.tasks) if t not in CYP_TASK_NAMES]
        self._cyp_set = set(self._cyp_idx)

        combined_dim = 0

        if config.use_graph:
            self.gin_cyp = nn.ModuleList([
                GINConv(self._make_mlp(config.node_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True),
            ])
            self.graph_bn_cyp = nn.BatchNorm1d(config.hidden_dim)

            self.gin_other = nn.ModuleList([
                GINConv(self._make_mlp(config.node_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True),
            ])
            self.graph_bn_other = nn.BatchNorm1d(config.hidden_dim)
            combined_dim += config.hidden_dim

        vector_input_dim = 0
        if config.use_morgan: vector_input_dim += config.morgan_dim
        if config.use_rdkit: vector_input_dim += config.rdkit_dim

        if vector_input_dim > 0:
            self.vector_mlp = nn.Sequential(
                nn.Linear(vector_input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            combined_dim += config.hidden_dim

        self.fusion_bn_cyp = nn.BatchNorm1d(combined_dim)
        self.fusion_bn_other = nn.BatchNorm1d(combined_dim)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            ) for _ in range(len(config.tasks))
        ])

    @staticmethod
    def _make_mlp(in_d, out_d):
        return nn.Sequential(
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
            nn.ReLU(),
            nn.Linear(out_d, out_d),
            nn.ReLU(),
        )

    def _gin_forward(self, gin_layers, graph_bn, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in gin_layers:
            x = F.relu(conv(x, edge_index))
        return graph_bn(global_add_pool(x, batch))

    def forward(self, data):
        features_cyp, features_other = [], []

        if self.config.use_graph:
            emb_cyp = self._gin_forward(self.gin_cyp, self.graph_bn_cyp, data)
            emb_other = self._gin_forward(self.gin_other, self.graph_bn_other, data)
            features_cyp.append(emb_cyp)
            features_other.append(emb_other)

        vec_parts = []
        if self.config.use_morgan: vec_parts.append(data.morgan)
        if self.config.use_rdkit: vec_parts.append(data.rdkit)

        if vec_parts:
            v_emb = self.vector_mlp(torch.cat(vec_parts, dim=1))
            features_cyp.append(v_emb)
            features_other.append(v_emb)

        combined_cyp = self.fusion_bn_cyp(torch.cat(features_cyp, dim=1))
        combined_other = self.fusion_bn_other(torch.cat(features_other, dim=1))

        preds = []
        for i in range(len(self.config.tasks)):
            if i in self._cyp_set:
                preds.append(self.heads[i](combined_cyp))
            else:
                preds.append(self.heads[i](combined_other))
        return preds


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
# MTL z Grouped backbone
# ============================================================

def train_mtl(train_loader, val_loader, test_loader, config, results_dir, timestamp):
    model = ADMET_Grouped_Model(config).to(config.device)
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

    cyp_names = [t for t in config.tasks if t in CYP_TASK_NAMES]
    other_names = [t for t in config.tasks if t not in CYP_TASK_NAMES]
    print(f"\n>>> Start treningu MTL z Grouped backbone ({config.epochs} epok max)...")
    print(f"    CYP backbone ({len(cyp_names)}):   {cyp_names}")
    print(f"    Other backbone ({len(other_names)}): {other_names}")

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

        train_loader, val_loader, test_loader, _, _, raw_train_df = get_full_data(cfg)
        _sync_dims(train_loader, cfg)
        print(f"Wymiary: node={cfg.node_dim}, morgan={cfg.morgan_dim}, rdkit={cfg.rdkit_dim}\n")

        with PlotSaver(os.path.join(results_dir, 'plots'), timestamp) as saver:
            plot_data_sparsity(raw_train_df, cfg.tasks)
            plot_label_correlations(raw_train_df, cfg.tasks)

            print(">>> MTL GNN + Grouped backbone...")
            _, y_true_test, y_pred_test, mtl_history = train_mtl(
                train_loader, val_loader, test_loader, cfg, results_dir, timestamp
            )
            mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

            plot_all_roc_curves(y_true_test, np.array(y_pred_test).T, cfg.tasks)
            plot_training_results(mtl_history, title='MTL GNN Grouped Training')

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
            'Task':             cfg.tasks,
            'MTL_GNN_GROUPED':  [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
            'MTL_GNN':          [main_mtl.get(t,   float('nan')) for t in cfg.tasks],
            'STL_GNN':          [stl_scores.get(t, float('nan')) for t in cfg.tasks],
            'XGBoost':          [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
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

BASE_DIR = os.path.join(os.path.dirname(__file__), 'EXPERIMENT_GROUPING')


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
