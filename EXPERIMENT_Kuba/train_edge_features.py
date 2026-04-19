"""
EXPERIMENT_EDGE_FEATURES — GINEConv z cechami wiązań chemicznych

Zmiana względem EXPERIMENT_MAIN: GINConv → GINEConv + 6-wymiarowy wektor cech krawędzi.
Cechy: bond_type (4×one-hot) + is_in_ring + is_conjugated.

Źródło: Hu et al., "Strategies for Pre-training Graph Neural Networks", ICLR 2020.

Uruchomienie:
    python -m EXPERIMENT.train_edge_features
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
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GNNLoader
from tdc.single_pred import ADME, Tox

from EXPERIMENT.config import Config
from EXPERIMENT.model import MaskedBCELoss
from EXPERIMENT.data import smiles_to_morgan, smiles_to_rdkit_descriptors, get_full_data
from EXPERIMENT.plots import (
    evaluate_per_task,
    evaluate_gnn_simple,
    plot_training_results,
    plot_all_roc_curves,
    plot_data_sparsity,
    plot_label_correlations,
    plot_model_comparison_simple,
)


EDGE_DIM = 6


# ============================================================
# Dane z cechami krawędzi
# ============================================================

def smiles_to_graph_edge(smiles, y_labels):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
            atom.GetImplicitValence(),
            int(atom.GetHybridization()),
            atom.GetNumRadicalElectrons(),
            atom.GetMass() * 0.01,
            float(atom.IsInRing()),
        ])
    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        feat = [
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.IsInRing()),
            float(bond.GetIsConjugated()),
        ]
        edge_indices += [[start, end], [end, start]]
        edge_attrs += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float)

    y = torch.tensor(y_labels, dtype=torch.float).view(1, -1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def smiles_to_hybrid_data_edge(smiles, y_labels, config):
    data = smiles_to_graph_edge(smiles, y_labels)
    if data is None:
        return None
    if config.use_morgan:
        fp = smiles_to_morgan(smiles, n_bits=config.morgan_dim)
        data.morgan = torch.tensor(fp, dtype=torch.float).view(1, -1)
    if config.use_rdkit:
        desc = smiles_to_rdkit_descriptors(smiles)
        data.rdkit = torch.tensor(desc, dtype=torch.float).view(1, -1)
    return data


def get_full_data_edge(config):
    task_hash = hashlib.md5("".join(sorted(config.tasks)).encode()).hexdigest()
    cache_base = "data_cache"
    os.makedirs(cache_base, exist_ok=True)
    edge_cache_path = os.path.join(cache_base, f"edge_cache_{task_hash}.pt")
    df_cache_path   = os.path.join(cache_base, f"edge_df_{task_hash}.pkl")

    if os.path.exists(edge_cache_path) and os.path.exists(df_cache_path):
        print(f"\n>>> Wczytywanie Edge Cache: {task_hash[:8]}...")
        all_data = torch.load(edge_cache_path, weights_only=False)
        raw_train_df = pd.read_pickle(df_cache_path)
    else:
        print(f"\n>>> Tworzenie Edge Cache (cechy krawędzi)...")

        split_dfs = {'train': None, 'valid': None, 'test': None}
        for task in config.tasks:
            try:   loader = ADME(name=task)
            except: loader = Tox(name=task)
            splits = loader.get_split()
            for s in ['train', 'valid', 'test']:
                df = splits[s][['Drug', 'Y']].rename(columns={'Drug': 'SMILES', 'Y': task})
                if split_dfs[s] is None: split_dfs[s] = df
                else: split_dfs[s] = pd.merge(split_dfs[s], df, on='SMILES', how='outer')

        full_cfg = Config()
        full_cfg.use_graph = full_cfg.use_morgan = full_cfg.use_rdkit = True

        all_data = {'train': [], 'valid': [], 'test': []}
        for s in ['train', 'valid', 'test']:
            print(f"Konwertowanie {s} z cechami krawędzi...")
            df = split_dfs[s]
            for _, row in tqdm(df.iterrows(), total=len(df)):
                labels = row[config.tasks].values.astype(float)
                obj = smiles_to_hybrid_data_edge(row['SMILES'], labels, full_cfg)
                if obj:
                    all_data[s].append(obj)

        raw_train_df = split_dfs['train']
        print(">>> Zapisywanie Edge Cache...")
        torch.save(all_data, edge_cache_path)
        raw_train_df.to_pickle(df_cache_path)

    for s in ['train', 'valid', 'test']:
        for obj in all_data[s]:
            if not config.use_rdkit  and hasattr(obj, 'rdkit'):  delattr(obj, 'rdkit')
            if not config.use_morgan and hasattr(obj, 'morgan'): delattr(obj, 'morgan')

    train_loader = GNNLoader(all_data['train'], batch_size=config.batch_size, shuffle=True)
    val_loader   = GNNLoader(all_data['valid'], batch_size=config.batch_size)
    test_loader  = GNNLoader(all_data['test'],  batch_size=config.batch_size)
    return train_loader, val_loader, test_loader, all_data['train'], all_data['test'], raw_train_df


# ============================================================
# Model z GINEConv
# ============================================================

class ADMET_EdgeFeature_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        combined_dim = 0

        if config.use_graph:
            self.gin_backbone = nn.ModuleList([
                GINEConv(self._make_mlp(config.node_dim, config.hidden_dim),
                         train_eps=True, edge_dim=EDGE_DIM),
                GINEConv(self._make_mlp(config.hidden_dim, config.hidden_dim),
                         train_eps=True, edge_dim=EDGE_DIM),
                GINEConv(self._make_mlp(config.hidden_dim, config.hidden_dim),
                         train_eps=True, edge_dim=EDGE_DIM),
            ])
            self.graph_bn = nn.BatchNorm1d(config.hidden_dim)
            combined_dim += config.hidden_dim

        vector_input_dim = 0
        if config.use_morgan: vector_input_dim += config.morgan_dim
        if config.use_rdkit:  vector_input_dim += config.rdkit_dim

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

        self.fusion_bn = nn.BatchNorm1d(combined_dim)
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

    def forward(self, data):
        features = []

        if self.config.use_graph:
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            for conv in self.gin_backbone:
                x = F.relu(conv(x, edge_index, edge_attr=edge_attr))
            g_emb = self.graph_bn(global_add_pool(x, batch))
            features.append(g_emb)

        vec_parts = []
        if self.config.use_morgan: vec_parts.append(data.morgan)
        if self.config.use_rdkit:  vec_parts.append(data.rdkit)

        if vec_parts:
            v_emb = self.vector_mlp(torch.cat(vec_parts, dim=1))
            features.append(v_emb)

        combined = self.fusion_bn(torch.cat(features, dim=1))
        return [head(combined) for head in self.heads]


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
# MTL z GINEConv
# ============================================================

def train_mtl(train_loader, val_loader, test_loader, config, results_dir, timestamp):
    model = ADMET_EdgeFeature_Model(config).to(config.device)
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

    print(f"\n>>> Start treningu MTL z GINEConv (edge features, {config.epochs} epok max)...")

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

        train_loader, val_loader, test_loader, _, _, raw_train_df = get_full_data_edge(cfg)
        _sync_dims(train_loader, cfg)
        print(f"Wymiary: node={cfg.node_dim}, morgan={cfg.morgan_dim}, rdkit={cfg.rdkit_dim}\n")

        with PlotSaver(os.path.join(results_dir, 'plots'), timestamp) as saver:
            plot_data_sparsity(raw_train_df, cfg.tasks)
            plot_label_correlations(raw_train_df, cfg.tasks)

            print(">>> MTL GNN + Edge Features (GINEConv)...")
            _, y_true_test, y_pred_test, mtl_history = train_mtl(
                train_loader, val_loader, test_loader, cfg, results_dir, timestamp
            )
            mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

            plot_all_roc_curves(y_true_test, np.array(y_pred_test).T, cfg.tasks)
            plot_training_results(mtl_history, title='MTL GNN Edge Features Training')

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
            'Task':          cfg.tasks,
            'MTL_GNN_EDGE':  [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
            'MTL_GNN':       [main_mtl.get(t,   float('nan')) for t in cfg.tasks],
            'STL_GNN':       [stl_scores.get(t, float('nan')) for t in cfg.tasks],
            'XGBoost':       [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
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

BASE_DIR = os.path.join(os.path.dirname(__file__), 'EXPERIMENT_EDGE_FEATURES')


def main():
    print(">>> Przygotowanie Edge Cache (pierwsze uruchomienie może chwilę trwać)...")
    cache_cfg = Config()
    cache_cfg.tasks = cache_cfg.tasks_old
    cache_cfg.use_graph = cache_cfg.use_rdkit = cache_cfg.use_morgan = True
    get_full_data_edge(cache_cfg)
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
