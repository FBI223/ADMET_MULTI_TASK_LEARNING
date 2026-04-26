import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F


class ADMET_Hybrid_Model(nn.Module):
    def __init__(self, config, single_task_idx=None):  # >>> ADDED
        super().__init__()
        self.config = config
        self.single_task_idx = single_task_idx  # >>> ADDED

        combined_dim = 0

        # --- NOGA GRAFOWA (GIN) ---
        if config.use_graph:
            self.gin_backbone = nn.ModuleList([
                GINConv(self._make_mlp(config.node_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True)
            ])
            self.graph_bn = nn.BatchNorm1d(config.hidden_dim)
            combined_dim += config.hidden_dim

        # --- NOGA WEKTOROWA ---
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
                nn.ReLU()
            )
            combined_dim += config.hidden_dim

        # --- FUZJA ---
        self.fusion_bn = nn.BatchNorm1d(combined_dim)

        # --- HEADY ---
        if self.single_task_idx is None:
            # MTL
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(combined_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim // 2, 1)
                ) for _ in range(len(config.tasks))
            ])
        else:
            # STL
            self.head = nn.Sequential(  # >>> ADDED
                nn.Linear(combined_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1)
            )

    def _make_mlp(self, in_d, out_d):
        return nn.Sequential(
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
            nn.ReLU(),
            nn.Linear(out_d, out_d),
            nn.ReLU()
        )

    def forward(self, data):
        features = []

        # --- GRAPH ---
        if self.config.use_graph:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.gin_backbone:
                x = conv(x, edge_index)
                x = F.relu(x)

            g_emb = global_add_pool(x, batch)
            g_emb = self.graph_bn(g_emb)
            features.append(g_emb)

        # --- VECTOR ---
        vec_parts = []
        if self.config.use_morgan: vec_parts.append(data.morgan)
        if self.config.use_rdkit: vec_parts.append(data.rdkit)

        if vec_parts:
            v_input = torch.cat(vec_parts, dim=1)
            v_emb = self.vector_mlp(v_input)
            features.append(v_emb)

        # --- FUSION ---
        combined_features = torch.cat(features, dim=1)
        combined_features = self.fusion_bn(combined_features)

        # --- OUTPUT ---
        if self.single_task_idx is None:
            return [head(combined_features) for head in self.heads]
        else:
            return self.head(combined_features)  # >>> ADDED

class ADMET_Hybrid_ModelSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        combined_dim = 0

        # --- NOGA GRAFOWA (GIN) ---
        if config.use_graph:
            # W GIN kluczowe jest, aby MLP wewnątrz GINConv miał BatchNorm
            self.gin_backbone = nn.ModuleList([
                GINConv(self._make_mlp(config.node_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True),
                GINConv(self._make_mlp(config.hidden_dim, config.hidden_dim), train_eps=True)
            ])
            # BN po sumowaniu globalnym (Readout)
            self.graph_bn = nn.BatchNorm1d(config.hidden_dim)
            combined_dim += config.hidden_dim

        # --- NOGA WEKTOROWA (Morgan + RDKit) ---
        vector_input_dim = 0
        if config.use_morgan: vector_input_dim += config.morgan_dim
        if config.use_rdkit: vector_input_dim += config.rdkit_dim

        if vector_input_dim > 0:
            self.vector_mlp = nn.Sequential(
                nn.Linear(vector_input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),  # Stabilizacja dużych wartości RDKit
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU()
            )
            combined_dim += config.hidden_dim

        # --- FUZJA I GŁOWICE ---
        # Dodajemy BN dla połączonych cech przed wejściem do specyficznych zadań
        self.fusion_bn = nn.BatchNorm1d(combined_dim)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1)
            ) for _ in range(len(config.tasks))
        ])

    def _make_mlp(self, in_d, out_d):
        # MLP wewnątrz GINConv MUSI mieć BatchNorm dla stabilności sumowania
        return nn.Sequential(
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
            nn.ReLU(),
            nn.Linear(out_d, out_d),
            nn.ReLU()
        )

    def forward(self, data):
        features = []

        # 1. Procesowanie grafu (Zgodnie z GIN - Sumowanie)
        if self.config.use_graph:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.gin_backbone:
                x = conv(x, edge_index)  # GINConv już ma wbudowane sumowanie sąsiadów
                x = F.relu(x)

            # Readout: Sumowanie wszystkich atomów w cząsteczce
            g_emb = global_add_pool(x, batch)
            g_emb = self.graph_bn(g_emb)  # Krytyczna normalizacja sumy
            features.append(g_emb)

        # 2. Procesowanie wektorów
        vec_parts = []
        if self.config.use_morgan: vec_parts.append(data.morgan)
        if self.config.use_rdkit: vec_parts.append(data.rdkit)

        if vec_parts:
            v_input = torch.cat(vec_parts, dim=1)
            v_emb = self.vector_mlp(v_input)
            features.append(v_emb)

        # 3. Fuzja (Concatenation)
        combined_features = torch.cat(features, dim=1)
        combined_features = self.fusion_bn(combined_features)

        # 4. Finalna predykcja
        return [head(combined_features) for head in self.heads]


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        p = torch.cat(preds, dim=1)
        mask = ~torch.isnan(targets)

        # Kluczowe: zastąp NaN zerami, aby BCE nie wygenerowało NaN
        clean_targets = torch.where(mask, targets, torch.zeros_like(targets))

        loss = self.crit(p, clean_targets)

        # Teraz maskowanie zadziała poprawnie (0.0 * real_loss)
        return (loss * mask.float()).sum() / (mask.sum() + 1e-8) if mask.any() else torch.tensor(0.0)


class LearnedWeightedMaskedBCELoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

        # Tworzymy uczone parametry s_t (inicjalizujemy zerami, co daje wagę exp(0)=1)
        # Muszą być owinięte w nn.Parameter, żeby optimizer je widział
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds_list, targets):
        preds = torch.cat(preds_list, dim=1)
        mask = ~torch.isnan(targets)

        clean_targets = torch.where(mask, targets, torch.zeros_like(targets))
        loss_matrix = self.crit(preds, clean_targets)

        # Obliczamy L_t dla każdego zadania
        task_losses = (loss_matrix * mask.float()).sum(dim=0) / (mask.sum(dim=0) + 1e-8)

        # APLIKACJA UCZONYCH WAG
        # exp(-s_t) * L_t + s_t
        weighted_losses = torch.exp(-self.log_vars) * task_losses + self.log_vars

        return weighted_losses.sum()

