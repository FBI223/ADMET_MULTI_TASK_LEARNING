import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.args import TrainArgs
from chemprop.models import MoleculeModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from chemprop.features import mol2graph
from chemprop.features import BatchMolGraph

'''
def build_encoder():
    args = TrainArgs()

    args.dataset_type = 'classification'
    args.hidden_size = 300
    args.depth = 3
    args.ffn_hidden_size = 300
    args.ffn_num_layers = 1
    args.activation = 'ReLU'

    model = MoleculeModel(args)

    return model

'''


'''
SMILES
→ mol2graph(list)
→ BatchMolGraph
→ encoder([batch])
→ embedding
→ concat features
→ multitask heads

'''


def smiles_to_graph(smiles_list):
    return mol2graph(smiles_list)

def load_dataset(path):

    df = pd.read_parquet(path)
    print(df.columns.tolist())

    tasks = sorted(df["task"].unique())
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    grouped = defaultdict(list)

    for _, row in df.iterrows():
        grouped[row["smiles"]].append(row)

    data = []

    for smiles, rows in grouped.items():

        T = len(tasks)

        y = np.zeros(T)
        mask = np.zeros(T)

        for r in rows:
            t = task_to_idx[r["task"]]
            y[t] = r["label"]
            mask[t] = 1

        # RDKit (200)


        rdkit_cols = [c for c in df.columns if c not in exclude]
        rdkit = rows[0][rdkit_cols].values.astype(np.float64)
        rdkit = np.nan_to_num(rdkit, nan=0.0, posinf=0.0, neginf=0.0)
        # QC (4)
        qc = np.array([
            rows[0]["dipole"],
            rows[0]["homo_lumo"],
            rows[0]["electrons"],
            rows[0]["energy"]
        ], dtype=np.float32)

        # QC mask (4)
        qc_mask = np.array([
            rows[0]["mask_dipole"],
            rows[0]["mask_homo_lumo"],
            rows[0]["mask_electrons"],
            rows[0]["mask_energy"]
        ], dtype=np.float32)

        split = rows[0]["split"]

        data.append({
            "smiles": smiles,
            "graph": smiles,
            "y": y,
            "mask": mask,
            "rdkit": rdkit,
            "qc": qc,
            "qc_mask": qc_mask,
            "split": split
        })

    return data, tasks




def normalize_features(data):

    rdkit_all = np.stack([d["rdkit"] for d in data])
    qc_all = np.stack([d["qc"] for d in data])

    print("INF rdkit:", np.isinf(rdkit_all).sum())
    print("NAN rdkit:", np.isnan(rdkit_all).sum())

    # 🔴 FIX INF/NAN
    rdkit_all = np.nan_to_num(rdkit_all, nan=0.0, posinf=0.0, neginf=0.0)
    qc_all = np.nan_to_num(qc_all, nan=0.0, posinf=0.0, neginf=0.0)

    # 🔴 CLIP
    rdkit_all = np.clip(rdkit_all, -1e6, 1e6)
    qc_all = np.clip(qc_all, -1e6, 1e6)

    rdkit_scaler = StandardScaler()
    qc_scaler = StandardScaler()

    rdkit_all = rdkit_scaler.fit_transform(rdkit_all)
    qc_all = qc_scaler.fit_transform(qc_all)

    for i, d in enumerate(data):
        d["rdkit"] = rdkit_all[i]
        d["qc"] = qc_all[i]

    return data, rdkit_scaler, qc_scaler





class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        return {
            "graph": d["graph"] ,   # mol2graph(smiles) # później Chemprop
            "rdkit": torch.tensor(d["rdkit"], dtype=torch.float32),
            "qc": torch.tensor(d["qc"], dtype=torch.float32),
            "qc_mask": torch.tensor(d["qc_mask"], dtype=torch.float32),
            "y": torch.tensor(d["y"], dtype=torch.float32),
            "mask": torch.tensor(d["mask"], dtype=torch.float32)
        }




def collate_fn(batch):

    smiles = [b["graph"] for b in batch]

    graph = mol2graph(smiles)  # 🔴 KLUCZ

    rdkit = torch.stack([b["rdkit"] for b in batch])
    qc = torch.stack([b["qc"] for b in batch])
    qc_mask = torch.stack([b["qc_mask"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])

    return {
        "graph": graph,
        "rdkit": rdkit,
        "qc": qc,
        "qc_mask": qc_mask,
        "y": y,
        "mask": mask
    }

def build_dataloader(data, batch_size=32, shuffle=True):

    dataset = SimpleDataset(data)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn  # 🔴 DODAJ
    )


def dummy_loader():

    B = 16
    T = 13

    smiles = ["CCO"] * B  # przykładowe molekuły
    graph = smiles_to_graph(smiles)

    for _ in range(10):

        yield {
            "graph": graph,  # 🔴 TERAZ GNN DZIAŁA
            "rdkit": torch.randn(B, 200),
            "qc": torch.randn(B, 4),
            "qc_mask": torch.randint(0, 2, (B, 4)).float(),
            "y": torch.randint(0, 2, (B, T)).float(),
            "mask": torch.ones(B, T)  # 🔴 ważne
        }


def split_data(data):

    train = [d for d in data if d["split"] == "train"]
    val = [d for d in data if d["split"] == "val"]
    test = [d for d in data if d["split"] == "test"]

    return train, val, test


def build_encoder():
    args = TrainArgs()

    args.dataset_type = 'classification'
    args.hidden_size = 300
    args.depth = 3
    args.ffn_hidden_size = 300

    model = MoleculeModel(args)

    return model.encoder  # 🔴 tylko encoder


class TaskHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class TaskWeighting(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.log_beta = nn.Parameter(torch.zeros(T))

    def forward(self, losses, n_valid):
        r = n_valid / (n_valid.sum() + 1e-8)
        beta = F.softplus(self.log_beta)
        w = r ** beta
        return (w * losses).sum(), w



class QW_MTL(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()

        self.encoder = build_encoder()

        self.input_dim = 300 + 200 + 4 + 4
        self.num_tasks = num_tasks

        self.norm = nn.LayerNorm(self.input_dim)

        self.heads = nn.ModuleList([
            TaskHead(self.input_dim) for _ in range(num_tasks)
        ])

        self.weighting = TaskWeighting(num_tasks)

    def forward(self, batch, rdkit, qc, qc_mask):

        device = rdkit.device

        rdkit = rdkit.to(device)  # 🔴 DODAJ
        qc = qc.to(device)
        qc_mask = qc_mask.to(device)

        if batch is None:
            z = torch.randn(rdkit.shape[0], 300, device=device)
        else:
            z = self.encoder([batch])

        z = z.to(device)

        x = torch.cat([z, rdkit, qc, qc_mask], dim=1)
        x = self.norm(x)

        out = torch.cat([head(x) for head in self.heads], dim=1)

        return out


def compute_loss(model, preds, targets, mask):

    T = preds.shape[1]

    losses = []
    n_valid = []

    for t in range(T):
        m = mask[:, t] == 1

        if m.sum() == 0:
            losses.append(torch.tensor(0.0, device=preds.device))
            n_valid.append(torch.tensor(0.0, device=preds.device))
            continue

        loss_t = F.binary_cross_entropy_with_logits(
            preds[m, t],
            targets[m, t].clamp(0, 1)
        )

        losses.append(loss_t)
        n_valid.append(m.sum())

    losses = torch.stack(losses)
    n_valid = torch.stack(n_valid).float()

    return model.weighting(losses, n_valid)



def train_step(model, batch, rdkit, qc, qc_mask, targets, mask, opt):

    model.train()

    preds = model(batch, rdkit, qc, qc_mask)

    loss, weights = compute_loss(model, preds, targets, mask)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), weights



def testing():
    B = 16
    T = 13

    model = QW_MTL(T)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # fake data
    rdkit = torch.randn(B, 200)
    qc = torch.randn(B, 4)
    qc_mask = torch.randint(0, 2, (B, 4)).float()
    targets = torch.randint(0, 2, (B, T)).float()
    mask = torch.randint(0, 2, (B, T)).float()

    batch = None  # tutaj idzie chemprop batch

    # forward bez batcha (debug)
    z = torch.randn(B, 300)
    x = torch.cat([z, rdkit, qc, qc_mask], dim=1)


exclude = [
    "smiles", "label", "task", "split",
    "dipole", "homo_lumo", "electrons", "energy",
    "mask_dipole", "mask_homo_lumo", "mask_electrons", "mask_energy",
    "success"
]



LR = 1e-3
EPOCHS = 50
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0
NUM_TASKS = 13


def train(model, dataloader, optimizer, scheduler, device):

    model.to(device)

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for batch in dataloader:

            batch_graph = batch["graph"]
            rdkit = batch["rdkit"].to(device)
            qc = batch["qc"].to(device)
            qc_mask = batch["qc_mask"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)

            preds = model(batch_graph, rdkit, qc, qc_mask)

            loss, _ = compute_loss(model, preds, targets, mask)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")



@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()
    total_loss = 0.0

    for batch in dataloader:

        batch_graph = batch["graph"]
        rdkit = batch["rdkit"].to(device)
        qc = batch["qc"].to(device)
        qc_mask = batch["qc_mask"].to(device)
        targets = batch["y"].to(device)
        mask = batch["mask"].to(device)

        preds = model(batch_graph, rdkit, qc, qc_mask)

        loss, _ = compute_loss(model, preds, targets, mask)

        total_loss += loss.item()

    return total_loss




def main_old():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QW_MTL(num_tasks=NUM_TASKS)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # 🔴 TODO: podmień na prawdziwe dataloadery
    train_loader = dummy_loader()
    val_loader = dummy_loader()

    train(model, train_loader, optimizer, scheduler, device)

    val_loss = evaluate(model, val_loader, device)

    print("Final Val Loss:", val_loss)



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔴 1. LOAD DATA
    data, tasks = load_dataset("dataset_raw.parquet")

    # 🔴 2. NORMALIZE
    data, _, _ = normalize_features(data)

    # 🔴 3. SPLIT
    train_data, val_data, test_data = split_data(data)

    # 🔴 4. DATALOADER
    train_loader = build_dataloader(train_data, batch_size=32)
    val_loader = build_dataloader(val_data, batch_size=32, shuffle=False)

    # 🔴 5. MODEL
    model = QW_MTL(num_tasks=len(tasks)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # 🔴 6. TRAIN
    train(model, train_loader, optimizer, scheduler, device)

    # 🔴 7. EVAL
    val_loss = evaluate(model, val_loader, device)

    print("Final Val Loss:", val_loss)


if __name__ == "__main__":
    main()

