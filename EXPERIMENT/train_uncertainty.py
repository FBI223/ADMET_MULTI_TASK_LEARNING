"""
EXPERIMENT_UNCERTAINTY — Homoscedastic Uncertainty-Weighted Multi-Task Learning

Zamiast równego ważenia wszystkich zadań (L = sum L_t), uczony jest
parametr sigma_t per zadanie. Zadania z większym szumem/trudnością
dostają automatycznie mniejszą wagę.

Podstawa matematyczna (Kendall et al., CVPR 2018):
    L = sum_t [ L_t / exp(log_sigma_sq_t) + log_sigma_sq_t / 2 ]

Nowe wyjścia vs. EXPERIMENT_MAIN:
    - sigma_evolution.csv  — wartości sigma per zadanie per epoka
    - sigma_final.png      — wizualizacja nauczonych sigma po treningu
    - sigma_evolution.png  — przebieg zmian sigma w trakcie treningu

Uruchomienie (z katalogu głównego projektu):
    python -m EXPERIMENT.train_uncertainty
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# --- Importy z istniejących modułów projektu ---
from EXPERIMENT.config import Config
from EXPERIMENT.model import ADMET_Hybrid_Model
from EXPERIMENT.data import get_full_data
from EXPERIMENT.plots import (
    evaluate_per_task,
    evaluate_gnn_simple,
    plot_training_results,
    plot_all_roc_curves,
    plot_task_correlation,
    plot_multi_task_confusion_matrices,
    plot_model_comparison_simple,
    plot_data_sparsity,
    plot_label_correlations,
    plot_single_roc,
)


# ============================================================
# NOWA KLASA: Uncertainty-Weighted Masked Loss
# ============================================================

class UncertaintyWeightedMaskedLoss(nn.Module):
    """
    Multi-task loss z uczalnymi wagami niepewności (homoscedastic uncertainty).

    Dla każdego zadania t uczy się log_sigma_sq_t = log(sigma_t^2).
    Gdy sigma_t rośnie, zadanie jest uznawane za bardziej niepewne
    i jego loss jest down-weightowany automatycznie.

    Wzór:
        L_total = sum_t [ L_t / exp(log_sigma_sq_t) + log_sigma_sq_t / 2 ]

    Inicjalizacja: log_sigma_sq = 0 -> sigma = 1 -> na starcie identyczne z równym ważeniem.

    Parametry do optymalizatora przekazujemy przez criterion.parameters().
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        # Jeden parametr per zadanie, inicjalizowany na 0
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_tasks))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: list, targets: torch.Tensor):
        """
        preds   : lista tensorów [batch, 1], długość = n_tasks (wyjście modelu)
        targets : tensor [batch, n_tasks] z wartościami NaN dla brakujących etykiet

        Zwraca:
            total_loss   : skalar do backprop
            task_losses  : lista raw loss per zadanie (detach, do logowania)
            sigma_values : lista sigma_t po detach (do logowania)
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        task_losses = []
        sigma_values = []

        for t, pred in enumerate(preds):
            mask = ~torch.isnan(targets[:, t])

            if not mask.any():
                task_losses.append(float('nan'))
                sigma_values.append(float('nan'))
                continue

            y_t = targets[mask, t].unsqueeze(1)
            p_t = pred[mask]

            # Średni raw BCE dla tego zadania
            raw_loss = self.bce(p_t, y_t).mean()

            # Uncertainty weighting:
            #   precision = 1 / sigma^2 = exp(-log_sigma_sq)
            #   weighted  = raw_loss * precision + log_sigma_sq / 2
            precision = torch.exp(-self.log_sigma_sq[t])
            weighted_loss = precision * raw_loss + self.log_sigma_sq[t] / 2.0

            total_loss = total_loss + weighted_loss

            task_losses.append(raw_loss.item())
            # sigma_t = exp(log_sigma_sq / 2)
            sigma_values.append(torch.exp(self.log_sigma_sq[t] / 2.0).item())

        return total_loss, task_losses, sigma_values

    def get_sigmas(self) -> list:
        """Zwraca aktualne wartości sigma (do logowania poza forward)."""
        return [torch.exp(s / 2.0).item() for s in self.log_sigma_sq]


# ============================================================
# NOWE WYKRESY
# ============================================================

def plot_sigma_final(sigma_values: list, tasks: list, save_dir: str):
    """
    Słupkowy wykres nauczonych sigma po treningu.
    Wysokie sigma = zadanie niepewne/trudne = model down-weightował.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if s > 1.1 else '#2ecc71' for s in sigma_values]
    bars = ax.bar(tasks, sigma_values, color=colors, edgecolor='black', alpha=0.85)

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='sigma=1 (równe ważenie)')
    ax.set_xlabel('Zadanie ADMET')
    ax.set_ylabel('Nauczona sigma (niepewność zadania)')
    ax.set_title('Homoscedastic Uncertainty — nauczone sigma per zadanie\n'
                 'Czerwony = model uznał za trudniejsze/bardziej zaszumione')
    ax.legend()

    for bar, val in zip(bars, sigma_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sigma_final.png'), dpi=150)
    plt.close()
    print(f"  [PLOT] Zapisano sigma_final.png")


def plot_sigma_evolution(sigma_history: pd.DataFrame, save_dir: str):
    """
    Wykres liniowy — jak zmieniały się sigma w trakcie treningu.
    Pozwala zobaczyć, kiedy model zaczął różnicować zadania.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette('tab20', n_colors=len(sigma_history.columns))

    for i, task in enumerate(sigma_history.columns):
        ax.plot(sigma_history.index + 1, sigma_history[task],
                label=task, color=palette[i], linewidth=1.8)

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.4, label='sigma=1 (start)')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('sigma_t (niepewność zadania)')
    ax.set_title('Ewolucja sigma per zadanie w trakcie treningu\n'
                 '(sigma > 1 = model down-weightował zadanie)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sigma_evolution.png'), dpi=150)
    plt.close()
    print(f"  [PLOT] Zapisano sigma_evolution.png")


# ============================================================
# PĘTLA TRENINGOWA MTL Z UNCERTAINTY WEIGHTING
# ============================================================

def train_mtl_uncertainty(train_loader, val_loader, test_loader, config, full_df, results_dir):
    """
    Trenuje MTL GNN z UncertaintyWeightedMaskedLoss.
    Zwraca (model, y_true_test, y_pred_test, final_sigmas).
    """
    model = ADMET_Hybrid_Model(config).to(config.device)

    # Nowa funkcja straty z uczalnymi sigma
    criterion = UncertaintyWeightedMaskedLoss(n_tasks=len(config.tasks)).to(config.device)

    # KLUCZOWE: dodajemy parametry criterion (log_sigma_sq) do optymalizatora
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    history = {'train_loss': [], 'val_auroc': []}
    sigma_history = []  # lista słowników {task: sigma} per epoka

    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    model_path = os.path.join(results_dir, 'best_mtl_uncertainty_model.pt')

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\n>>> Wizualizacje danych treningowych...")
    plot_data_sparsity(full_df, config.tasks)
    plot_label_correlations(full_df, config.tasks)

    print(f"\n>>> Start treningu MTL z Uncertainty Weighting ({config.epochs} epok max)...")

    for epoch in range(config.epochs):
        model.train()
        criterion.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(config.device)
            optimizer.zero_grad()

            preds = model(batch)
            loss, _, _ = criterion(preds, batch.y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- Walidacja ---
        model.eval()
        criterion.eval()
        val_scores = evaluate_gnn_simple(model, val_loader, config)
        avg_val_auc = np.mean(list(val_scores.values()))
        scheduler.step(avg_val_auc)

        # --- Logowanie sigma ---
        current_sigmas = criterion.get_sigmas()
        sigma_history.append(dict(zip(config.tasks, current_sigmas)))

        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_auroc'].append(avg_val_auc)

        sigma_str = " | ".join(f"{t[:6]}:{s:.3f}" for t, s in zip(config.tasks, current_sigmas))
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(train_loader):.4f} "
              f"| Val AUC: {avg_val_auc:.4f} | sigma: [{sigma_str}]")

        # --- Early stopping + zapis najlepszego modelu ---
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'sigmas': current_sigmas,
            }, model_path)
            print(f"  >>> Nowy najlepszy model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  !!! Early Stopping po {epoch+1} epokach.")
                break

    # --- Wczytanie najlepszego checkpointu ---
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    final_sigmas = checkpoint['sigmas']
    print(f"\n>>> Wczytano model z epoki {checkpoint['epoch']+1} (Val AUC: {checkpoint['val_auc']:.4f})")

    # --- Zapis sigma_history ---
    sigma_df = pd.DataFrame(sigma_history)
    sigma_df.to_csv(os.path.join(results_dir, 'sigma_evolution.csv'), index=False)

    # --- Wykresy sigma ---
    plot_sigma_final(final_sigmas, config.tasks, results_dir)
    plot_sigma_evolution(sigma_df, results_dir)

    # --- Wykres treningu ---
    plot_training_results(history, title='MTL GNN + Uncertainty Weighting')

    # --- Ewaluacja testowa ---
    model.eval()
    y_true_parts, y_pred_parts = [], [[] for _ in config.tasks]

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.device)
            out = model(batch)
            y_true_parts.append(batch.y.cpu().numpy())
            for i in range(len(config.tasks)):
                y_pred_parts[i].extend(torch.sigmoid(out[i]).cpu().numpy().flatten())

    y_true_test = np.vstack(y_true_parts)

    # --- Wykresy testowe ---
    plot_all_roc_curves(y_true_test, np.array(y_pred_parts).T, config.tasks)
    plot_task_correlation(np.array(y_pred_parts).T, config.tasks)
    plot_multi_task_confusion_matrices(y_true_test, np.array(y_pred_parts).T, config.tasks)

    return model, y_true_test, y_pred_parts, final_sigmas


# ============================================================
# PĘTLA STL (taka sama jak w train.py, skopiowana lokalnie)
# ============================================================

def train_stl(train_loader, val_loader, test_loader, config, results_dir):
    """
    Trenuje osobny STL GNN dla każdego zadania.
    Identyczna logika jak w train.py::train_stl_and_evaluate.
    """
    from EXPERIMENT.plots import plot_single_roc

    stl_scores = {}
    roc_save_dir = os.path.join(results_dir, 'results')
    os.makedirs(roc_save_dir, exist_ok=True)

    for task_idx, task_name in enumerate(config.tasks):
        print(f"\n>>> STL: {task_name}")

        model = ADMET_Hybrid_Model(config, single_task_idx=task_idx).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion_stl = nn.BCEWithLogitsLoss()

        best_val_auc = 0.0
        patience = 0
        early_stop = 8
        best_weights = None

        for epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(config.device)
                y = batch.y[:, task_idx].unsqueeze(1)
                mask = ~torch.isnan(y)
                if mask.sum() == 0:
                    continue
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion_stl(out[mask], y[mask])
                loss.backward()
                optimizer.step()

            # Walidacja
            model.eval()
            y_true_v, y_pred_v = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(config.device)
                    y = batch.y[:, task_idx].unsqueeze(1)
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0:
                        continue
                    out = model(batch)
                    pred = torch.sigmoid(out)
                    y_true_v.extend(y[mask].cpu().numpy())
                    y_pred_v.extend(pred[mask].cpu().numpy())

            val_auc = roc_auc_score(y_true_v, y_pred_v) if len(np.unique(y_true_v)) > 1 else 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_weights = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    break

        # Test
        model.load_state_dict(best_weights)
        model.eval()
        y_true_t, y_pred_t = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(config.device)
                y = batch.y[:, task_idx].unsqueeze(1)
                mask = ~torch.isnan(y)
                if mask.sum() == 0:
                    continue
                out = model(batch)
                pred = torch.sigmoid(out)
                y_true_t.extend(y[mask].cpu().numpy())
                y_pred_t.extend(pred[mask].cpu().numpy())

        auc_score = roc_auc_score(y_true_t, y_pred_t) if len(np.unique(y_true_t)) > 1 else 0.5
        stl_scores[task_name] = auc_score
        print(f"  TEST {task_name}: AUC = {auc_score:.4f}")

        plot_single_roc(y_true_t, y_pred_t, task_name, save_dir=roc_save_dir)

    return stl_scores


# ============================================================
# XGBOOST (identyczny jak w train.py)
# ============================================================

def train_xgboost(train_ds, test_ds, config):
    from xgboost import XGBClassifier

    X_train, Y_train = _prepare_flat_features(train_ds)
    X_test, Y_test = _prepare_flat_features(test_ds)
    xgb_results = {}

    for i, task in enumerate(config.tasks):
        mask_tr = ~np.isnan(Y_train[:, i])
        mask_te = ~np.isnan(Y_test[:, i])
        if mask_tr.sum() == 0 or mask_te.sum() == 0:
            continue

        clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                            use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train[mask_tr], Y_train[mask_tr, i])
        preds = clf.predict_proba(X_test[mask_te])[:, 1]
        auc_score = roc_auc_score(Y_test[mask_te, i], preds)
        xgb_results[task] = auc_score
        print(f"  XGBoost {task}: AUROC = {auc_score:.4f}")

    return xgb_results


def _prepare_flat_features(dataset):
    X, Y = [], []
    for data in dataset:
        feats = []
        if hasattr(data, 'morgan'):
            feats.append(data.morgan.numpy().reshape(1, -1))
        if hasattr(data, 'rdkit'):
            rd = data.rdkit.numpy().reshape(1, -1)
            rd = np.nan_to_num(rd, nan=0.0, posinf=1e6, neginf=-1e6)
            feats.append(rd)
        if len(feats) == 0 and hasattr(data, 'x'):
            feats.append(data.x.mean(dim=0).numpy().reshape(1, -1))
        if feats:
            X.append(np.concatenate(feats, axis=1))
            Y.append(data.y.numpy().reshape(1, -1))
    X_final = np.nan_to_num(np.vstack(X), nan=0.0, posinf=1e6, neginf=-1e6)
    return X_final, np.vstack(Y)


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Konfiguracja ---
    cfg = Config()

    # Używamy tych samych zadań co EXPERIMENT_MAIN (tasks_old = 13 zadań)
    cfg.tasks = cfg.tasks_old

    # Pełna hybryda: GNN + RDKit + Morgan (najlepsza konfiguracja z EXPERIMENT_MAIN)
    cfg.use_graph = True
    cfg.use_rdkit = True
    cfg.use_morgan = True

    # Folder wyników
    results_dir = os.path.join(
        os.path.dirname(__file__),
        'EXPERIMENT_UNCERTAINTY',
        'GNN_RDKIT_MORGAN'
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'results'), exist_ok=True)

    # Przekierowanie stdout do pliku run_stats.txt (oraz na ekran)
    log_path = os.path.join(results_dir, 'run_stats.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, msg):
            for s in self.streams:
                s.write(msg)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.stdout, log_file)

    print("=" * 60)
    print("EXPERIMENT_UNCERTAINTY — Homoscedastic Uncertainty Weighting")
    print("=" * 60)
    print(f"Zadania ({len(cfg.tasks)}): {cfg.tasks}")
    print(f"Wejścia: graph={cfg.use_graph}, rdkit={cfg.use_rdkit}, morgan={cfg.use_morgan}")
    print(f"Device: {cfg.device}")
    print(f"Wyniki -> {results_dir}")
    print()

    # --- Cache ---
    print(">>> Przygotowanie Master Cache...")
    cache_cfg = Config()
    cache_cfg.tasks = cfg.tasks
    cache_cfg.use_graph = cache_cfg.use_rdkit = cache_cfg.use_morgan = True
    get_full_data(cache_cfg)
    print(">>> Cache gotowy.\n")

    # --- Dane ---
    train_loader, val_loader, test_loader, train_ds, test_ds, raw_train_df = get_full_data(cfg)

    # Zapis info o datasetach
    datasets_path = os.path.join(os.path.dirname(results_dir), 'datasets.txt')
    with open(datasets_path, 'w') as f:
        for task in cfg.tasks:
            n = raw_train_df[task].notnull().sum() if task in raw_train_df.columns else 'N/A'
            f.write(f"{task}: {n} próbek treningowych\n")

    # --- MODEL 1: MTL GNN + Uncertainty Weighting ---
    print("\n>>> [1/3] Trenowanie MTL GNN + Uncertainty Weighting...")
    mtl_model, y_true_test, y_pred_test, final_sigmas = train_mtl_uncertainty(
        train_loader, val_loader, test_loader, cfg, raw_train_df, results_dir
    )
    mtl_scores = evaluate_per_task(y_true_test, y_pred_test, cfg.tasks)

    # --- MODEL 2: STL GNN ---
    print("\n>>> [2/3] Trenowanie STL GNN (baseline)...")
    stl_scores = train_stl(train_loader, val_loader, test_loader, cfg, results_dir)

    # --- MODEL 3: XGBoost ---
    print("\n>>> [3/3] Trenowanie XGBoost (baseline)...")
    xgb_scores = train_xgboost(train_ds, test_ds, cfg)

    # --- Zapis wyników końcowych ---
    results_df = pd.DataFrame({
        'Task': cfg.tasks,
        'MTL_GNN_Uncertainty': [mtl_scores.get(t, float('nan')) for t in cfg.tasks],
        'STL_GNN':             [stl_scores.get(t, float('nan')) for t in cfg.tasks],
        'XGBoost':             [xgb_scores.get(t, float('nan')) for t in cfg.tasks],
        'Final_Sigma':         final_sigmas,
    })
    csv_path = os.path.join(results_dir, 'final_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n>>> Wyniki zapisane do: {csv_path}")
    print(results_df.to_string(index=False))

    # --- Porównanie wizualne ---
    df_for_plot = results_df[['Task', 'MTL_GNN_Uncertainty', 'STL_GNN', 'XGBoost']].copy()
    df_for_plot.columns = ['Task', 'MTL_GNN', 'STL_GNN', 'XGBoost']
    plot_model_comparison_simple(df_for_plot)

    # --- Podsumowanie sigma ---
    print("\n>>> Nauczone sigma (wyższe = zadanie bardziej niepewne):")
    for task, sigma in zip(cfg.tasks, final_sigmas):
        arrow = " <-- najtrudniejsze" if sigma == max(final_sigmas) else ""
        print(f"  {task:<35} sigma = {sigma:.4f}{arrow}")

    log_file.close()
    print(f"\nGOTOWE. Logi: {log_path}")


if __name__ == '__main__':
    main()
