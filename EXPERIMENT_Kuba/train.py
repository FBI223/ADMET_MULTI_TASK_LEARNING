from xgboost import XGBClassifier

from EXPERIMENT.config import Config
from plots import *
from EXPERIMENT.data import get_full_data
from EXPERIMENT.model import ADMET_Hybrid_Model,  MaskedBCELoss
import torch
import numpy as np
import pandas as pd


def get_predictions_and_labels(model, loader, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            # Model zwraca listę tensorów (jeden na zadanie)
            preds_list = model(batch)
            # Łączymy w tensor [batch_size, num_tasks] i nakładamy Sigmoid
            preds_combined = torch.cat([torch.sigmoid(p) for p in preds_list], dim=1)

            all_preds.append(preds_combined.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    return np.vstack(all_targets), np.vstack(all_preds)


# --- FUNKCJE POMOCNICZE DLA XGBOOST ---
def prepare_flat_features(dataset):
    """Konwertuje obiekty grafowe na płaskie wektory dla XGBoost z czyszczeniem danych."""
    X, Y = [], []
    for data in dataset:
        feats = []

        # Morgan Fingerprints (zazwyczaj [1, 1024])
        if hasattr(data, 'morgan'):
            feats.append(data.morgan.numpy().reshape(1, -1))

        # RDKit Descriptors (zazwyczaj [1, 208])
        if hasattr(data, 'rdkit'):
            # Konwersja na numpy i usuwanie ewentualnych Inf/NaN przed konkatenacją
            rd_feat = data.rdkit.numpy().reshape(1, -1)
            rd_feat = np.nan_to_num(rd_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            feats.append(rd_feat)

        # Jeśli nie ma wektorów, a chcesz użyć XGBoost, musisz zrobić Pooling z grafu
        if len(feats) == 0 and hasattr(data, 'x'):
            graph_mean = data.x.mean(dim=0).numpy().reshape(1, -1)
            feats.append(graph_mean)

        if feats:
            X.append(np.concatenate(feats, axis=1))
            Y.append(data.y.numpy().reshape(1, -1))

    # Konwersja na finalne macierze
    X_final = np.vstack(X)
    Y_final = np.vstack(Y)

    # Ostateczne czyszczenie całej macierzy (podwójna weryfikacja dla XGBoost)
    X_final = np.nan_to_num(X_final, nan=0.0, posinf=1e6, neginf=-1e6)

    return X_final, Y_final


def train_stl_and_evaluate(train_loader, val_loader, test_loader, config, full_df):

    results = []
    stl_scores = {}

    # --- wizualizacje danych (tak jak MTL) ---
    plot_data_sparsity(full_df, config.tasks)
    plot_label_correlations(full_df, config.tasks)

    for task_idx, task_name in enumerate(config.tasks):

        print(f"\n>>> STL: {task_name}")

        model = ADMET_Hybrid_Model(config, single_task_idx=task_idx).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        history = {"train_loss": [], "val_auroc": []}

        best_val_auc = 0
        patience = 0
        early_stop = 8

        # --- TRAIN ---
        for epoch in range(config.epochs):

            model.train()
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(config.device)

                y = batch.y[:, task_idx].unsqueeze(1)
                mask = ~torch.isnan(y)

                if mask.sum() == 0:
                    continue

                optimizer.zero_grad()

                out = model(batch)
                loss = criterion(out[mask], y[mask])

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # --- VAL ---
            model.eval()
            y_true, y_pred = [], []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(config.device)

                    y = batch.y[:, task_idx].unsqueeze(1)
                    mask = ~torch.isnan(y)

                    if mask.sum() == 0:
                        continue

                    out = model(batch)
                    pred = torch.sigmoid(out)

                    y_true.extend(y[mask].cpu().numpy())
                    y_pred.extend(pred[mask].cpu().numpy())

            if len(np.unique(y_true)) > 1:
                val_auc = roc_auc_score(y_true, y_pred)
            else:
                val_auc = 0.5

            history["train_loss"].append(total_loss / len(train_loader))
            history["val_auroc"].append(val_auc)

            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val AUC: {val_auc:.4f}")

            # --- early stopping ---
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_weights = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    print("Early stopping")
                    break

        # --- plot training ---
        plot_training_results(history, title=f"STL - {task_name}")

        # --- TEST ---
        model.load_state_dict(best_weights)
        model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(config.device)

                y = batch.y[:, task_idx].unsqueeze(1)
                mask = ~torch.isnan(y)

                if mask.sum() == 0:
                    continue

                out = model(batch)
                pred = torch.sigmoid(out)

                y_true.extend(y[mask].cpu().numpy())
                y_pred.extend(pred[mask].cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        auc = roc_auc_score(y_true, y_pred)
        stl_scores[task_name] = auc

        print(f"TEST {task_name}: AUC = {auc:.4f}")

        # --- ROC per task ---
        plot_single_roc(y_true, y_pred, task_name)

        results.append({
            "Task": task_name,
            "AUROC": auc,
            "Model_Type": "STL_GNN"
        })

    return stl_scores, pd.DataFrame(results)

def train_single_task_xgboost(train_ds, test_ds, config):
    """Trenuje 13 osobnych modeli XGBoost."""
    print("\n>>> Trenowanie 13 modeli XGBoost (Single-Task)...")
    X_train, Y_train = prepare_flat_features(train_ds)
    X_test, Y_test = prepare_flat_features(test_ds)

    xgb_results = {}

    for i, task in enumerate(config.tasks):
        # Filtrowanie NaN dla konkretnego zadania
        mask_train = ~np.isnan(Y_train[:, i])
        mask_test = ~np.isnan(Y_test[:, i])

        if mask_train.sum() > 0 and mask_test.sum() > 0:
            model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False,
                                  eval_metric='logloss')
            model.fit(X_train[mask_train], Y_train[mask_train, i])

            preds = model.predict_proba(X_test[mask_test])[:, 1]
            auc = roc_auc_score(Y_test[mask_test, i], preds)
            xgb_results[task] = auc
            print(f"XGBoost {task}: AUROC = {auc:.4f}")

    return xgb_results


# --- PĘTLA TRENINGOWA MULTI-TASK (GNN) ---
def train_mtl_and_visualize(train_loader, val_loader, test_loader, config, full_df):
    model = ADMET_Hybrid_Model(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = MaskedBCELoss()

    history = {"train_loss": [], "val_auroc": []}

    # 1. Wizualizacja rzadkości danych przed startem
    plot_data_sparsity(full_df, config.tasks)
    plot_label_correlations(full_df, config.tasks)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(config.device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Szybka ewaluacja na zbiorze walidacyjnym
        y_true_v, y_pred_v = get_predictions_and_labels(model, val_loader, config)
        val_aucs = []
        for i in range(len(config.tasks)):
            mask = ~np.isnan(y_true_v[:, i])
            if len(np.unique(y_true_v[mask, i])) > 1:
                val_aucs.append(roc_auc_score(y_true_v[mask, i], y_pred_v[mask, i]))

        avg_val_auc = np.mean(val_aucs)
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_auroc"].append(avg_val_auc)

        print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.4f} | Val AUC: {avg_val_auc:.4f}")

    # 2. Wizualizacja po treningu
    plot_training_results(history, title="MTL GNN Hybrid")

    # 3. Finalna ewaluacja na zbiorze testowym
    y_true_t, y_pred_t = get_predictions_and_labels(model, test_loader, config)

    plot_all_roc_curves(y_true_t, y_pred_t, config.tasks)
    plot_task_correlation(y_pred_t, config.tasks)
    plot_multi_task_confusion_matrices(y_true_t, y_pred_t, config.tasks)

    return model, y_true_t, y_pred_t


def run_full_benchmarking(train_loader, val_loader, test_loader, train_ds, test_ds, config, full_df):
    results_list = []
    dataset_sizes = {task: full_df[task].notnull().sum() for task in config.tasks}

    # --- MODEL 1: MTL GNN ---
    mtl_model, y_true_mtl, y_pred_mtl = train_mtl_and_visualize(train_loader, val_loader, test_loader, config, full_df)
    mtl_scores = evaluate_per_task(y_true_mtl, y_pred_mtl, config.tasks)
    for task, score in mtl_scores.items():
        results_list.append({'Task': task, 'AUROC': score, 'Model_Type': 'MTL_GNN'})

    # --- MODEL 2: STL GNN (Pętla po każdym zadaniu) ---
    stl_scores_dict = {}
    for i, task in enumerate(config.tasks):
        print(f"\n>>> Trenowanie STL GNN dla: {task}")
        # Tworzymy model z tylko jedną głowicą lub maskujemy resztę
        # Dla uproszczenia używamy tej samej architektury, ale liczymy loss tylko dla jednego zadania
        model_stl = ADMET_Hybrid_Model(config).to(config.device)
        # ... (skrócona pętla treningowa tylko dla zadania i) ...
        # Po treningu:
        score_stl = evaluate_single_task(model_stl, test_loader, i, config)
        results_list.append({'Task': task, 'AUROC': score_stl, 'Model_Type': 'STL_GNN'})
        stl_scores_dict[task] = score_stl

    # --- MODEL 3: XGBoost ---
    xgb_results = train_single_task_xgboost(train_ds, test_ds, config)
    for task, score in xgb_results.items():
        results_list.append({'Task': task, 'AUROC': score, 'Model_Type': 'XGBoost'})

        # Jeśli masz już model XGBoost, narysuj ważność cech dla pierwszego zadania
        if task == config.tasks[0]:
            # plot_feature_importance_xgboost(last_xgb_model, feature_names)
            pass

    # --- FINALNA WIZUALIZACJA PORÓWNAWCZA ---
    results_df = pd.DataFrame(results_list)
    plot_model_comparison(results_df)

    # Wizualizacja zysku z MTL względem wielkości danych
    mtl_gain = {t: mtl_scores[t] - stl_scores_dict[t] for t in config.tasks}
    plot_performance_vs_size(mtl_gain, dataset_sizes)


def debug_epoch_zero(loader, model, config):
    print("\n=== ANALIZA DEBUGUJĄCA: EPOKA 0 (Pierwszy Batch) ===")
    model.eval()
    batch = next(iter(loader)).to(config.device)

    with torch.no_grad():
        # 1. Sprawdzenie surowych wejść
        print(
            f"Node features (x) - Min: {batch.x.min().item():.4f}, Max: {batch.x.max().item():.4f}, NaN: {torch.isnan(batch.x).any()}")

        if hasattr(batch, 'rdkit'):
            print(
                f"RDKit desc - Min: {batch.rdkit.min().item():.4f}, Max: {batch.rdkit.max().item():.4f}, NaN: {torch.isnan(batch.rdkit).any()}")
            # Sprawdźmy ile jest wartości ekstremalnych
            extreme_vals = (batch.rdkit.abs() > 1000).sum().item()
            print(f"Liczba wartości RDKit > 1000: {extreme_vals}")

        # 2. Przejście przez model krok po kroku
        # Sprawdzamy wyjście z GNN
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for i, conv in enumerate(model.gin_backbone):
            x = conv(x, edge_index)
            print(f"Po GIN Layer {i} - Max: {x.max().item():.4f}, NaN: {torch.isnan(x).any()}")

        # Sprawdzamy predykcje głowic
        out_heads = model(batch)
        for i, head_out in enumerate(out_heads):
            if torch.isnan(head_out).any() or torch.isinf(head_out).any():
                print(f"!!! Głowica zadania {config.tasks[i]} wygenerowała NaN/Inf!")

    print("=== KONIEC ANALIZY ===\n")


def main():

    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    # --- NOWE: SEKCJA PRE-CACHE (identyczna jak w run_experiments) ---
    print("\n>>> Przygotowanie Master Cache (zapewnienie kompletności danych)...")
    cache_cfg = Config()
    # Włączamy wszystko, aby Master Cache zawierał komplet deskryptorów
    #cache_cfg.use_graph, cache_cfg.use_rdkit, cache_cfg.use_morgan = True, True, True
    # To wywołanie stworzy plik .pt, jeśli go nie ma, lub nic nie zrobi, jeśli już jest
    get_full_data(cache_cfg)
    print(">>> Dane w cache są gotowe.\n")


    # 1. Pobieranie danych
    train_loader, val_loader, test_loader, train_ds, test_ds, raw_train_df = get_full_data(cfg)

    # 2.1 XGBoost Baseline
    print("\n>>> Trenowanie XGBoost (STL)...")
    xgb_scores = train_single_task_xgboost(train_ds, test_ds, cfg)

    # 2.2 STL Baseline GNN
    print("\n>>> Trenowanie STL GNN...")
    stl_scores, _ = train_stl_and_evaluate(
        train_loader, val_loader, test_loader, cfg, raw_train_df
    )

    # 3. MTL GNN (Multi-Task)
    print("\n>>> Trenowanie GNN (MTL)...")
    mtl_model = ADMET_Hybrid_Model(cfg).to(cfg.device)
    opt = torch.optim.Adam(mtl_model.parameters(), lr=cfg.lr)
    crit = MaskedBCELoss()

    # --- NOWE: MECHANIZMY KONTROLNE ---
    # Scheduler: zmniejsza LR o połowę, jeśli AUC nie rośnie przez 3 epoki
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    best_val_auc = 0
    patience_counter = 0
    early_stop_patience = 8  # Po ilu epokach bez poprawy przerwać trening
    model_path = os.path.join(cfg.results_dir, "best_mtl_model.pt")  # Ścieżka zapisu

    history = {"train_loss": [], "val_auroc": []}

    debug_epoch_zero(train_loader, mtl_model, cfg)

    for epoch in range(cfg.epochs):
        mtl_model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(cfg.device)
            opt.zero_grad()
            out = mtl_model(batch)
            loss = crit(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Walidacja
        val_scores = evaluate_gnn_simple(mtl_model, val_loader, cfg)
        avg_val_auc = np.mean(list(val_scores.values()))

        # Aktualizacja Schedulera
        scheduler.step(avg_val_auc)

        history["train_loss"].append(total_loss / len(train_loader))
        history["val_auroc"].append(avg_val_auc)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | Loss: {total_loss / len(train_loader):.4f} | Val AUC: {avg_val_auc:.4f}")

        # --- NOWE: LOGIKA ZAPISU I EARLY STOPPING ---
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            patience_counter = 0
            # Zapisujemy cały stan modelu
            torch.save({
                'epoch': epoch,
                'model_state_dict': mtl_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_auc': best_val_auc,
            }, model_path)
            print(f"  >>> Zapisano nową najlepszą wersję modelu (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  !!! Early Stopping wywołany po {epoch + 1} epokach.")
                break

    # --- NOWE: WCZYTANIE NAJLEPSZEGO MODELU PRZED TESTAMI ---
    print(
        f"\n>>> Wczytywanie najlepszego modelu z epoki {torch.load(model_path, weights_only=False)['epoch'] + 1} do finalnej ewaluacji...")
    checkpoint = torch.load(model_path, weights_only=False)
    mtl_model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Ewaluacja końcowa na zbiorze testowym
    mtl_model.eval()
    y_true, y_pred = [], [[] for _ in cfg.tasks]
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(cfg.device)
            out = mtl_model(batch)
            y_true.append(batch.y.cpu().numpy())
            for i in range(len(cfg.tasks)):
                y_pred[i].extend(torch.sigmoid(out[i]).cpu().numpy().flatten())

    y_true = np.vstack(y_true)

    # Wykresy i zapis wyników
    plot_all_roc_curves(y_true, np.array(y_pred).T, cfg.tasks)
    plot_training_results(history, title="MTL GNN Training")

    gnn_scores = evaluate_per_task(y_true, y_pred, cfg.tasks)
    results_df = pd.DataFrame({
        'Task': cfg.tasks,
        'MTL_GNN': [gnn_scores[t] for t in cfg.tasks],
        'STL_GNN': [stl_scores[t] for t in cfg.tasks],
        'XGBoost': [xgb_scores[t] for t in cfg.tasks]
    })
    results_df.to_csv(f"{cfg.results_dir}/final_results.csv", index=False)
    plot_model_comparison_simple(results_df)

    print(f"\nGOTOWE! Najlepszy model: {model_path}")


def run_experiments():
    # 1. Definicja 6 kombinacji wejść
    combinations = [
        (True, False, False),  # Graph Only
        (True, True, False),  # Graph + RDKit
        (True, False, True),  # Graph + Morgan
        (True, True, True),  # All (Hybrid)
        (False, True, False),  # RDKit Only (MLP)
        (False, False, True),  # Morgan Only (MLP)
    ]

    base_results_dir = "experiments_results"
    os.makedirs(base_results_dir, exist_ok=True)
    summary_results = []

    # --- CACHE DATA SECTION ---
    print("\n>>> Przygotowanie cache danych (obliczanie wszystkich deskryptorów)...")
    cache_cfg = Config()
    cache_cfg.use_graph, cache_cfg.use_rdkit, cache_cfg.use_morgan = True, True, True
    # Wywołanie get_full_data raz wypełnia pliki .pt na dysku
    get_full_data(cache_cfg)
    print(">>> Dane gotowe w cache.\n")

    for g, r, m in combinations:

        features = []
        if g: features.append("Graph")
        if r: features.append("RDKit")
        if m: features.append("Morgan")
        exp_label = " + ".join(features) if features else "Baseline"

        print(f"\n{'=' * 50}\n ROZPOCZYNAM: {exp_label}\n{'=' * 50}")

        cfg = Config()
        cfg.use_graph, cfg.use_rdkit, cfg.use_morgan = g, r, m
        cfg.results_dir = os.path.join(base_results_dir, exp_label)
        os.makedirs(cfg.results_dir, exist_ok=True)

        train_loader, val_loader, test_loader, _, _, _ = get_full_data(cfg)

        model = ADMET_Hybrid_Model(cfg).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = MaskedBCELoss()

        # Scheduler pomaga ustabilizować trening
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_val_auc = 0
        history = {"epoch": [], "train_loss": [], "val_auroc": []}

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(cfg.device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Walidacja
            y_true_v, y_pred_v = get_predictions_and_labels(model, val_loader, cfg)
            val_scores = evaluate_per_task(y_true_v, y_pred_v, cfg.tasks)
            avg_val_auc = np.mean(list(val_scores.values()))

            scheduler.step(avg_val_auc)

            # Zapis do history (Teraz używane!)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(epoch_loss / len(train_loader))  # Naprawiony loss
            history["val_auroc"].append(avg_val_auc)

            if avg_val_auc > best_val_auc:
                best_val_auc = avg_val_auc
                torch.save(model.state_dict(), os.path.join(cfg.results_dir, "best_model.pt"))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:02d} | Loss: {history['train_loss'][-1]:.4f} | Val AUC: {avg_val_auc:.4f}")

        # --- FINALIZACJA EKSPERYMENTU ---
        # Zapis historii do CSV
        pd.DataFrame(history).to_csv(os.path.join(cfg.results_dir, "history.csv"), index=False)

        # Wykres lokalny (Używając Twojej funkcji z plots.py)
        plot_training_results(history, title=f"MTL Training - {exp_label}")

        # Testowanie najlepszego modelu
        model.load_state_dict(torch.load(os.path.join(cfg.results_dir, "best_model.pt"), weights_only=False))
        y_true_t, y_pred_t = get_predictions_and_labels(model, test_loader, cfg)
        test_scores = evaluate_per_task(y_true_t, y_pred_t, cfg.tasks)
        final_avg_auc = np.mean(list(test_scores.values()))

        # Zbieranie wyników do tabeli zbiorczej
        res_entry = {"Experiment": exp_label, "Avg_AUROC": final_avg_auc}
        res_entry.update(test_scores)
        summary_results.append(res_entry)

        print(f"KONIEC {exp_label} | Test Avg AUC: {final_avg_auc:.4f}")

    # 2. Raport zbiorczy i wykres porównawczy
    df_results = pd.DataFrame(summary_results)
    df_results.to_csv(os.path.join(base_results_dir, "comparison_results.csv"), index=False)

    # Wywołanie nowej funkcji wykresu
    plot_experiment_comparison(df_results, base_results_dir)



if __name__ == "__main__":
    main()
    #run_experiments()
