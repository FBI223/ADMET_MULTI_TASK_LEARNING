import torch
from torch.utils.data import DataLoader, random_split
from model import MTAN_ADMET
from trainer import MTANTrainer
from data_utils import ADMETDataLoader, get_task_metadata
import time
import numpy as np

# Parametry zgodne z publikacją [cite: 334]
CONFIG = {
    "batch_size": 512,
    "epochs": 50,
    "base_lr": 0.001,
    "k": 2,
    "weight_decay": 1e-4,
    "cddd_path": "./cddd_model",
    "data_path": "admet_data.csv"
}


def print_summary(metrics, is_class):
    print("\n" + "=" * 50)
    print(f"{'Zadanie':<30} | {'Metryka':<10} | {'Wynik':<8}")
    print("-" * 50)
    for i, score in enumerate(metrics):
        m_type = "AUC" if is_class[i] else "R^2"
        print(f"Task {i + 1:<25} | {m_type:<10} | {score:.4f}")
    print("=" * 50)
    print(f"Średni wynik (Average Performance): {np.mean(metrics):.4f} ")


def main():
    print("--- MTAN-ADMET: Parametry Treningu ---")
    for k, v in CONFIG.items(): print(f"{k}: {v}")

    # 1. Ładowanie i podział danych (8:1:1) [cite: 382]
    full_dataset = ADMETDataLoader(CONFIG["data_path"], CONFIG["cddd_path"])
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

    metadata = get_task_metadata(CONFIG["data_path"])

    # 2. Inicjalizacja
    model = MTAN_ADMET()
    trainer = MTANTrainer(model, metadata, CONFIG["base_lr"], CONFIG["k"])

    best_val_score = -float('inf')

    # 3. Pętla treningowa
    print("\nRozpoczynanie treningu (50 epok)...")
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        train_loss = trainer.train_step(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        avg_val = np.mean(val_metrics)

        trainer.step_schedulers()

        # Zapis najlepszego modelu [cite: 659]
        if avg_val > best_val_score:
            best_val_score = avg_val
            torch.save(model.state_dict(), "best_mtan_model.pt")

        print(
            f"Epoch {epoch + 1:02d} | Loss: {train_loss:.4f} | Val Avg: {avg_val:.4f} | Time: {time.time() - start_time:.1f}s")

    # 4. Finalna ewaluacja na zbiorze testowym
    print("\nŁadowanie najlepszego modelu do testów...")
    model.load_state_dict(torch.load("best_mtan_model.pt"))
    test_metrics = trainer.evaluate(test_loader)

    print_summary(test_metrics, metadata['is_classification'])


if __name__ == "__main__":
    main()