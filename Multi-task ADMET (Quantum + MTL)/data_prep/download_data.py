import pandas as pd
import os
from tdc.single_pred import ADME, Tox

# --- KONFIGURACJA (Twoje parametry) ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX


def download_tdc_data():
    """Pobiera dane zachowując niezależne splity scaffold dla każdego zadania."""
    print(f"KROK 1: Pobieranie {len(ALL_TASKS)} zadań z TDC (Scaffold Split)...")
    all_data = []

    for name in ALL_TASKS:
        print(f" Przetwarzanie zadania: {name}...")
        try:
            # Wybór odpowiedniej klasy TDC na podstawie Twojej listy
            group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)

            # Pobranie oficjalnego podziału (Scaffold Split)
            split = group.get_split()

            for s_name in ["train", "valid", "test"]:
                # Pobieramy Drug (SMILES) i Y (Label)
                df = split[s_name][['Drug', 'Y']].copy()

                # Standaryzacja nazw kolumn
                df.columns = ['smiles', 'label']

                # Dodanie metadanych, które będą potrzebne do MTL
                df['task'] = name
                df['split'] = s_name

                all_data.append(df)
        except Exception as e:
            print(f" BŁĄD podczas pobierania {name}: {e}")

    # Połączenie wszystkich zadań w jedną ramkę danych
    final_raw = pd.concat(all_data, ignore_index=True)

    # Zapisanie do pliku RAW, który będzie używany jako INPUT do MOPAC/RDKit
    output_path = os.path.join(DATA_DIR, "tdc_multitask_raw.parquet")
    final_raw.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy"
    )

    print(f"\nSukces! Pobrano łącznie {len(final_raw)} wierszy.")
    print(f"Plik zapisany w: {output_path}")

    return final_raw


def generate_unique_smiles(df):
    """Generuje listę unikalnych SMILES do obliczeń (MOPAC/RDKit)."""
    unique_smiles = df[['smiles']].drop_duplicates()
    output_path = os.path.join(DATA_DIR, "unique_smiles_to_calculate.parquet")
    unique_smiles.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy"
    )
    print(f"Wygenerowano {len(unique_smiles)} unikalnych SMILES w: {output_path}")
    return unique_smiles


# --- URUCHOMIENIE ---
if __name__ == "__main__":
    # 1. Pobierz dane
    raw_df = download_tdc_data()
