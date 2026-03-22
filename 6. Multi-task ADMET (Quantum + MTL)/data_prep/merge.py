import pandas as pd
import os

# --- KONFIGURACJA ŚCIEŻEK ---
TASKS_RAW_FILE = "tdc_multitask_raw.csv"
QUANTUM_FILE = "tdc_final_quantum_full.csv"
RDKIT_FILE = "rdkit_descriptors.csv"
FINAL_OUTPUT = "dataset.csv"


def merge_everything_including_failures():
    print("--- ŁĄCZENIE WSZYSTKICH DANYCH (BEZ USUWANIA BŁĘDÓW) ---")

    # 1. Wczytywanie plików
    # Używamy drop_duplicates, aby upewnić się, że do jednego SMILES nie dokleimy danych dwa razy
    df_tasks = pd.read_csv(TASKS_RAW_FILE)
    df_quantum = pd.read_csv(QUANTUM_FILE).drop_duplicates(subset=['smiles'])
    df_rdkit = pd.read_csv(RDKIT_FILE).drop_duplicates(subset=['smiles'])

    initial_count = len(df_tasks)

    # 2. ŁĄCZENIE (Left Join)
    # Zaczynamy od bazy TDC i dołączamy deskryptory.
    # Jeśli czegoś brakuje w Quantum lub RDKit, Pandas wstawi NaN.
    merged = pd.merge(df_tasks, df_quantum, on='smiles', how='left')
    final_df = pd.merge(merged, df_rdkit, on='smiles', how='left')

    # 3. OBSŁUGA BRAKÓW (Zamiast usuwania)

    # Dla kolumn kwantowych: jeśli MOPAC nie istniał w ogóle w pliku, maski będą NaN.
    # Zamieniamy NaN w maskach na 0 (oznacza brak danych).
    mask_cols = ['mask_1', 'mask_2', 'mask_3', 'mask_4']
    for col in mask_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0).astype(int)

    # Wypełniamy brakujące wartości deskryptorów zerami, aby sieć neuronowa
    # nie wyrzuciła błędu podczas liczenia (maski i tak powiedzą modelowi, by je ignorował).
    quantum_features = ['dipole', 'homo_lumo', 'electrons', 'energy']
    final_df[quantum_features] = final_df[quantum_features].fillna(0)

    # Uzupełniamy też ewentualne braki w RDKit (jeśli jakieś SMILES się nie policzyło)
    # Pobieramy nazwy kolumn deskryptorów RDKit (wszystkie poza metadanymi)
    rdkit_cols = df_rdkit.columns.drop('smiles')
    final_df[rdkit_cols] = final_df[rdkit_cols].fillna(0)

    # 4. STATYSTYKI (Tylko informacyjne)
    success_mopac = (
            (final_df['mask_1'] == 1) &
            (final_df['mask_2'] == 1) &
            (final_df['mask_3'] == 1) &
            (final_df['mask_4'] == 1)
    ).sum()

    # Zapisujemy wszystko
    final_df.to_csv(FINAL_OUTPUT, index=False)

    #df = pd.read_csv("dataset.csv")
    df.to_parquet("dataset.parquet", compression='brotli')

    print("\n" + "=" * 45)
    print("PODSUMOWANIE GENEROWANIA ZBIORU FULL:")
    print("=" * 45)
    print(f"Liczba wejściowych rekordów TDC:  {initial_count}")
    print(f"Liczba zapisanych rekordów:      {len(final_df)}")
    print("-" * 45)
    print(f"Rekordy z poprawnym MOPAC:       {success_mopac}")
    print(f"Rekordy z błędami MOPAC/Brakami: {len(final_df) - success_mopac}")
    print("-" * 45)
    print(f"Wszystkie dane zapisano w: {FINAL_OUTPUT}")
    print("=" * 45)

    return final_df


if __name__ == "__main__":
    df = merge_everything_including_failures()