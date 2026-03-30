import pandas as pd

TASKS_RAW_FILE = "tdc_multitask_raw.parquet"
QUANTUM_FILE = "tdc_final_quantum_full.parquet"
RDKIT_FILE = "rdkit_descriptors.parquet"

FINAL_PARQUET = "dataset.parquet"
FINAL_CSV = "dataset.csv"


def merge_all():
    print("--- MERGE ---")

    # =========================
    # LOAD
    # =========================
    df_tasks = pd.read_parquet(TASKS_RAW_FILE)
    df_quantum = pd.read_parquet(QUANTUM_FILE)
    df_rdkit = pd.read_parquet(RDKIT_FILE)

    initial_count = len(df_tasks)

    # =========================
    # MERGE
    # =========================
    merged = pd.merge(df_tasks, df_quantum, on="smiles", how="left")
    final_df = pd.merge(merged, df_rdkit, on="smiles", how="left")

    # =========================
    # STATS (jak wcześniej)
    # =========================
    total = len(final_df)

    if all(col in final_df.columns for col in ["mask_1", "mask_2", "mask_3", "mask_4"]):
        ok_all = (
            (final_df["mask_1"] == 1) &
            (final_df["mask_2"] == 1) &
            (final_df["mask_3"] == 1) &
            (final_df["mask_4"] == 1)
        ).sum()
    else:
        ok_all = 0

    print("\n=== GLOBAL ===")
    print(f"TDC input: {initial_count}")
    print(f"FINAL rows: {total}")
    print(f"ALL OK: {ok_all}")
    print(f"ALL OK %: {100 * ok_all / total:.2f}" if total > 0 else "0")

    if "mask_1" in final_df.columns:
        print("\n=== PER PROPERTY ===")
        for m_col, name in [
            ("mask_1", "dipole"),
            ("mask_2", "homo_lumo"),
            ("mask_3", "electrons"),
            ("mask_4", "energy")
        ]:
            if m_col in final_df.columns:
                ok = (final_df[m_col] == 1).sum()
                print(f"{name}: OK={ok}, FAIL={total - ok}, %={100 * ok / total:.2f}")

    # =========================
    # SAVE
    # =========================
    final_df.to_parquet(
        FINAL_PARQUET,
        engine="pyarrow",
        compression="snappy"
    )

    final_df.to_csv(
        FINAL_CSV,
        index=False,
        float_format="%.6f"
    )

    print(f"\nDONE → {FINAL_PARQUET} + {FINAL_CSV}")

    return final_df


if __name__ == "__main__":
    df = merge_all()