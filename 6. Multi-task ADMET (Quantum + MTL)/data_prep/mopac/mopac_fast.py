import pandas as pd
import os
import subprocess
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Optymalizacja pod klaster/wielordzeniowość
os.environ["MOPAC_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# --- KONFIGURACJA ---
BATCH_SIZE = 100  # Co ile cząsteczek zapisywać checkpoint
CHECKPOINT_FILE =  "tdc_quantum_checkpoint.csv"
#INPUT_FILE = "tdc_multitask_raw.csv"
INPUT_FILE = "unique_smiles_to_calculate.csv"
FINAL_FULL = "tdc_final_quantum_full.csv"
FINAL_CLEAN = "tdc_final_quantum_clean.csv"


def parse_mopac_output(content):
    res = {"dipole": None, "homo_lumo": None, "electrons": None, "energy": None}
    mask = [0, 0, 0, 0]
    if "JOB ENDED NORMALLY" not in content:
        return res, mask

    e_match = re.search(r"FINAL HEAT OF FORMATION\s+=\s+(-?\d+\.\d+)", content)
    if e_match:
        res["energy"] = float(e_match.group(1))
        mask[3] = 1

    d_match = re.search(r"SUM\s+[-\d\.]+\s+[-\d\.]+\s+[-\d\.]+\s+([\d\.]+)", content)
    if d_match:
        val = float(d_match.group(1))
        if val < 20:
            res["dipole"] = val
            mask[0] = 1

    gap_match = re.search(r"HOMO LUMO ENERGIES.*?(-?\d+\.\d+)\s+(-?\d+\.\d+)", content, re.DOTALL)
    if gap_match:
        homo, lumo = float(gap_match.group(1)), float(gap_match.group(2))
        gap = abs(lumo - homo)
        if gap < 20:
            res["homo_lumo"] = gap
            mask[1] = 1

    el_match = re.search(r"NO\. OF FILLED LEVELS\s+=\s+(\d+)", content)
    if el_match:
        res["electrons"] = int(el_match.group(1)) * 2
        mask[2] = 1

    return res, mask


def extract_worker(smiles, idx):
    """Funkcja pomocnicza dla pojedynczego procesu."""
    output_template = {
        "smiles": smiles, "dipole": None, "homo_lumo": None, "electrons": None, "energy": None,
        "mask_1": 0, "mask_2": 0, "mask_3": 0, "mask_4": 0
    }

    tmp_dir = tempfile.mkdtemp(prefix=f"mop_{idx}_")
    m_file = os.path.join(tmp_dir, "mol.mop")
    o_file = os.path.join(tmp_dir, "mol.out")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == 0:
                AllChem.UFFOptimizeMolecule(mol)
                xyz = Chem.MolToXYZBlock(mol).split('\n', 2)[2]

                with open(m_file, "w") as f:
                    f.write("PM7 OPT PRECISE CHARGE=0 DIPOLE\nQW-MTL\n\n" + xyz)

                subprocess.run(["mopac", m_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_dir)

                if os.path.exists(o_file):
                    with open(o_file, "r") as f:
                        content = f.read()
                    if "SCF FIELD WAS ACHIEVED" in content:
                        vals, m = parse_mopac_output(content)
                        output_template.update(vals)
                        output_template.update({f"mask_{i + 1}": m[i] for i in range(4)})
    except:
        pass
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_template


# --- MAIN ---
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    all_results = []

    # Usuwamy stary checkpoint jeśli istnieje
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"Start: {len(df)} cząsteczek. Batch size: {BATCH_SIZE}")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_worker, sm, i): i for i, sm in enumerate(df["smiles"])}

        with tqdm(total=len(df), desc="Obliczenia MOPAC") as pbar:
            current_batch = []
            for future in as_completed(futures):
                res = future.result()
                all_results.append(res)
                current_batch.append(res)

                # Batch save (Checkpoint)
                if len(current_batch) >= BATCH_SIZE:
                    pd.DataFrame(current_batch).to_csv(CHECKPOINT_FILE, mode='a', index=False,
                                                       header=not os.path.exists(CHECKPOINT_FILE))
                    current_batch = []

                pbar.update(1)

            # Zapisz resztkę z ostatniego batcha
            if current_batch:
                pd.DataFrame(current_batch).to_csv(CHECKPOINT_FILE, mode='a', index=False,
                                                   header=not os.path.exists(CHECKPOINT_FILE))

    df_out = pd.DataFrame(all_results)

    # --- SEKCA STATYSTYK (Twoja oryginalna logika) ---
    total = len(df_out)
    ok_all = ((df_out["mask_1"] == 1) & (df_out["mask_2"] == 1) &
              (df_out["mask_3"] == 1) & (df_out["mask_4"] == 1)).sum()

    print("\n=== GLOBAL ===")
    print(f"TOTAL: {total}")
    print(f"ALL OK: {ok_all}")
    print(f"ALL OK %: {100 * ok_all / total:.2f}%" if total > 0 else "0%")

    print("\n=== PER PROPERTY ===")
    for m_col, name in [("mask_1", "dipole"), ("mask_2", "homo_lumo"), ("mask_3", "electrons"), ("mask_4", "energy")]:
        ok = df_out[m_col].sum()
        print(f"{name}: OK={ok}, FAIL={total - ok}, %={100 * ok / total:.2f}")

    print("\n=== MISSING / ZERO VALUES ===")
    for col in ["dipole", "homo_lumo", "electrons", "energy"]:
        m = df_out[col].isna().sum()
        z = (df_out[col] == 0).sum()
        print(f"{col}: missing={m} ({100 * m / total:.1f}%), zero={z} ({100 * z / total:.1f}%)")

    # --- FINAL SAVE ---
    df_out.to_csv(FINAL_FULL, index=False)
    df_clean = df_out[(df_out["mask_1"] == 1) & (df_out["mask_2"] == 1) &
                      (df_out["mask_3"] == 1) & (df_out["mask_4"] == 1)]
    df_clean.to_csv(FINAL_CLEAN, index=False)

    print(f"\nGotowe! Wyniki zapisane w {FINAL_FULL} oraz {FINAL_CLEAN}")