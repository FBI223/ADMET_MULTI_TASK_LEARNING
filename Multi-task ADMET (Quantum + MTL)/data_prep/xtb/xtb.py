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

os.environ["OMP_NUM_THREADS"] = "1"

BATCH_SIZE = 100
CHECKPOINT_FILE = "tdc_quantum_checkpoint.csv"
INPUT_FILE = "unique_smiles_to_calculate.parquet"
FINAL_FULL = "tdc_final_quantum_full.parquet"



# =========================
# SMILES → XYZ
# =========================
def smiles_to_xyz(smiles, fname):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        return False

    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    with open(fname, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

    return True


# =========================
# PARSER
# =========================
def parse_xtb(output):
    res = {"dipole": None, "homo_lumo": None, "electrons": None, "energy": None}
    mask = [0, 0, 0, 0]

    lines = output.splitlines()

    for i, line in enumerate(lines):
        l = line.lower()

        if "total energy" in l and "eh" in l:
            res["energy"] = float(re.findall(r"-?\d+\.\d+", line)[0])
            mask[3] = 1

        if "homo-lumo gap" in l:
            res["homo_lumo"] = float(re.findall(r"-?\d+\.\d+", line)[0])
            mask[1] = 1

        if "# electrons" in l:
            res["electrons"] = int(re.findall(r"\d+", line)[0])
            mask[2] = 1

        if "molecular dipole" in l and i + 2 < len(lines):
            nums = re.findall(r"-?\d+\.\d+", lines[i + 2])
            if len(nums) >= 3:
                x, y, z = map(float, nums[:3])
                res["dipole"] = (x**2 + y**2 + z**2) ** 0.5
                mask[0] = 1

    return res, mask


# =========================
# WORKER (XTB)
# =========================
def extract_worker(smiles, idx):
    output_template = {
        "smiles": smiles,
        "dipole": None,
        "homo_lumo": None,
        "electrons": None,
        "energy": None,
        "mask_1": 0,
        "mask_2": 0,
        "mask_3": 0,
        "mask_4": 0
    }

    tmp_dir = tempfile.mkdtemp(prefix=f"xtb_{idx}_")
    xyz_file = os.path.join(tmp_dir, "mol.xyz")

    try:
        ok = smiles_to_xyz(smiles, xyz_file)
        if not ok:
            return output_template

        # --- OPT ---
        subprocess.run(
            ["xtb", "mol.xyz", "--gfn", "2", "--opt"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=tmp_dir
        )

        # --- SP ---
        result = subprocess.run(
            ["xtb", "xtbopt.xyz", "--gfn", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=tmp_dir
        )

        output = result.stdout

        vals, m = parse_xtb(output)

        output_template.update(vals)
        output_template.update({f"mask_{i + 1}": m[i] for i in range(4)})

    except Exception as e:
        pass

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_template


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    all_results = []

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"Start: {len(df)} cząsteczek")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_worker, sm, i): i for i, sm in enumerate(df["smiles"])}

        with tqdm(total=len(df), desc="XTB") as pbar:
            current_batch = []

            for future in as_completed(futures):
                res = future.result()
                all_results.append(res)
                current_batch.append(res)

                if len(current_batch) >= BATCH_SIZE:
                    pd.DataFrame(current_batch).to_csv(
                        CHECKPOINT_FILE,
                        mode='a',
                        index=False,
                        header=not os.path.exists(CHECKPOINT_FILE)
                    )
                    current_batch = []

                pbar.update(1)

            if current_batch:
                pd.DataFrame(current_batch).to_csv(
                    CHECKPOINT_FILE,
                    mode='a',
                    index=False,
                    header=not os.path.exists(CHECKPOINT_FILE)
                )

    df_out = pd.DataFrame(all_results)

    # csv (backup / debug)
    df_out.to_csv(
        FINAL_FULL.replace(".parquet", ".csv"),
        index=False
    )

    # =========================
    # CAST TYPES
    # =========================
    #df_out["dipole"] = pd.to_numeric(df_out["dipole"], errors="coerce").astype("float32")
    #df_out["homo_lumo"] = pd.to_numeric(df_out["homo_lumo"], errors="coerce").astype("float32")
    #df_out["energy"] = pd.to_numeric(df_out["energy"], errors="coerce").astype("float32")
    #df_out["electrons"] = pd.to_numeric(df_out["electrons"], errors="coerce").astype("Int32")

    # =========================
    # STATS
    # =========================
    total = len(df_out)
    ok_all = ((df_out["mask_1"] == 1) &
              (df_out["mask_2"] == 1) &
              (df_out["mask_3"] == 1) &
              (df_out["mask_4"] == 1)).sum()

    print("\n=== GLOBAL ===")
    print(f"TOTAL: {total}")
    print(f"ALL OK: {ok_all}")
    print(f"ALL OK %: {100 * ok_all / total:.2f}")

    print("\n=== PER PROPERTY ===")
    for m_col, name in [
        ("mask_1", "dipole"),
        ("mask_2", "homo_lumo"),
        ("mask_3", "electrons"),
        ("mask_4", "energy")
    ]:
        ok = df_out[m_col].sum()
        print(f"{name}: OK={ok}, FAIL={total - ok}, %={100 * ok / total:.2f}")

    # =========================
    # SAVE
    # =========================
    df_out.to_parquet(
        FINAL_FULL,
        engine="pyarrow",
        compression="snappy"
    )


    print(f"\nDONE → {FINAL_FULL}")