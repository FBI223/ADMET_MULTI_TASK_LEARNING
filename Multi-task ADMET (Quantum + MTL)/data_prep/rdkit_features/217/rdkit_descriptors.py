import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- KONFIG ---
N_CORES = 8
INPUT_FILE = "unique_smiles_to_calculate.parquet"
OUTPUT_RDKIT = "rdkit_descriptors.parquet"


# 🔥 INIT GLOBAL (ważne dla speed)
DESC_NAMES = [x[0] for x in Descriptors._descList]
CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def get_rdkit_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "success": False}

        ds = CALC.CalcDescriptors(mol)

        res = {"smiles": smiles, "success": True}
        for name, value in zip(DESC_NAMES, ds):
            res[name] = value

        return res

    except:
        return {"smiles": smiles, "success": False}


if __name__ == "__main__":
    # =========================
    # LOAD (PARQUET)
    # =========================
    df = pd.read_parquet(INPUT_FILE)
    smiles_list = df["smiles"].tolist()

    print(f"RDKit dla {len(smiles_list)}")

    results = []
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(get_rdkit_descriptors, sm) for sm in smiles_list]

        for f in tqdm(as_completed(futures), total=len(futures), desc="RDKit"):
            results.append(f.result())

    # =========================
    # DATAFRAME
    # =========================
    df_res = pd.DataFrame(results)

    # NIE USUWAMY (zgodnie z Twoim stylem)
    # ale możesz zostawić success jako maskę
    df_res["success"] = df_res["success"].fillna(False).astype("int8")

    # =========================
    # SAVE (PARQUET)
    # =========================
    df_res.to_parquet(
        OUTPUT_RDKIT,
        engine="pyarrow",
        compression="snappy"
    )

    # opcjonalnie CSV
    df_res.to_csv(
        OUTPUT_RDKIT.replace(".parquet", ".csv"),
        index=False,
        float_format="%.6f"
    )

    print(f"DONE → {OUTPUT_RDKIT}")