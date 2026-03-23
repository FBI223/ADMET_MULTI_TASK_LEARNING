import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- KONFIG ---
N_CORES = 8
INPUT_FILE = "../unique_smiles_to_calculate.parquet"
OUTPUT_RDKIT = "rdkit_descriptors.parquet"

# deskryptory do wykluczenia
EXCLUDE = {
    "AvgIpc",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "NumAmideBonds",
    "NumAtomStereoCenters",
    "NumBridgeheadAtoms",
    "NumHeterocycles",
    "NumSpiroAtoms",
    "NumUnspecifiedAtomStereoCenters",
    "Phi",
    "SPS",
}

# globalny kalkulator
ALL_NAMES = [x[0] for x in Descriptors._descList]
FILTERED_NAMES = [n for n in ALL_NAMES if n not in EXCLUDE]
CALC = MoleculeDescriptors.MolecularDescriptorCalculator(FILTERED_NAMES)


def get_rdkit_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "success": 0}

        values = CALC.CalcDescriptors(mol)

        res = {"smiles": smiles, "success": 1}
        for name, val in zip(FILTERED_NAMES, values):
            res[name] = val

        return res

    except:
        return {"smiles": smiles, "success": 0}


if __name__ == "__main__":
    # =========================
    # LOAD (PARQUET)
    # =========================
    df = pd.read_parquet(INPUT_FILE)
    smiles_list = df["smiles"].tolist()

    print(f"RDKit: {len(smiles_list)} cząsteczek")
    print(f"Deskryptory: {len(FILTERED_NAMES)}")

    results = []
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(get_rdkit_descriptors, sm) for sm in smiles_list]

        for f in tqdm(as_completed(futures), total=len(futures), desc="RDKit"):
            results.append(f.result())

    # =========================
    # DF
    # =========================
    df_res = pd.DataFrame(results)

    # zachowujemy wszystko (jak chcesz), ale możesz filtrować:
    # df_res = df_res[df_res["success"] == 1].drop(columns=["success"])

    # =========================
    # SAVE
    # =========================
    df_res.to_parquet(
        OUTPUT_RDKIT,
        engine="pyarrow",
        compression="snappy"
    )

    df_res.to_csv(
        OUTPUT_RDKIT.replace(".parquet", ".csv"),
        index=False,
        float_format="%.6f"
    )

    print(f"DONE → {OUTPUT_RDKIT}")