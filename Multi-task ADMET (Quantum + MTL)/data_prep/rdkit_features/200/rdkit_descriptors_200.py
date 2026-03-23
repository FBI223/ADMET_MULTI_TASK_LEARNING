import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- KONFIGURACJA ---
N_CORES = 8
INPUT_FILE = "../unique_smiles_to_calculate.csv"
OUTPUT_RDKIT = "rdkit_descriptors.csv"

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

# globalny kalkulator (tylko dozwolone deskryptory)
ALL_NAMES = [x[0] for x in Descriptors._descList]
FILTERED_NAMES = [n for n in ALL_NAMES if n not in EXCLUDE]
CALC = MoleculeDescriptors.MolecularDescriptorCalculator(FILTERED_NAMES)


def get_rdkit_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "success": False}

        values = CALC.CalcDescriptors(mol)

        res = {"smiles": smiles, "success": True}
        for name, val in zip(FILTERED_NAMES, values):
            res[name] = val

        return res
    except:
        return {"smiles": smiles, "success": False}


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    smiles_list = df["smiles"].tolist()

    print(f"Obliczanie deskryptorów RDKit dla {len(smiles_list)} cząsteczek...")
    print(f"Liczba deskryptorów po filtracji: {len(FILTERED_NAMES)}")  # powinno być 200

    results = []
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(get_rdkit_descriptors, sm): sm for sm in smiles_list}

        for future in tqdm(as_completed(futures), total=len(smiles_list), desc="RDKit Features"):
            results.append(future.result())

    df_res = pd.DataFrame(results)
    df_res = df_res[df_res["success"] == True].drop(columns=["success"])

    df_res.to_csv(OUTPUT_RDKIT, index=False)

    print(f"Gotowe! Deskryptory zapisane w: {OUTPUT_RDKIT}")