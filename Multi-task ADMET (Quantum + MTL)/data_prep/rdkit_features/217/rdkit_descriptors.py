import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- KONFIGURACJA ---
N_CORES = 8
INPUT_FILE = "../unique_smiles_to_calculate.csv"
OUTPUT_RDKIT = "rdkit_descriptors.csv"


def get_rdkit_descriptors(smiles):
    """Oblicza pełny zestaw deskryptorów RDKit dla pojedynczej cząsteczki."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "success": False}

        # Inicjalizacja kalkulatora wszystkich dostępnych deskryptorów
        nms = [x[0] for x in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

        ds = calc.CalcDescriptors(mol)
        res = {"smiles": smiles, "success": True}
        for name, value in zip(nms, ds):
            res[name] = value
        return res
    except:
        return {"smiles": smiles, "success": False}


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    smiles_list = df['smiles'].tolist()

    print(f"Obliczanie deskryptorów RDKit dla {len(smiles_list)} cząsteczek...")

    results = []
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(get_rdkit_descriptors, sm): sm for sm in smiles_list}

        for future in tqdm(as_completed(futures), total=len(smiles_list), desc="RDKit Features"):
            results.append(future.result())

    # Tworzenie DataFrame i zapis
    df_res = pd.DataFrame(results)
    # Usuwamy te, które się nie policzyły
    df_res = df_res[df_res['success'] == True].drop(columns=['success'])

    df_res.to_csv(OUTPUT_RDKIT, index=False)
    print(f"Gotowe! Deskryptory zapisane w: {OUTPUT_RDKIT}")