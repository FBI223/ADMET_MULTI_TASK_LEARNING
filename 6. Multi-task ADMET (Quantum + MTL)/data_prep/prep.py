import pandas as pd
import numpy as np
from tdc.single_pred import ADME, Tox
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from xtb.interface import Calculator
from xtb.utils import get_method
from multiprocessing import Pool
import os
import sys
from tqdm import tqdm

# --- KONFIGURACJA ---
N_CORES = 16  # Wykorzystujemy pełną moc Twojego RunPoda
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX

# Kalkulator RDKit (200 cech)
DESC_NAMES = [d[0] for d in Descriptors._descList][:200]
RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def download_tdc_data():
    """Pobiera dane zachowując niezależne splity scaffold dla każdego zadania."""
    print("KROK 1: Pobieranie danych z TDC (Scaffold Split)...")
    all_data = []
    for name in ALL_TASKS:
        group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)
        split = group.get_split()  # Pobiera oficjalny Scaffold Split
        for s_name in ["train", "valid", "test"]:
            df = split[s_name][['Drug', 'Y']].copy()
            df.columns = ['smiles', 'label']
            df['task'] = name
            df['split'] = s_name
            all_data.append(df)

    final_raw = pd.concat(all_data, ignore_index=True)
    return final_raw


def compute_rdkit_worker(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return list(RDKIT_CALC.CalcDescriptors(mol)) if mol else [0.0] * 200
    except:
        return [0.0] * 200


def compute_quantum_worker(smiles):
    """Zoptymalizowane obliczenia QC z maską binarną."""
    mask = [1, 1, 1, 1]
    feats = [0.0, 0.0, 0.0, 0.0]
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        # Szybki embedding 3D zgodny z artykułem
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=1) == 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=100)
            coords = np.array([list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

            calc = Calculator(get_method("GFN2-xTB"), Z, coords)
            calc.set_verbosity(0)  # Kluczowe: wyciszenie logów xTB
            res = calc.singlepoint()

            feats = [
                np.linalg.norm(res.get_dipole()),  # Dipole moment
                res.get_lumo() - res.get_homo(),  # HOMO-LUMO gap
                float(Z.sum()),  # Electrons
                res.get_energy()  # Total energy
            ]
    except:
        mask = [0, 0, 0, 0]
    return feats + mask


def main():
    # 1. Pobieranie danych
    full_df = download_tdc_data()
    unique_smiles = full_df['smiles'].unique()
    print(f"Liczba unikalnych cząsteczek: {len(unique_smiles)}")

    # 2. RDKit (Bardzo szybko)
    print("\nKROK 2: Obliczanie 200 cech RDKit...")
    with Pool(N_CORES) as p:
        r_res = list(tqdm(p.imap(compute_rdkit_worker, unique_smiles, chunksize=50), total=len(unique_smiles)))

    rdkit_df = pd.DataFrame(r_res, columns=[f'rdkit_{i}' for i in range(200)])
    rdkit_df['smiles'] = unique_smiles

    # 3. Quantum QC (Najwolniej - chunksize=1 dla płynnego progresu)
    print("\nKROK 3: Obliczanie cech kwantowych (QC)...")
    with Pool(N_CORES) as p:
        # chunksize=1 sprawia, że pasek tqdm rusza się od razu po przeliczeniu jednej cząsteczki
        q_res = list(tqdm(p.imap(compute_quantum_worker, unique_smiles, chunksize=1), total=len(unique_smiles)))

    q_cols = ['q_dipole', 'q_gap', 'q_electrons', 'q_energy', 'm_dipole', 'm_gap', 'm_electrons', 'm_energy']
    q_df = pd.DataFrame(q_res, columns=q_cols)
    q_df['smiles'] = unique_smiles

    # 4. Finalne scalanie w Long Format (Zgodność z Scaffold Split)
    print("\nKROK 4: Scalanie danych i zapis...")
    features = pd.merge(rdkit_df, q_df, on='smiles')
    final_dataset = pd.merge(full_df, features, on='smiles', how='left')

    final_dataset.to_csv(f"{DATA_DIR}/qwmtl_final_dataset.csv", index=False)
    print(f"SUKCES! Plik gotowy: {final_dataset.shape}")


if __name__ == "__main__":
    main()