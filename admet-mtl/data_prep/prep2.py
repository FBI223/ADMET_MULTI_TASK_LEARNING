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
N_CORES = 8
# Zwiększamy chunksize, aby rdzenie nie czekały na nowe dane
CHUNKSIZE = 16

TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX

# Globalny kalkulator RDKit (200 cech) [cite: 139]
DESC_NAMES = [d[0] for d in Descriptors._descList][:200]
RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def download_tdc_data():
    """Pobiera dane zachowując niezależne wiersze (Independent Rows)[cite: 169]."""
    print("KROK 1: Pobieranie danych z TDC (Scaffold Split)...")
    all_data = []
    for name in ALL_TASKS:
        group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)
        split = group.get_split()  # Oficjalny podział TDC [cite: 163]
        for s_name in ["train", "valid", "test"]:
            df = split[s_name][['Drug', 'Y']].copy()
            df.columns = ['smiles', 'label']
            df['task'] = name
            df['split'] = s_name
            all_data.append(df)

    final_raw = pd.concat(all_data, ignore_index=True)
    final_raw.to_csv("data/raw_tdc_data.csv", index=False)
    return final_raw


def compute_rdkit_worker(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [0.0] * 200
        return list(RDKIT_CALC.CalcDescriptors(mol))
    except:
        return [0.0] * 200


def compute_quantum_worker(smiles):
    """Zoptymalizowana ekstrakcja 4 cech QC[cite: 133]."""
    mask = [1, 1, 1, 1]
    features = [0.0, 0.0, 0.0, 0.0]
    try:
        # Szybsza metoda generowania konformera 3D (ETKDGv3)
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 1  # Szybsza próba dla dużych zbiorów

        if AllChem.EmbedMolecule(mol, params) == 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=100)  # Ograniczona liczba iteracji

            coords = np.array([list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

            # Obliczenia GFN2-xTB
            calc = Calculator(get_method("GFN2-xTB"), Z, coords)
            calc.set_verbosity(0)  # Wyciszenie xTB bez używania dup2
            res = calc.singlepoint()

            features = [
                np.linalg.norm(res.get_dipole()),  # dipole moment [cite: 133]
                res.get_lumo() - res.get_homo(),  # HOMO-LUMO gap [cite: 133]
                float(Z.sum()),  # electrons [cite: 133]
                res.get_energy()  # total energy [cite: 133]
            ]
    except:
        mask = [0, 0, 0, 0]  # Maska binarna dla spójności [cite: 138]
    return features + mask


def main():
    os.makedirs('data', exist_ok=True)
    full_obs_df = download_tdc_data()
    unique_smiles = full_obs_df['smiles'].unique()

    # RDKit 200D
    print(f"\nKROK 2: RDKit (200D) dla {len(unique_smiles)} cząsteczek...")
    with Pool(N_CORES) as p:
        rdkit_res = list(
            tqdm(p.imap(compute_rdkit_worker, unique_smiles, chunksize=CHUNKSIZE), total=len(unique_smiles)))
    pd.DataFrame(rdkit_res, columns=[f'rdkit_{i}' for i in range(200)]).to_csv("data/rdkit_features.csv", index=False)

    # Quantum QC (To zawsze będzie najwolniejszy etap)
    print(f"\nKROK 3: Quantum QC dla {len(unique_smiles)} cząsteczek...")
    with Pool(N_CORES) as p:
        q_res = list(tqdm(p.imap(compute_quantum_worker, unique_smiles, chunksize=CHUNKSIZE), total=len(unique_smiles)))

    q_cols = ['q_dipole', 'q_gap', 'q_electrons', 'q_energy', 'm_dipole', 'm_gap', 'm_electrons', 'm_energy']
    q_df = pd.DataFrame(q_res, columns=q_cols)
    q_df['smiles'] = unique_smiles
    q_df.to_csv("data/quantum_features.csv", index=False)

    # Finalne scalanie (Long Format) [cite: 169]
    print("\nKROK 4: Scalanie do formatu Long...")
    rdkit_df = pd.read_csv("data/rdkit_features.csv")
    rdkit_df['smiles'] = unique_smiles
    final_df = full_obs_df.merge(rdkit_df, on='smiles').merge(q_df, on='smiles')
    final_df.to_csv("data/qwmtl_final_dataset.csv", index=False)
    print(f"Sukces! Plik gotowy: {final_df.shape}")


if __name__ == "__main__":
    main()