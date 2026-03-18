import pandas as pd
import numpy as np
import os
import time
from tdc.single_pred import ADME, Tox
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from xtb.interface import Calculator
from xtb.utils import get_method
from multiprocessing import Pool

# --- KONFIGURACJA ---
N_CORES = 8
TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX

# Pre-inicjalizacja nazw deskryptorów RDKit (200 sztuk zgodnie z QW-MTL )
DESC_NAMES = [d[0] for d in Descriptors._descList][:200]
RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


# --- FUNKCJE WORKER (MUSZĄ BYĆ NA POZIOMIE MODUŁU) ---

def rdkit_worker(s):
    """Oblicza 200 deskryptorów RDKit."""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return [0.0] * 200
    try:
        return list(RDKIT_CALC.CalcDescriptors(mol))
    except:
        return [0.0] * 200


def quantum_worker(s):
    """Oblicza 4 cechy kwantowe + 4 bity maski."""
    res_data = {
        'q_dipole': 0.0, 'q_gap': 0.0, 'q_elec': 0.0, 'q_energy': 0.0,
        'm_dipole': 0, 'm_gap': 0, 'm_elec': 0, 'm_energy': 0
    }
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(s))
        if AllChem.EmbedMolecule(mol, randomSeed=42) == 0:
            AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

            calc = Calculator(get_method("GFN2-xTB"), Z, coords)
            sp = calc.singlepoint()

            res_data.update({
                'q_dipole': np.linalg.norm(sp.get_dipole()),
                'q_gap': sp.get_lumo() - sp.get_homo(),
                'q_elec': float(Z.sum()),
                'q_energy': sp.get_energy(),
                'm_dipole': 1, 'm_gap': 1, 'm_elec': 1, 'm_energy': 1
            })
    except:
        pass
    return res_data


# --- GŁÓWNE FUNKCJE PRZETWARZANIA ---

def prepare_base_data():
    print("Pobieranie danych TDC i zachowywanie splitów...")
    all_rows = []
    os.makedirs('data/raw_tasks', exist_ok=True)

    for name in ALL_TASKS:
        group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)
        split = group.get_split()
        for s_name in ["train", "valid", "test"]:
            df = split[s_name][['Drug', 'Y']].copy()
            df.columns = ['smiles', 'label']
            df['task_name'] = name
            df['split_type'] = s_name
            df.to_csv(f'data/raw_tasks/{name}_{s_name}.csv', index=False)
            all_rows.append(df)

    full_metadata = pd.concat(all_rows, ignore_index=True)
    full_metadata.to_csv("data/master_metadata.csv", index=False)
    return full_metadata['smiles'].unique()


def main():
    os.makedirs('data/features', exist_ok=True)

    # 1. Pobranie SMILES
    unique_smiles = prepare_base_data()

    # 2. RDKit Batch
    print(f"Obliczanie 200 deskryptorów RDKit dla {len(unique_smiles)} cząsteczek...")
    with Pool(N_CORES) as p:
        rdkit_results = p.map(rdkit_worker, unique_smiles)

    rdkit_df = pd.DataFrame(rdkit_results, columns=[f'rd_feat_{i}' for i in range(200)])
    rdkit_df['smiles'] = unique_smiles
    rdkit_df.to_parquet("data/features/rdkit_200.parquet", index=False)

    # 3. Quantum Batch
    print(f"Obliczanie cech QC (GFN2-xTB) dla {len(unique_smiles)} cząsteczek[cite: 130]...")
    with Pool(N_CORES) as p:
        qc_results = p.map(quantum_worker, unique_smiles)

    qc_df = pd.DataFrame(qc_results)
    qc_df['smiles'] = unique_smiles
    qc_df.to_parquet("data/features/quantum_physical.parquet", index=False)
    print("Pipeline zakończony sukcesem!")


if __name__ == "__main__":
    main()