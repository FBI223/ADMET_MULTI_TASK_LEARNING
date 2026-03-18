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


N_CORES = 8
TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX

# Tworzenie struktury folderów
DIRS = ['data/raw_tasks', 'data/features', 'data/splits']
for d in DIRS: os.makedirs(d, exist_ok=True)


# --- 1. POBIERANIE I ZACHOWANIE NIEZALEŻNYCH WIERSZY  ---
def prepare_base_data():
    print("Pobieranie danych TDC i zachowywanie niezależnych wierszy...")
    all_rows = []

    for name in ALL_TASKS:
        group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)
        # TDC domyślnie używa scaffold split [cite: 163]
        split = group.get_split()

        for s_name in ["train", "valid", "test"]:
            df = split[s_name][['Drug', 'Y']].copy()
            df.columns = ['smiles', 'label']
            df['task_name'] = name
            df['split_type'] = s_name

            # Zapisujemy każde zadanie osobno dla przejrzystości
            task_path = f'data/raw_tasks/{name}_{s_name}.csv'
            df.to_csv(task_path, index=False)
            all_rows.append(df)

    full_metadata = pd.concat(all_rows, ignore_index=True)
    full_metadata.to_csv("data/master_metadata.csv", index=False)
    return full_metadata['smiles'].unique()


# --- 2. DESKRYPTORY RDKit (DOKŁADNIE 200 CECH) ---
def compute_rdkit_batch(smiles_list):
    print(f"Obliczanie 200 deskryptorów RDKit dla {len(smiles_list)} unikalnych cząsteczek...")
    desc_names = [d[0] for d in Descriptors._descList][:200]  # [cite: 139]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    def worker(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None: return [0.0] * 200
        try:
            return list(calc.CalcDescriptors(mol))
        except:
            return [0.0] * 200

    with Pool(N_CORES) as p:
        results = p.map(worker, smiles_list)

    rdkit_df = pd.DataFrame(results, columns=[f'rd_feat_{i}' for i in range(200)])
    rdkit_df['smiles'] = smiles_list
    rdkit_df.to_parquet("data/features/rdkit_200.parquet", index=False)


# --- 3. WYSOKIEJ JAKOŚCI CECHY KWANTOWE (GFN2-xTB)  ---
def compute_quantum_batch(smiles_list):
    print(f"Obliczanie cech kwantowych (QC) dla {len(smiles_list)} cząsteczek...")

    def worker(s):
        # [cite: 133]: dipole, gap, electrons, total energy
        # [cite: 138]: 4-dimensional binary mask
        res_data = {'smiles': s, 'q_dipole': 0.0, 'q_gap': 0.0, 'q_elec': 0.0, 'q_energy': 0.0,
                    'm_dipole': 0, 'm_gap': 0, 'm_elec': 0, 'm_energy': 0}
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

    start_time = time.time()
    with Pool(N_CORES) as p:
        results = p.map(worker, smiles_list)

    qc_df = pd.DataFrame(results)
    qc_df.to_parquet("data/features/quantum_physical.parquet", index=False)
    print(f"Zakończono QC w czasie: {time.time() - start_time:.2f}s")


# --- 4. GŁÓWNY PROCES ---
if __name__ == "__main__":
    # Krok 1: Unikalne SMILES z zachowaniem struktury zadań
    unique_smiles = prepare_base_data()

    # Krok 2: RDKit 200D (Zapis do Parquet dla szybkości/rozmiaru)
    compute_rdkit_batch(unique_smiles)

    # Krok 3: Quantum 4D + Mask 4D
    compute_quantum_batch(unique_smiles)

    print("\nPipeline zakończony pomyślnie.")
    print("Struktura 'data/' jest gotowa do ładowania przez model QW-MTL.")