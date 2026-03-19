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
import contextlib

# --- KONFIGURACJA ---
N_CORES = 8
TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX

# OPTYMALIZACJA: Inicjalizacja kalkulatora raz na poziomie modułu
DESC_NAMES = [d[0] for d in Descriptors._descList][:200]
RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


@contextlib.contextmanager
def mute_outputs():
    """Wycisza logi xTB na poziomie deskryptorów plików (C++)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def download_tdc_data():
    """Pobiera dane zachowując niezależne wiersze i oficjalne splity[cite: 163, 169]."""
    print("KROK 1: Pobieranie danych z TDC (Scaffold Split)...")
    all_data = []
    for name in ALL_TASKS:
        group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)
        split = group.get_split()  # Pobranie oficjalnego podziału [cite: 163]
        for s_name in ["train", "valid", "test"]:
            df = split[s_name][['Drug', 'Y']].copy()
            df.columns = ['smiles', 'label']
            df['task'] = name
            df['split'] = s_name
            all_data.append(df)

    final_raw = pd.concat(all_data, ignore_index=True)
    # Zapis punktu kontrolnego 1 [cite: 168]
    final_raw.to_csv("data/raw_tdc_data.csv", index=False)
    return final_raw


def compute_rdkit_worker(smiles):
    """Przetwarza deskryptory RDKit (200 cech)[cite: 139]."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [0.0] * 200
        return list(RDKIT_CALC.CalcDescriptors(mol))
    except:
        return [0.0] * 200


def compute_quantum_worker(smiles):
    """Ekstrakcja 4 cech QC + 4 bity maski binarnej."""
    mask = [1, 1, 1, 1]
    features = [0.0, 0.0, 0.0, 0.0]
    try:
        with mute_outputs():
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                raise ValueError("Embedding failed")
            AllChem.UFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

            # Obliczenia GFN2-xTB [cite: 114, 130]
            calc = Calculator(get_method("GFN2-xTB"), Z, coords)
            res = calc.singlepoint()

            features = [
                np.linalg.norm(res.get_dipole()),  # dipole moment norm [cite: 133]
                res.get_lumo() - res.get_homo(),  # HOMO-LUMO gap [cite: 133]
                float(Z.sum()),  # total electrons [cite: 133]
                res.get_energy()  # total energy [cite: 133]
            ]
    except:
        mask = [0, 0, 0, 0]  # Maska binarna dla niepowodzeń [cite: 138]
    return features + mask


def main():
    if not os.path.exists('data'): os.makedirs('data')

    # 1. Surowe dane TDC [cite: 163, 169]
    full_obs_df = download_tdc_data()
    unique_smiles = full_obs_df['smiles'].unique()
    print(f"Pobrano {len(full_obs_df)} obserwacji dla {len(unique_smiles)} unikalnych SMILES.")

    # 2. Deskryptory RDKit (Kluczowe dla rozpuszczalności/przenikalności) [cite: 121, 122]
    print(f"\nKROK 2: Przetwarzanie {len(unique_smiles)} unikalnych cząsteczek (RDKit 200D)...")
    with Pool(N_CORES) as p:
        rdkit_res = list(tqdm(p.imap(compute_rdkit_worker, unique_smiles), total=len(unique_smiles)))

    rdkit_df = pd.DataFrame(rdkit_res, columns=[f'rdkit_{i}' for i in range(200)])
    rdkit_df['smiles'] = unique_smiles
    # Zapis punktu kontrolnego 2
    rdkit_df.to_csv("data/rdkit_features.csv", index=False)

    # 3. Cechy Kwantowe (Fizycznie ugruntowana reprezentacja 3D) [cite: 129, 130]
    print(f"\nKROK 3: Obliczanie cech kwantowych (QC + Maska)...")
    with Pool(N_CORES) as p:
        q_res = list(tqdm(p.imap(compute_quantum_worker, unique_smiles), total=len(unique_smiles)))

    q_cols = ['q_dipole', 'q_gap', 'q_electrons', 'q_energy', 'm_dipole', 'm_gap', 'm_electrons', 'm_energy']
    q_df = pd.DataFrame(q_res, columns=q_cols)
    q_df['smiles'] = unique_smiles
    # Zapis punktu kontrolnego 3
    q_df.to_csv("data/quantum_features.csv", index=False)

    # 4. Finalne scalanie w formacie "Long" [cite: 115, 116, 168]
    print("\nKROK 4: Scalanie cech i mapowanie do zadań...")
    features_df = pd.merge(rdkit_df, q_df, on='smiles')
    final_df = pd.merge(full_obs_df, features_df, on='smiles', how='left')

    # Finalny zbiór danych gotowy do treningu QW-MTL
    final_df.to_csv("data/qwmtl_final_dataset.csv", index=False)
    print(f"Sukces! Plik zapisany: data/qwmtl_final_dataset.csv. Rozmiar: {final_df.shape}")


if __name__ == "__main__":
    main()