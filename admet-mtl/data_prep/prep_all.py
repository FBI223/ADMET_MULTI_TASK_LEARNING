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

# --- KONFIGURACJA ---
N_CORES = 8
TASKS_ADME = [
    'bioavailability_ma', 'hia_hou', 'pgp_broccatelli', 'bbb_martins',
    'cyp2c9_veith', 'cyp2d6_veith', 'cyp3a4_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels'
]
TASKS_TOX = ['hERG_Karim', 'ames', 'dili']
ALL_TASKS = TASKS_ADME + TASKS_TOX


# --- KROK 1: POBIERANIE I STANDARYZACJA SPLITÓW ---
def download_tdc_data():
    print("Pobieranie danych z TDC...")
    all_data = []
    for name in ALL_TASKS:
        group = ADME(name=name) if name in TASKS_ADME else Tox(name=name)
        split = group.get_split()
        for s_name in ["train", "valid", "test"]:
            df = split[s_name][['Drug', 'Y']].copy()
            df.columns = ['smiles', 'label']
            df['task'] = name
            df['split'] = s_name
            all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    # Pivotowanie z zachowaniem splitów [cite: 168]
    # Używamy mean, by obsłużyć ewentualne duplikaty w ramach tego samego splitu
    df_wide = full_df.pivot_table(index=['smiles', 'split'], columns='task', values='label').reset_index()
    return df_wide


# --- KROK 2: DESKRYPTORY RDKit (200 CECH) ---
def compute_rdkit_200(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [0.0] * 200

    # Pobieramy 200 najistotniejszych deskryptorów fizykochemicznych
    desc_names = [d[0] for d in Descriptors._descList][:200]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    try:
        return list(calc.CalcDescriptors(mol))
    except:
        return [0.0] * 200


# --- KROK 3: CECHY KWANTOWE + MASKA BINARNA --- [cite: 133, 138]
def compute_quantum_features(smiles):
    """Oblicza dipole, gap, electrons, energy oraz maskę[cite: 133]."""
    mask = [1, 1, 1, 1]  # 1 = dane obecne, 0 = brak
    features = [0.0, 0.0, 0.0, 0.0]

    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

        calc = Calculator(get_method("GFN2-xTB"), Z, coords)
        res = calc.singlepoint()

        features = [
            np.linalg.norm(res.get_dipole()),  # dipole moment
            res.get_lumo() - res.get_homo(),  # HOMO-LUMO gap
            float(Z.sum()),  # total electrons
            res.get_energy()  # total energy
        ]
    except:
        mask = [0, 0, 0, 0]  # Zaznaczamy brak danych dla modelu

    return features + mask


# --- KROK 4: PROCESOWANIE RÓWNOLEGŁE I ZAPIS ---
def main():
    if not os.path.exists('data'): os.makedirs('data')

    # 1. Pobierz dane
    df = download_tdc_data()
    smiles_list = df['smiles'].tolist()

    # 2. Oblicz RDKit (200)
    print("Obliczanie 200 deskryptorów RDKit...")
    with Pool(N_CORES) as p:
        rdkit_feats = p.map(compute_rdkit_200, smiles_list)
    rdkit_df = pd.DataFrame(rdkit_feats, columns=[f'rdkit_{i}' for i in range(200)])

    # 3. Oblicz Quantum + Mask
    print("Obliczanie cech kwantowych (może potrwać)...")
    with Pool(N_CORES) as p:
        q_results = p.map(compute_quantum_features, smiles_list)

    q_cols = ['q_dipole', 'q_gap', 'q_electrons', 'q_energy',
              'm_dipole', 'm_gap', 'm_electrons', 'm_energy']
    q_df = pd.DataFrame(q_results, columns=q_cols)

    # 4. Połącz wszystko
    final_df = pd.concat([df, rdkit_df, q_df], axis=1)

    # Zapisz do formatu kompatybilnego z Chemprop (CSV)
    final_df.to_csv("data/qwmtl_final_dataset.csv", index=False)
    print(f"Sukces! Zbiór danych zapisany. Rozmiar: {final_df.shape}")


if __name__ == "__main__":
    main()