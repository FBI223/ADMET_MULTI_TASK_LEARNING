
from sklearn.preprocessing import StandardScaler
from tdc.utils import retrieve_dataset_names
import os
import hashlib
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from tdc.single_pred import ADME, Tox
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GNNLoader

from EXPERIMENT.config import Config

'''
=== DOSTĘPNE DATASETY ADME ===
1. lipophilicity_astrazeneca
2. solubility_aqsoldb
3. hydrationfreeenergy_freesolv
4. caco2_wang
5. pampa_ncats
6. approved_pampa_ncats
7. hia_hou
8. pgp_broccatelli
9. bioavailability_ma
10. vdss_lombardo
11. cyp2c19_veith
12. cyp2d6_veith
13. cyp3a4_veith
14. cyp1a2_veith
15. cyp2c9_veith
16. cyp2c9_substrate_carbonmangels
17. cyp2d6_substrate_carbonmangels
18. cyp3a4_substrate_carbonmangels
19. bbb_martins
20. b3db_classification
21. b3db_regression
22. ppbr_az
23. half_life_obach
24. clearance_hepatocyte_az
25. clearance_microsome_az
26. hlm
27. rlm

=== DOSTĘPNE DATASETY TOXICITY ===
1. tox21
2. toxcast
3. clintox
4. herg_karim
5. herg
6. herg_central
7. dili
8. skin_reaction
9. ames
10. carcinogens_lagunin
11. ld50_zhu
'''

def list_available_tdc_datasets():
    """
    Wyświetla listę dostępnych datasetów w TDC.
    Poprawiono klucze na zgodne z aktualnym API TDC.
    """
    print("=== DOSTĘPNE DATASETY ADME ===")
    try:
        # TDC używa 'ADME' (dużymi literami)
        adme_list = retrieve_dataset_names('ADME')
        for i, name in enumerate(adme_list):
            print(f"{i+1}. {name}")
    except KeyError:
        print("Błąd: Nie znaleziono grupy 'ADME'. Spróbuj 'admet_group'.")

    print("\n=== DOSTĘPNE DATASETY TOXICITY ===")
    try:
        # TDC używa 'Toxicity' zamiast 'tox'
        tox_list = retrieve_dataset_names('Tox')
        for i, name in enumerate(tox_list):
            print(f"{i+1}. {name}")
    except KeyError:
        print("Błąd: Nie znaleziono grupy 'Toxicity'.")



def fetch_admet_data(tasks):
    """
    Pobiera dane z TDC i łączy je w jeden DataFrame, zachowując informację o splitach.
    """
    df_final = None

    for task in tasks:
        try:
            data = ADME(name=task)
        except:
            data = Tox(name=task)

        # Pobieramy słownik ze splitami: {'train': df, 'valid': df, 'test': df}
        splits = data.get_split()

        task_df_list = []
        for split_name, df in splits.items():
            df = df[['Drug', 'Y']].copy()
            df.columns = ['SMILES', task]
            df['split'] = split_name  # Dodajemy kolumnę informacyjną
            task_df_list.append(df)

        # Łączymy train/valid/test dla danego zadania w jeden df
        current_task_df = pd.concat(task_df_list, axis=0)

        if df_final is None:
            df_final = current_task_df
        else:
            # Łączymy po SMILES i split, aby cząsteczka trafiła do właściwego worka w MTL
            # 'outer' join zapewnia, że nie zgubimy danych, jeśli cząsteczka jest tylko w jednym zadaniu
            df_final = pd.merge(df_final, current_task_df, on=['SMILES', 'split'], how='outer')

    return df_final


def prepare_flat_features(dataset):
    """Konwertuje obiekty grafowe na płaskie wektory, obsługując brak Morgana/RDKit."""
    X, Y = [], []
    for data in dataset:
        feats = []

        # 1. Dodaj Morgana, jeśli jest
        if hasattr(data, 'morgan'):
            feats.append(data.morgan.numpy().reshape(1, -1))

        # 2. Dodaj RDKit, jeśli jest
        if hasattr(data, 'rdkit'):
            feats.append(data.rdkit.numpy().reshape(1, -1))

        # 3. Jeśli NIE MA Morgana ani RDKit, zrób uśrednienie cech grafu (Pooling)
        if len(feats) == 0:
            # data.x ma kształt [liczba_atomów, 9 cech]
            # Liczymy średnią po atomach -> wynik [1, 9]
            graph_mean = data.x.mean(dim=0).numpy().reshape(1, -1)
            feats.append(graph_mean)

        X.append(np.concatenate(feats, axis=1))
        Y.append(data.y.numpy())

    return np.vstack(X), np.vstack(Y)




def smiles_to_morgan(smiles, n_bits=1024):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    # Tworzymy generator (radius=2 odpowiada ECFP4)
    # fpSize zastępuje stary parametr nBits
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)

    # Pobieramy fingerprint bezpośrednio jako tablicę NumPy (bitową)
    fp = generator.GetFingerprintAsNumPy(mol)

    return fp.astype(np.float32)


def smiles_to_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(208)

    results = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            # Sprawdzanie czy wartość jest poprawną liczbą
            if val is None or np.isnan(val) or np.isinf(val):
                results.append(0.0)
            else:
                # Ograniczenie ekstremalnie dużych wartości (clipping)
                # Niektóre deskryptory potrafią wyrzucić 1e+30
                results.append(np.clip(float(val), -1e6, 1e6))
        except:
            results.append(0.0)

    return np.array(results)


def smiles_to_graph(smiles, y_labels):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 2. Wyciąganie cech węzłów (Atomów)
    node_features = []
    for atom in mol.GetAtoms():
        # Lista cech dla każdego atomu
        features = [
            atom.GetAtomicNum(),  # Liczba atomowa
            atom.GetDegree(),  # Liczba sąsiadów
            atom.GetFormalCharge(),  # Ładunek formalny
            float(atom.GetIsAromatic()),  # Czy aromatyczny (0/1)
            atom.GetImplicitValence(),  # Wartościowość implikowana
            int(atom.GetHybridization()),  # Typ hybrydyzacji (jako int z Enum)
            atom.GetNumRadicalElectrons(),  # Liczba niesparowanych elektronów
            atom.GetMass() * 0.01,  # Masa atomowa (skalowana dla stabilności)
            float(atom.IsInRing())  # Czy w pierścieniu (0/1)
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # 3. Wyciąganie krawędzi (Wiązań) - Graf nieskierowany
    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Dodajemy krawędź w obie strony (standard w GNN)
        edge_indices.append([start, end])
        edge_indices.append([end, start])

    # Jeśli cząsteczka to pojedynczy atom (np. He), edge_index musi być pusty o dobrym kształcie
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 4. Przygotowanie etykiet (Y)
    # Konwersja na tensor [1, num_tasks], obsługa wartości NaN
    y = torch.tensor(y_labels, dtype=torch.float).view(1, -1)

    # 5. Złożenie obiektu Data
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def normalize_descriptors(dataframe, tasks):
    # Wyciągamy kolumny, które nie są SMILES ani etykietami (czyli nasze deskryptory)
    feature_cols = [c for c in dataframe.columns if c not in ['SMILES'] + tasks]
    scaler = StandardScaler()
    dataframe[feature_cols] = scaler.fit_transform(dataframe[feature_cols])
    return dataframe, scaler

def smiles_to_hybrid_data(smiles, y_labels, config):
    # 1. Graf (jak wcześniej)
    data = smiles_to_graph(smiles, y_labels)  # Funkcja z poprzedniego kroku
    if data is None: return None

    # 2. Morgan Fingerprints
    if config.use_morgan:
        fp = smiles_to_morgan(smiles, n_bits=config.morgan_dim)
        data.morgan = torch.tensor(fp, dtype=torch.float).view(1, -1)

    # 3. RDKit Descriptors
    if config.use_rdkit:
        desc = smiles_to_rdkit_descriptors(smiles)  # Funkcja z poprzedniego kroku
        data.rdkit = torch.tensor(desc, dtype=torch.float).view(1, -1)

    return data










def get_cache_hash(config):
    """Tworzy unikalny skrót na podstawie aktualnej konfiguracji zadań i wejść."""
    config_dict = {
        "tasks": sorted(config.tasks),
        "morgan_dim": config.morgan_dim,
        "use_graph": config.use_graph,
        "use_morgan": config.use_morgan,
        "use_rdkit": config.use_rdkit
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def get_full_data(config):
    """
    Strategia Master Cache:
    1. Zawsze oblicza/wczytuje komplet cech (Graph + Morgan + RDKit).
    2. Dynamicznie usuwa niepotrzebne cechy przed zwróceniem loaderów.
    """

    # 1. Stały hash zależny tylko od zadań
    task_hash = hashlib.md5("".join(sorted(config.tasks)).encode()).hexdigest()

    # --- ZMIANA: Stały folder na cache, niezależny od results_dir eksperymentu ---
    cache_base_dir = "data_cache"
    os.makedirs(cache_base_dir, exist_ok=True)

    master_cache_path = os.path.join(cache_base_dir, f"master_cache_{task_hash}.pt")
    df_cache_path = os.path.join(cache_base_dir, f"master_df_{task_hash}.pkl")
    # --------------------------------------------------------------------------

    all_data = None
    raw_train_df = None

    # 2. PRÓBA ODCZYTU LUB GENEROWANIA
    if os.path.exists(master_cache_path) and os.path.exists(df_cache_path):
        print(f"\n>>> Wczytywanie Master Cache: {task_hash[:8]}...")
        try:
            all_data = torch.load(master_cache_path, weights_only=False)
            raw_train_df = pd.read_pickle(df_cache_path)
        except Exception as e:
            print(f">>> Cache niekompatybilny ({type(e).__name__}), regeneruję...")
            for p in (master_cache_path, df_cache_path):
                if os.path.exists(p):
                    os.remove(p)
            all_data = None
            raw_train_df = None

    if all_data is None:
        print(f"\n>>> Master Cache nie znaleziony. Obliczam WSZYSTKIE cechy (RDKit + Morgan + Graph)...")

        # Pobieranie danych z TDC (łączone DataFrame)
        split_dfs = {'train': None, 'valid': None, 'test': None}
        for task in config.tasks:
            try:
                data_loader = ADME(name=task)
            except:
                data_loader = Tox(name=task)

            splits = data_loader.get_split()
            for s in ['train', 'valid', 'test']:
                df = splits[s][['Drug', 'Y']].rename(columns={'Drug': 'SMILES', 'Y': task})
                if split_dfs[s] is None:
                    split_dfs[s] = df
                else:
                    split_dfs[s] = pd.merge(split_dfs[s], df, on='SMILES', how='outer')

        # Wymuszamy obliczenie wszystkiego do Master Cache
        # Tworzymy kopię configu z włączonymi wszystkimi flagami
        full_feat_config = Config()
        full_feat_config.use_graph = True
        full_feat_config.use_morgan = True
        full_feat_config.use_rdkit = True

        all_data = {'train': [], 'valid': [], 'test': []}
        for s in ['train', 'valid', 'test']:
            print(f"Konwertowanie splitu {s} na pełne dane hybrydowe...")
            df = split_dfs[s]
            for _, row in tqdm(df.iterrows(), total=len(df)):
                labels = row[config.tasks].values.astype(float)
                # Obliczamy komplet cech
                data_obj = smiles_to_hybrid_data(row['SMILES'], labels, full_feat_config)
                if data_obj:
                    all_data[s].append(data_obj)

        raw_train_df = split_dfs['train']

        # Zapis do cache
        print(">>> Zapisywanie Master Cache do pliku...")
        torch.save(all_data, master_cache_path)
        raw_train_df.to_pickle(df_cache_path)

    # 3. FILTROWANIE DANYCH W LOCIE
    # Tutaj naprawiamy Twój błąd - all_data na pewno już istnieje (z cache lub z generowania)
    print(f">>> Dostosowuję dane: Graph={config.use_graph}, RDKit={config.use_rdkit}, Morgan={config.use_morgan}")

    for s in ['train', 'valid', 'test']:
        for data_obj in all_data[s]:
            # Jeśli w danym eksperymencie nie chcemy cechy, usuwamy ją z obiektu Data
            if not config.use_rdkit and hasattr(data_obj, 'rdkit'):
                delattr(data_obj, 'rdkit')
            if not config.use_morgan and hasattr(data_obj, 'morgan'):
                delattr(data_obj, 'morgan')

            # Jeśli nie chcemy grafu, model i tak go zignoruje,
            # ale możemy np. wyczyścić data_obj.x dla pewności (opcjonalne)

    # 4. TWORZENIE LOADERÓW
    train_loader = GNNLoader(all_data['train'], batch_size=config.batch_size, shuffle=True)
    val_loader = GNNLoader(all_data['valid'], batch_size=config.batch_size)
    test_loader = GNNLoader(all_data['test'], batch_size=config.batch_size)

    return (
        train_loader,
        val_loader,
        test_loader,
        all_data['train'],
        all_data['test'],
        raw_train_df
    )

def get_full_data_old(config):
    """
    Pobiera dane z TDC, łączy je i przygotowuje Loadery dla GNN.
    Wykorzystuje cache .pt (dla grafów) i .pkl (dla DataFrame).
    """
    cache_hash = get_cache_hash(config)
    cache_path = os.path.join(config.results_dir, f"cache_data_{cache_hash}.pt")
    df_cache_path = os.path.join(config.results_dir, f"cache_df_{cache_hash}.pkl")

    # 1. PRÓBA ODCZYTU Z CACHE
    if os.path.exists(cache_path) and os.path.exists(df_cache_path):
        print(f"\n>>> Wczytywanie danych z cache: {cache_hash[:8]}...")
        processed_datasets = torch.load(cache_path, weights_only=False)
        raw_train_df = pd.read_pickle(df_cache_path)
    else:
        print(f"\n>>> Cache nie znaleziony. Rozpoczynam pełne przetwarzanie (Tasks: {len(config.tasks)})...")

        # 2. POBIERANIE SUROWYCH DANYCH Z TDC
        split_dfs = {'train': None, 'valid': None, 'test': None}
        for task in config.tasks:
            print(f"Pobieranie: {task}")
            try:
                data_loader = ADME(name=task)
            except:
                data_loader = Tox(name=task)

            splits = data_loader.get_split()
            for s in ['train', 'valid', 'test']:
                df = splits[s][['Drug', 'Y']].rename(columns={'Drug': 'SMILES', 'Y': task})
                if split_dfs[s] is None:
                    split_dfs[s] = df
                else:
                    split_dfs[s] = pd.merge(split_dfs[s], df, on='SMILES', how='outer')

        # 3. KONWERSJA NA OBIEKTY HYBRYDOWE
        processed_datasets = {'train': [], 'valid': [], 'test': []}
        for s in ['train', 'valid', 'test']:
            print(f"Konwertowanie splitu {s} na grafy/wektory...")
            df = split_dfs[s]
            for _, row in tqdm(df.iterrows(), total=len(df)):
                labels = row[config.tasks].values.astype(float)
                # Wykorzystujemy Twoją funkcję smiles_to_hybrid_data
                data_obj = smiles_to_hybrid_data(row['SMILES'], labels, config)
                if data_obj:
                    processed_datasets[s].append(data_obj)

        raw_train_df = split_dfs['train']

        # 4. ZAPIS DO CACHE
        print(">>> Zapisywanie przetworzonych danych do plików cache...")
        torch.save(processed_datasets, cache_path)
        raw_train_df.to_pickle(df_cache_path)

    # 5. TWORZENIE LOADERÓW
    train_loader = GNNLoader(processed_datasets['train'], batch_size=config.batch_size, shuffle=True)
    val_loader = GNNLoader(processed_datasets['valid'], batch_size=config.batch_size)
    test_loader = GNNLoader(processed_datasets['test'], batch_size=config.batch_size)

    return (
        train_loader,
        val_loader,
        test_loader,
        processed_datasets['train'],
        processed_datasets['test'],
        raw_train_df
    )

