import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

top_20_feats = [
    'MolLogP',           # Lipofilowość (kluczowa dla ADMET)
    'TPSA',              # Polarność (przenikanie przez błony)
    'MolWt',             # Masa cząsteczkowa
    'qed',               # Wskaźnik "lekopodobności"
    'NumHAcceptors',     # Akceptory wiązań wodorowych
    'NumHDonors',        # Donory wiązań wodorowych
    'NumRotatableBonds', # Elastyczność cząsteczki
    'FractionCSP3',      # Nasycenie (trójwymiarowość)
    'HeavyAtomCount',    # Liczba atomów ciężkich
    'NumAromaticRings',  # Liczba pierścieni aromatycznych
    'NumSaturatedRings', # Liczba pierścieni nasyconych
    'MolMR',             # Refrakcja molowa (objętość/polaryzowalność)
    'NHOHCount',         # Liczba grup NH i OH
    'NOCount',           # Liczba atomów N i O
    'NumValenceElectrons', # Liczba elektronów walencyjnych
    'MaxAbsPartialCharge', # Maksymalny ładunek cząstkowy (reaktywność)
    'LabuteASA',         # Powierzchnia dostępna dla rozpuszczalnika
    'BalabanJ',          # Indeks topologiczny (rozgałęzienie)
    'RingCount',         # Całkowita liczba pierścieni
    'MaxPartialCharge'   # Najwyższy ładunek dodatni
]


def get_morgan_features(smiles_series, radius=2, n_bits=1024):
    """Nowoczesny generator Morgan Fingerprints (RDKit > 2023.03)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles_series:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = gen.GetFingerprintAsNumPy(mol)
            fps.append(fp)
        else:
            fps.append(np.zeros(n_bits, dtype=np.uint8))

    return np.array(fps)


def evaluate_model(model, X_test, y_test, task_name):
    """Funkcja do wyświetlania wyników."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # print(f"--- Wyniki dla zadania: {task_name} ---")
    # print(f"AUC-ROC:  {auc:.4f}")
    # print(f"Accuracy: {acc:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    # print("-" * 35)

    return {"task": task_name, "auc": auc, "acc": acc, "f1": f1}


def plot_importance(model, feature_names, task_name, top_n=15):
    importances = model.feature_importances_

    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(top_n)

    print(f"\n--- Top {top_n} najważniejszych cech dla {task_name} ---")
    print(feat_imp.to_string(index=False))



def run_single_task_pipeline(df, task_list, *, features=top_20_feats, n_estimators=300):
    """Główna pętla trenująca modele."""
    all_results = []

    for task in task_list:
        task_df = df[df['task'] == task].dropna(subset=['label'])

        if task_df.empty:
            continue

        morgan_fp = get_morgan_features(task_df['smiles'])

        rdkit_data = task_df[features].values
        X_combined = np.hstack([rdkit_data, morgan_fp])
        y = task_df['label'].values

        train_idx = task_df['split'] == 'train'
        val_idx = task_df['split'] == 'valid'
        test_idx = task_df['split'] == 'test'

        X_train, y_train = X_combined[train_idx], y[train_idx]
        X_test, y_test = X_combined[test_idx], y[test_idx]

        if len(X_test) == 0:
            X_test, y_test = X_combined[val_idx], y[val_idx]

        # n_jobs=-1 oznacza wykorzystanie wszystkich rdzeni logicznych
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42, max_features='sqrt')
        rf.fit(X_train, y_train)

        res = evaluate_model(rf, X_test, y_test, task)

        # 1. Tworzysz nazwy dla bitów Morgana
        morgan_names = [f"bit_{i}" for i in range(1024)]
        all_feature_names = features + morgan_names
        plot_importance(rf, all_feature_names, task)
        all_results.append(res)

    return pd.DataFrame(all_results)
