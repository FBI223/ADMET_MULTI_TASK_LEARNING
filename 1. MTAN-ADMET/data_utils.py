import pandas as pd
import torch
from torch.utils.data import Dataset
from cddd.inference import InferenceModel


class ADMETDataLoader(Dataset):
    def __init__(self, csv_path, cddd_model_path=None):
        df = pd.read_csv(csv_path)
        self.smiles = df['SMILES'].tolist()
        # Zakładamy, że pozostałe 24 kolumny to endpointy ADMET [cite: 73]
        self.labels = df.drop(columns=['SMILES']).values.astype(np.float32)

        # Inicjalizacja CDDD [cite: 197, 209]
        self.featurizer = InferenceModel(model_dir=cddd_model_path)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # Generowanie deskryptora 512-dim w locie lub cache
        smi = self.smiles[idx]
        emb = self.featurizer.to_vector([smi])[0]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(self.labels[idx])


def get_task_metadata(csv_path):
    df = pd.read_csv(csv_path).drop(columns=['SMILES'])
    fractions = df.notna().sum().values / len(df)
    # 18 pierwszych to klasyfikacja, 6 ostatnich to regresja (wg Table 1) [cite: 206]
    is_class = [True] * 18 + [False] * 6
    return {'fractions': fractions, 'is_classification': is_class}