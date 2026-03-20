import numpy as np
from cddd.inference import InferenceModel

class CDDDFeaturizer:
    def __init__(self, model_dir=None):
        # Inicjalizacja pretrenowanego modelu CDDD (wymaga wag modelu)
        self.model = InferenceModel(model_dir)

    def featurize(self, smiles_list):
        # Translacja SMILES na 512-wymiarowe wektory latentne
        embeddings = self.model.to_vector(smiles_list)
        return np.array(embeddings, dtype=np.float32)