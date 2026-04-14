
import torch
import os

class Config:
    def __init__(self):
        self.tasks_quant = [
            # --- ADME Classification ---
            'bioavailability_ma',  # Dostępność biologiczna
            'hia_hou',  # Wchłanianie jelitowe
            'pgp_broccatelli',  # Transport (P-gp)
            'bbb_martins',  # Bariera krew-mózg

            # CYP Inhibition (Veith)
            'cyp2c9_veith',
            'cyp2d6_veith',
            'cyp3a4_veith',

            # CYP Substrate (CarbonMangels) - świetne do MTL z Veith!
            'cyp2c9_substrate_carbonmangels',
            'cyp2d6_substrate_carbonmangels',
            'cyp3a4_substrate_carbonmangels',

            # --- Toxicity Classification ---
            'hERG_Karim',  # Kardiotoksyczność
            'ames',  # Mutagenność
            'dili'  # Toksyczność wątroby
        ]

        self.tasks_old = [
            # --- ADME Classification ---
            'hia_hou',  # Wchłanianie jelitowe
            'pgp_broccatelli',  # P-glikoproteina (transporter)
            'bbb_martins',  # Bariera krew-mózg
            'cyp2c19_veith',  # Metabolizm (enzymy)
            'cyp2d6_veith',
            'cyp3a4_veith',
            'cyp1a2_veith',
            'cyp2c9_veith',

            # --- Toxicity Classification ---
            'herg',  # Kardiotoksyczność
            'dili',  # Toksyczność wątroby
            'skin_reaction',  # Reakcje skórne
            'ames',  # Mutagenność
            'clintox'  # Toksyczność kliniczna (FDA approved vs clinical failure)
        ]

        self.tasks = [
            # --- Klaster Metaboliczny (Największy boost z MTL) ---
            'cyp2c19_veith',
            'cyp2d6_veith',
            'cyp3a4_veith',
            'cyp1a2_veith',
            'cyp2c9_veith',

            # --- Bariery i Transport (Wspólne cechy fizykochemiczne) ---
            'hia_hou',  # Wchłanianie
            'pgp_broccatelli',  # Transport (pompka)
            'bbb_martins',  # Przenikanie do mózgu

            # --- Toksyczność (Powiązania z metabolizmem i transportem) ---
            'herg',  # Kardio
            'dili',  # Wątroba (silna ujemna korelacja z P-gp!)
        ]

        self.use_graph = True
        self.use_morgan = False
        self.use_rdkit = False
        self.node_dim = 9 # Zgodnie z Twoją funkcją smiles_to_graph
        self.morgan_dim = 1024
        self.rdkit_dim = 217 # Zmienione na pełną listę deskryptorów
        self.hidden_dim = 128
        self.batch_size = 64
        self.lr = 0.001
        self.epochs = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- STRUKTURA FOLDERÓW (Kluczowa poprawka!) ---
        self.base_dir = "experiments_results"
        self.cache_dir = "data_cache"  # Tutaj Master Cache będzie bezpieczny
        self.results_dir = "results_default"  # To będzie nadpisane w pętli

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.base_dir, exist_ok=True)

