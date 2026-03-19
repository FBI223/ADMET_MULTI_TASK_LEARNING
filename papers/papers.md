# ADMET Multi-Task – 5 Projekty (Detailed)

## 📊 Porównanie projektów

| Projekt | Dataset | Model | Wejście | Wyjście | Metody (jak działa) | Dane (szczegóły) | Problemy | Trudność | Skuteczność |
|--------|--------|------|--------|--------|--------------------|------------------|---------|----------|-------------|
| **Projekt 1 (Baseline MLP)** | TDC ADMET | MLP (RDKit) | SMILES → features | regression + classification | feature engineering + shared NN | publiczne benchmarki ADMET (bioavailability, toxicity, etc.) | imbalance, overfitting | ⭐⭐ | średnia |
| **Projekt 2 (MTAN-ADMET)** | MTAN dataset (~24 tasks) | MLP + attention | SMILES → embedding (CDDD) | multitask (24 endpoints) | adaptive task attention (ważenie tasków) | multitask dataset ADMET; bez grafów | noisy labels | ⭐⭐ | wysoka :contentReference[oaicite:0]{index=0} |
| **Projekt 3 (ADMET-AI)** | TDC (41 datasets) | GNN (Chemprop-RDKit) | SMILES → graph + RDKit | multitask | message passing + global descriptors | 41 datasetów (10 reg + 31 cls) | missing labels, imbalance | ⭐⭐⭐ | bardzo wysoka :contentReference[oaicite:1]{index=1} |
| **Projekt 4 (Quantum MTL)** | TDC (13 tasks) | GNN + quantum features | SMILES + quantum descriptors | multitask (classification) | dynamic loss weighting + feature fusion | standard TDC benchmark | feature complexity | ⭐⭐⭐⭐ | bardzo wysoka (12/13 tasks better) :contentReference[oaicite:2]{index=2} |
| **Projekt 5 (MTGL-ADMET)** | custom ADMET (~24 tasks) | GNN + gating | graph | multitask | auxiliary task selection + gating | multi-source ADMET datasets | training instability | ⭐⭐⭐⭐ | wysoka (+2–5%) |

---

## 🔍 Różnice (techniczne)

| Projekt | Encoder | Feature engineering | Multi-task strategy |
|--------|--------|--------------------|---------------------|
| Projekt 1 | brak (MLP) | RDKit | shared layers |
| Projekt 2 | embedding NN | CDDD | attention weighting |
| Projekt 3 | GNN (D-MPNN) | RDKit + graph | shared + heads |
| Projekt 4 | GNN | + quantum chemistry | weighted multitask |
| Projekt 5 | GNN | graph | dynamic task selection |

---