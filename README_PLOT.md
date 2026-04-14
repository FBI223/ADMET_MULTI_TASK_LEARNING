
## RYSUNEK ARCHITEKTURY MODELU 
## ADMET Hybrid Model
##

```text
  WEJŚCIE
      |
      +-------------------------------------------------------+
      |                                                       |
[ A: Dane Grafowe ]                                   [ B: Dane Wektorowe ]
 (Atomy, Wiązania, Batch)                             (Morgan FP / RDKit Deskryptory)
      |                                                       |
      V                                                       V
========== NOGA GRAFOWA (GIN) ==========        ========== NOGA WEKTOROWA (MLP) ==========
|                                      |        |                                        |
|  [Warstwa Wejściowa (Atom Feat)]     |        |  [Konkatenacja Wektorów]               |
|            |                         |        |            |                           |
|            V                         |        |            V                           |
|  [3x GINConv + ReLU] <------------+  |        |  [Linear -> BN -> ReLU -> Dropout]     |
|  (Agregacja wiadomości z sąsiedztwa)|  |        |            |                           |
|            |                      |  |        |            V                           |
|            V                      |  |        |  [Linear -> ReLU]                      |
|  [Global Add Pool] (Readout)         |        |                                        |
|  (Suma cech atomów -> 1 wektor)      |        |                                        |
|            |                         |        |                                        |
|            V                         |        |                                        |
|  [Graph BatchNorm]                   |        |                                        |
|                                      |        |                                        |
===== [ g_emb ] (Graph Embedding) ======        ===== [ v_emb ] (Vector Embedding) ======
      |                                                       |
      |                 Wymiar: hidden_dim                    |
      |                                                       |
      +----------------------> [ F U Z J A ] <------------------+
                               ( torch.cat )
                                     |
                                     V
                          Wymiar: 2 * hidden_dim
                                     |
                          [ Fusion BatchNorm ]
                                     |
                          [combined_features] (Wspólny reprezentant ADMET)
                                     |
                                     V
      +-------------------------------------------------------+
      | (Tryb Multi-Task)                                     | (Tryb Single-Task)
      |                                                       |
[Głowica Task 1] -> [Pred 1]                            [Głowica Single] -> [Pred]
[Głowica Task 2] -> [Pred 2]                                  |
     ...                                                      V
[Głowica Task N] -> [Pred N]                            Zwraca: Tensor
      |
      V
Zwraca: Lista Tensorów
```


