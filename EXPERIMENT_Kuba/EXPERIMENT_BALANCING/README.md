# EXPERIMENT_BALANCING

## Co się zmieniło względem EXPERIMENT_MAIN

Jedyna zmiana: funkcja kosztu MTL i STL.

**EXPERIMENT_MAIN** używa `MaskedBCELoss` — każda pozytywna i negatywna próbka waży tyle samo.

**EXPERIMENT_BALANCING** używa `BalancedMaskedBCELoss` — przed treningiem obliczany jest `pos_weight_t = neg_t / pos_t` dla każdego taska osobno ze zbioru treningowego. Przekazywany do `BCEWithLogitsLoss(pos_weight=...)`. Zadania z silną nierównowagą klas (np. CYP z 90% negatywnych) mają wyrównany wpływ gradientu.

Model, dane, konfiguracja, pipeline — bez zmian.

## Struktura wyników (identyczna z EXPERIMENT_MAIN)

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv     — Task, MTL_GNN, STL_GNN, XGBoost
    run_stats.txt         — logi
    plots/                — ponumerowane wykresy
    results/              — krzywe ROC per task
    pos_weights.csv       — nauczone wagi klas (nowe)
```
