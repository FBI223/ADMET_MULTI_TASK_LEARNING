# EXPERIMENT_CROSS_UNCERTAINTY

## Co się zmieniło względem EXPERIMENT_UNCERTAINTY

Zamiast **13 skalarów** sigma (jeden per task), uczona jest **macierz 13×13** gdzie `σ[i,j]` oznacza:
> *"z perspektywy trenowania taska i, jak niepewny/zaszumiony jest task j"*

```
L_total = sum_i sum_j [ L_j / σ[i,j]² + log(σ[i,j]) ]
```

Efektywna waga gradientu taska j = suma kolumny j: `sum_i 1/σ[i,j]²`

Model, dane, konfiguracja, pipeline — bez zmian.

## Jak interpretować macierz

| σ[i,j] | Znaczenie |
|---|---|
| < 1.0 | task j **pomaga** taskowi i — dostaje większą wagę |
| = 1.0 | neutralny (jak EXPERIMENT_MAIN) |
| > 1.0 | task j **koliduje** z taskiem i — dostaje mniejszą wagę |

**Diagonal** σ[i,i] = self-uncertainty (identyczne z EXPERIMENT_UNCERTAINTY).
**Off-diagonal** σ[i,j] = cross-task interaction — to co jest nowe.

## Struktura wyników

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv       — Task, MTL_GNN_CROSS, MTL_GNN, STL_GNN, XGBoost
    sigma_matrix_final.csv  — macierz 13×13 finalnych sigma
    sigma_colsum.csv        — Task, effective_weight (= suma kolumny 1/σ²)
    sigma_heatmap.png       — wizualizacja macierzy sigma
    run_stats.txt           — logi
    plots/                  — ponumerowane wykresy
    results/                — krzywe ROC per task
```
