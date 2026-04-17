# EXPERIMENT_UNCERTAINTY

## Co się zmieniło względem EXPERIMENT_MAIN

Jedyna zmiana: funkcja kosztu MTL używa **uczalnych wag niepewności** per zadanie (Kendall et al., CVPR 2018).

```
L = sum_t [ L_t / exp(log_sigma_sq_t) + log_sigma_sq_t / 2 ]
```

`log_sigma_sq_t` — uczalny parametr per zadanie (nn.Parameter), inicjalizowany na 0 (sigma=1 → równe ważenie na starcie). Zadania trudniejsze lub bardziej zaszumione dostają automatycznie mniejszą wagę w trakcie treningu.

Model, dane, konfiguracja, pipeline — bez zmian.

## Struktura wyników (identyczna z EXPERIMENT_MAIN + dodatkowe)

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv      — Task, MTL_GNN_UNCERTAINTY, MTL_GNN, STL_GNN, XGBoost
    sigma_weights.csv      — Task, sigma_final, weight_final (= 1 / sigma^2)
    sigma_evolution.csv    — sigma per zadanie per epoka
    run_stats.txt          — logi
    plots/                 — ponumerowane wykresy
    results/               — krzywe ROC per task
```

## Jak interpretować sigma

| sigma | Znaczenie |
|---|---|
| < 1.0 | zadanie dostało większą wagę niż na starcie |
| = 1.0 | waga bez zmian (jak EXPERIMENT_MAIN) |
| > 1.0 | zadanie dostało mniejszą wagę — model uznał je za trudniejsze/bardziej zaszumione |
