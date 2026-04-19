# EXPERIMENT_PCGRAD

## Co się zmieniło względem EXPERIMENT_MAIN

Jedyna zmiana: pętla treningowa używa **Gradient Surgery (PCGrad)** zamiast zwykłego backward.

Model, architektura, dane, konfiguracja — bez zmian.

## Dlaczego

W EXPERIMENT_MAIN MTL traci z STL na `herg`, `dili`, `skin_reaction`. Przyczyna:
gradienty z klastru CYP i gradienty z tych tasków mają **ujemny cosinus** — ciągną
wspólny backbone w przeciwnych kierunkach. Backbone kompromisuje i jest przeciętny dla obu grup.

PCGrad (Yu et al., NeurIPS 2020) rozwiązuje to bez zmiany architektury:
jeśli gradient taska A koliduje z gradientem taska B (`dot < 0`), projekt A na
prostopadłą do B. Backbone dostaje tylko niesprzeczne komponenty gradientów.

```
Zwykłe MTL:   grad_total = sum_t grad_t          (kolizje sumują się)
PCGrad:        grad_i → grad_i - proj(grad_i, grad_j) jeśli dot < 0
```

Projekcja stosowana tylko na **wspólnym backbone** (GIN + vector MLP).
Głowice task-specific dostają własny gradient bez modyfikacji.

## Struktura wyników (identyczna z EXPERIMENT_MAIN)

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv    — Task, MTL_GNN_PCGRAD, MTL_GNN, STL_GNN, XGBoost
    run_stats.txt
    plots/
    results/
```
