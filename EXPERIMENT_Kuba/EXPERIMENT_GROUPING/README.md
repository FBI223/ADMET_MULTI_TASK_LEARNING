# EXPERIMENT_GROUPING

## Co się zmieniło względem EXPERIMENT_MAIN

Zamiast **jednego wspólnego backbone'u GIN** używane są **dwa osobne backbone'y GIN**:

- `gin_cyp` — obsługuje 5 tasków CYP (metabolizm enzymów)
- `gin_other` — obsługuje 8 pozostałych tasków (transport, toksyczność)

Wspólny zostaje vector MLP (Morgan/RDKit) oraz głowice task-specific.

## Dlaczego dzielimy

### Obserwacja z EXPERIMENT_MAIN

MTL traci z STL dokładnie na `herg`, `dili`, `skin_reaction`:

| Task | MTL_GNN | STL_GNN | Delta |
|---|---|---|---|
| herg | ~0.74 | ~0.79 | -5.1% |
| dili | ~0.73 | ~0.77 | -4.1% |

Przyczyna: backbone kompromisuje między dwoma grupami tasków o **kolizyjnych gradientach**.

### Potwierdzenie z EXPERIMENT_CROSS_UNCERTAINTY

Macierz σ[i,j] w EXPERIMENT_CROSS_UNCERTAINTY pokazała:

- Dla tasków CYP → off-diagonal σ z taskami herg/dili/skin_reaction ≈ 1.2–1.6 (kolidujące)
- Dla herg → σ z taskami CYP ≈ 1.3–1.5 (CYP traktuje herg jako szum)
- `herg` z CROSS_UNCERTAINTY zyskał +4.6% względem EXPERIMENT_MAIN

Wniosek: CYP-cluster i klaster toksyczności (herg/dili/skin_reaction) mają **różne wymagania
od backbone'u** — CYP potrzebuje wrażliwości na strukturę aromatyczną pierścieni (metabolizm),
a herg/dili potrzebuje globalnych cech cząsteczkowych (lipofilia, masa, ładunek).

## Podział tasków

```
CYP Group (gin_cyp):
    cyp2c19_veith, cyp2d6_veith, cyp3a4_veith, cyp1a2_veith, cyp2c9_veith

Other Group (gin_other):
    hia_hou, pgp_broccatelli, bbb_martins, herg, dili, skin_reaction, ames, clintox
```

## Architektura

```
                    ┌─────────────┐
                    │  Molekuła   │
                    └──────┬──────┘
               ┌───────────┼───────────┐
          gin_cyp        (graf)     gin_other
         (3×GINConv)               (3×GINConv)
               │                       │
         graph_bn_cyp             graph_bn_other
               │                       │
               └──────┐   ┌────────────┘
                      │   │
                  vector_mlp (shared, opcjonalny)
                      │   │
              fusion_bn_cyp │ fusion_bn_other
                      │   │
           ┌──────────┘   └──────────────┐
      CYP heads (×5)               Other heads (×8)
```

## Struktura wyników

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv    — Task, MTL_GNN_GROUPED, MTL_GNN, STL_GNN, XGBoost
    run_stats.txt
    plots/
    results/
```
