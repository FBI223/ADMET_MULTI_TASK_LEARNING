# EXPERIMENT_EDGE_FEATURES

## Co się zmieniło względem EXPERIMENT_MAIN

Zamiast **GINConv** (ignoruje cechy krawędzi) używany jest **GINEConv** z 6-wymiarowym
wektorem cech każdego wiązania chemicznego.

## Dlaczego cechy krawędzi mają znaczenie

GINConv traktuje wszystkie wiązania identycznie — liczy tylko połączenia (topologię grafu).
W chemii wiązanie podwójne C=C różni się fundamentalnie od pojedynczego C-C:
- Wiązanie podwójne: planarna geometria, reaktywność elektrofilowa, konjugacja
- Wiązanie aromatyczne: delokalizacja elektronów, stabilność termodynamiczna

Dla tasków ADMET opartych na metabolizmie (CYP) i lipofilności (BBB, hERG) struktura
elektroniczna wiązań bezpośrednio wpływa na wynik.

## Cechy krawędzi (6 na wiązanie)

| Cecha | Wymiar | Znaczenie |
|---|---|---|
| Bond type SINGLE | 1 | σ-wiązanie C-C, C-N, C-O |
| Bond type DOUBLE | 1 | π-wiązanie, reaktywność |
| Bond type TRIPLE | 1 | Nitryle, alkiny |
| Bond type AROMATIC | 1 | Systemy aromatyczne |
| IsInRing | 1 | Część pierścienia (cykliczny) |
| IsConjugated | 1 | System koniugowany |

## Jak GINEConv używa cech krawędzi

```
GINConv:  h_v = MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)
GINEConv: h_v = MLP((1+ε)·h_v + Σ_{u∈N(v)} ReLU(h_u + W·edge_attr_{u,v}))
```

W (linear projection) mapuje 6-wymiarowe cechy krawędzi na przestrzeń hidden_dim,
następnie dodaje do reprezentacji sąsiada przed agregacją.

## Cache

Dane są przetwarzane osobno (cechy krawędzi nie są w Master Cache).
Nowy cache: `data_cache/edge_cache_{hash}.pt`

## Struktura wyników

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv    — Task, MTL_GNN_EDGE, MTL_GNN, STL_GNN, XGBoost
    run_stats.txt
    plots/
    results/
```
