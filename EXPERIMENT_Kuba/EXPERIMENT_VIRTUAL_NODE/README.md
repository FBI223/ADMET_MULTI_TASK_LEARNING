# EXPERIMENT_VIRTUAL_NODE

## Co się zmieniło względem EXPERIMENT_MAIN

Do każdej cząsteczki dodawany jest **wirtualny węzeł** (virtual node) połączony
krawędziami ze wszystkimi atomami w obie strony, przed trenowaniem GIN.

Model, dane (poza rozszerzeniem grafu), konfiguracja, funkcja kosztu — bez zmian.

## Dlaczego wirtualny węzeł pomaga

### Problem: ograniczone pole recepcyjne GIN

3-warstwowy GIN ma zasięg 3 przeskoków (3-hop neighborhood):

```
Atom A widzi: bezpośrednich sąsiadów + sąsiadów sąsiadów + sąsiadów³
```

Dla dużych cząsteczek (> 20 atomów) atomy po przeciwnych stronach cząsteczki
**nie wymieniają informacji** — GIN nie widzi globalnej struktury.

Taski ADMET zależące od globalnych właściwości:
- `herg` — zależy od masy cząsteczkowej i ogólnej lipofilności (całościowej)
- `bbb_martins` — bariera krew-mózg: masa + lipofilność + polarność całej cząsteczki
- `dili` — hepatotoksyczność: często wynika z globalnej reaktywności metabolicznej

### Rozwiązanie: wirtualny węzeł jako globalny komunikat

```
Molekuła bez VN:   A - B - C - D - E    (A i E nie komunikują się w 3-hop GIN)
Molekuła z VN:     A - B - C - D - E
                    \  |   |   |  /
                         VN            (VN widzi wszystkie atomy w 1 przeskoku)
```

Po jednej warstwie GIN:
- VN otrzymuje informacje od WSZYSTKICH atomów
- Każdy atom otrzymuje informacje od VN (= globalny kontekst)

Efektywnie: 2 warstwy GIN z VN ≈ "nieskończony" zasięg bez VN.

## Implementacja

Dla każdej cząsteczki (przed DataLoader):

```python
# Nowy węzeł: zero features [1, 9]
x = cat([x, zeros(1, 9)])

# Krawędzie: VN ↔ każdy atom
new_edges = [[0..n-1, vn_idx], [vn_idx, 0..n-1]]
edge_index = cat([edge_index, new_edges])
```

Dane przechowywane w osobnym cache: `data_cache/vn_cache_{hash}.pt`

## Struktura wyników

```
GNN / GNN_MORGAN / GNN_RDKIT / GNN_RDKIT_MORGAN
    final_results.csv    — Task, MTL_GNN_VN, MTL_GNN, STL_GNN, XGBoost
    run_stats.txt
    plots/
    results/
```
