
---

# Predykcja profilu ADMET z wykorzystaniem Multi-Task Learning

---

# Spis treści

1. Opis projektu  
2. Hipotezy do sprawdzenia  
   - Przewaga MTL nad STL  
   - Korelacje biologiczne  
   - Wpływ reprezentacji  
3. Teoria  
   - GIN (Graph Isomorphism Network)  
   - MLP (Multi-Layer Perceptron)  
   - Multi-Head / Multi-Task Learning  
4. Model: ADMET Hybrid Model  
   - Architektura modelu  
   - Komponenty modelu  
   - Noga wektorowa (MLP)  
   - Fuzja (Concatenation)  
   - Multi-Head Architecture  
   - Schemat przepływu danych  
   - Architektura matematyczna  
   - Multi-Task vs Single-Task  
   - Funkcja straty  
5. Config / Kontroler  
   - Wybór zadań  
   - Dane wejściowe  
   - Parametry treningu  
6. Zarządzanie danymi i pipeline  
   - Źródła danych (TDC)  
   - Data merging  
   - Feature engineering  
   - Cache  
   - Loadery  
7. Trening i ewaluacja  
   - Pętla treningowa  
   - Tryby eksperymentów  
   - Metryki  
   - Korelacje  
   - Porównanie modeli  
8. Problemy i ograniczenia  
   - Konflikt gradientów  
   - Ograniczenia modelu  
9. Planowane usprawnienia  
   - Virtual node  
   - Edge features  
   - Attention (GAT)  

---

# 1. Opis projektu
Nawet najbardziej aktywny związek nie stanie się lekiem, jeśli posiada słabe właściwości farmakokinetyczne. Profil **ADMET** (*Absorption, Distribution, Metabolism, Excretion, Toxicity*) określa, czy cząsteczka dotrze do miejsca docelowego w organizmie, jak długo będzie działać i czy nie wywoła niepożądanych efektów. Przewidywanie tych właściwości *in silico* jest kluczowe na wczesnych etapach projektowania leków, pozwalając wyeliminować problematyczne związki przed kosztownymi badaniami eksperymentalnymi.

Celem projektu jest budowa zaawansowanego modelu predykcyjnego dla wielu parametrów ADMET jednocześnie, wykorzystując podejście **Multi-Task Learning (MTL)**. Projekt opiera się na założeniu, że wspólne uczenie powiązanych właściwości (np. rozpuszczalność a wchłanianie) pozwala na transfer wiedzy wewnątrz sieci neuronowej i poprawę jakości predykcji.

---



# 2. Hipotezy do sprawdzenia
## A. Przewaga MTL nad STL:
Model multi-task osiąga lepszą średnią jakość predykcji ($AUROC$ dla klasyfikacji, $RMSE$ dla regresji) niż analogiczne modele single-task.

## B, Korelacje biologiczne: 
Endpointy powiązane (np. rozpuszczalność i *Caco-2 permeability*) wykazują większy zysk z MTL niż endpointy niepowiązane.

## C. Wpływ reprezentacji:
Połączenie grafów z deskryptorami (Hybrid Approach) daje stabilniejsze wyniki niż każda z tych metod stosowana osobno.



---

# 3. Teoria


## A. GIN (Graph Isomorphism Network) 

W projekcie wykorzystano warstwy **GINConv**, które stanowią odpowiednia architekture w dziedzinie chemioinformatyki.

### Jak działa GIN?
Większość sieci GNN (np. GCN czy SAGE) stosuje proste uśrednianie cech sąsiadów. GIN wyróżnia się tym, że potrafi rozróżniać struktury grafowe tak skutecznie, jak **test izomorfizmu grafów Weisfeilera-Lehmana (1-WL)**. Oznacza to, że model jest w stanie odróżnić cząsteczki o identycznej liczbie atomów i wiązań, ale innej topologii, czego prostsze sieci często nie potrafią.


**Mechanizm warstwy GINConv:**
Dla każdego atomu $v$, nowa reprezentacja cech $h_v^{(k)}$ jest obliczana według wzoru:

$$h_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$

* **Agregacja Sumą:** GIN używa sumowania zamiast średniej, co pozwala zachować informację o liczbie sąsiadów (stopniu atomu).
* **Rola MLP:** Wykorzystanie wielowarstwowego perceptrona (MLP) po agregacji pozwala sieci uczyć się funkcji iniektywnych (różnowartościowych), co jest kluczem do jej potężnej zdolności dyskryminacyjnej.
* **Parametr $\epsilon$ (Epsilon):** Pozwala modelowi dynamicznie ważyć znaczenie cech samego atomu w stosunku do informacji płynącej z jego otoczenia.

---




# B. Multi-Layer Perceptron (MLP)

## Jak działa MLP?

**Multi-Layer Perceptron (MLP)** to sieć typu feed-forward do przetwarzania danych wektorowych (np. fingerprinty, deskryptory).

Struktura:

* warstwa wejściowa
* warstwy ukryte (Dense + aktywacja)
* warstwa wyjściowa

## Wzór warstwy

$$
h^{(l)} = \sigma\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)
$$


## Właściwości

* przetwarza dane globalne
* dobre dla RDKit i fingerprintów
* brak informacji o strukturze grafu

---

# C. Multi-Head (Multi-Task Learning)

## Idea

Architektura:

* wspólny **backbone**
* wiele **headów** (zadań)

## Schemat

```text
        [ Backbone ]
             |
     +-------+-------+
     |       |       |
   Head1   Head2   HeadN
```

## Matematyka

$$
y_t = f_t(h_{shared})
$$

## Funkcja straty

$$
\mathcal{L} = \sum_{t=1}^{T} \lambda_t \mathcal{L}_t
$$

## Kluczowe elementy

* **Shared Backbone**

  * wspólne reprezentacje
  * redukcja overfittingu

* **Task-Specific Heads**

  * każda głowica = osobny MLP
  * specjalizacja pod zadanie

* **Transfer wiedzy**

  * wykorzystanie korelacji między zadaniami

## Wlasciwosci 

* lepsza generalizacja
* efektywność parametrów
* konflikty gradientów
* potrzeba ważenia strat








---

# 4. Model: ADMET Hybrid Model

##  A. Architektura Modelu
W projekcie zaimplementowano hybrydową architekturę sieci neuronowej, która łączy uczenie geometryczne z klasyczną cheminformatyką:

1.  **Noga Grafowa (GIN):** Wykorzystuje sieć **Graph Isomorphism Network**. Cząsteczka traktowana jest jako graf ($G = (V, E)$), co pozwala na ekstrakcję cech topologicznych i strukturalnych bezpośrednio z atomów i wiązań.
2.  **Noga Wektorowa (MLP):** Przetwarza wejściowe dane tabelaryczne, takie jak fingerprinty molekularne (Morgan Fingerprints) oraz deskryptory fizykochemiczne RDKit.
3.  **Fuzja i Multi-Head:** Cechy z obu ścieżek są łączone w jeden wspólny wektor (embedding). Następnie sieć rozdziela się na niezależne "głowice" (heads), z których każda odpowiada za predykcję konkretnego endpointu (np. barierę krew-mózg czy toksyczność).

---




## B. Szczegolowe komponenty modelu:
* **Noga Grafowa (GIN Branch):** Wykorzystuje warstwy `GINConv` (Graph Isomorphism Network) do przetwarzania grafowej struktury molekuły.
    * **Agregacja:** Stosuje funkcję `global_add_pool`, która sumuje cechy atomów w celu uzyskania globalnego embeddingu cząsteczki.
    * **Stabilizacja:** Każda warstwa konwolucyjna wspierana jest przez wewnętrzne sieci MLP z `BatchNorm1d`.
* **Noga Wektorowa (Descriptor Branch):** Przetwarza dane stałe (fingerprinty Morgana lub deskryptory RDKit) za pomocą dedykowanej sieci MLP.
* **Moduł Fuzji (Fusion Layer):** Łączy wektory cech z obu ścieżek (`Concatenation`) i poddaje je normalizacji w `fusion_bn`.
* **Głowice Predykcyjne (Task-Specific Heads):** Zbiór niezależnych warstw liniowych, z których każda odpowiada za predykcję konkretnego parametru ADMET.




**Uwaga, Dlaczego suma?** W przeciwieństwie do uśredniania, suma pozwala sieci rozróżnić, czy atom jest połączony z dwoma atomami tlenu, czy tylko z jednym. Jest to kluczowe dla odróżnienia różnych grup chemicznych.





---

## C. Noga Wektorowa (MLP) 

Podczas gdy GIN skupia się na lokalnej topologii grafu, **Noga Wektorowa** odpowiada za przetwarzanie globalnych cech fizykochemicznych cząsteczki. Wykorzystuje ona architekturę typu **Multi-Layer Perceptron (MLP)** do analizy danych tabelarycznych.

### Przetwarzane dane:
* **Morgan Fingerprints:** Kodują obecność konkretnych podstruktur chemicznych w formie bitowej (domyślnie wektor o wymiarze 1024).
* **Deskryptory RDKit:** 217 konkretnych wartości liczbowych, takich jak Masa Cząsteczkowa, LogP (lipofilowość), TPSA czy liczba wiązań rotowalnych.

---

## D. Mechanizm Fuzji (Concatenation)
Cechy strukturalne (`g_emb`) oraz wektorowe (`v_emb`) są łączone w jeden długi wektor reprezentujący kompletną wiedzę o cząsteczce. 
* **Synergia:** Dzięki fuzji model wie nie tylko, jakie grupy funkcyjne ma cząsteczka (GIN), ale też jaki jest ich globalny wpływ na jej polarność czy rozpuszczalność (MLP).
* **Fusion BatchNorm:** Połączony wektor przechodzi przez dodatkową normalizację przed wejściem do głowic predykcyjnych, co zapewnia stabilność gradientu w podejściu Multi-Task.


---


## E. Multi-Head Architecture (Podejście MTL)
Zamiast jednego wyjścia, model posiada `nn.ModuleList` składający się z $N$ niezależnych głowic predykcyjnych.
* **Współdzielony kręgosłup (Shared Backbone):** Wszystkie głowice uczą te same warstwy grafowe i wektorowe. Dzięki temu wiedza o tym, co czyni cząsteczkę "trudną do wchłonięcia", jest przenoszona między zadaniami.
* **Dedykowane MLP:** Każda głowica (Head) to mini-sieć neuronowa (Linear -> ReLU -> Linear), która uczy się interpretować wspólne cechy pod kątem konkretnego parametru (np. toksyczności hERG vs metabolizmu CYP3A4).
* **Elastyczność:** Architektura pozwala na łatwe dodawanie nowych zadań bez konieczności redefiniowania całego modelu.






---


## F. Schemat Przepływu Danych
```text
      [ Cząsteczka SMILES ]
               |
      +--------+--------+
      |                 |
 [ Dane Grafowe ]  [ Dane Wektorowe ]
 (Atomy/Wiązania)  (Fingerprinty/RDKit)
      |                 |
      V                 V
 [ Noga GIN ]      [ Noga MLP ]
 (Graph Isom. Net) (Warstwy Gęste)
      |                 |
      +------> F U Z J A <------+
               |
      [ Wspólny Reprezentant ]
               |
      +--------+--------+
      |                 |
 [Głowica 1]  ...  [Głowica N]
 (Zadanie 1)       (Zadanie N)

```




## G. Architektura Matematyczna Całego Modelu


Cały model można zapisać jako kompozycję funkcji:

$$
h_g = f_{GIN}(G)
$$

$$
h_v = f_{MLP}(x)
$$

$$
h = \text{Concat}(h_g, h_v)
$$

$$
y_t = f_t(h)
$$

gdzie:
- $G$ – graf molekuły  
- $x$ – deskryptory / fingerprinty  
- $h_g$ – embedding grafowy  
- $h_v$ – embedding wektorowy  



---


## H. Mechanizm Multi-Task vs Single-Task
Model jest elastyczny i może pracować w dwóch trybach:
* **Multi-Task (MTL):** Posiada osobne "głowice" (Heads) dla każdego zadania. Wszystkie zadania dzielą wspólny kręgosłup (backbone), co wymusza naukę uniwersalnych cech chemicznych.
* **Single-Task (STL):** Poprzez parametr `single_task_idx`, model może zostać zainicjalizowany do optymalizacji pod tylko jeden, konkretny task.


---


## J. Funkcja Straty
Ze względu na to, że bazy danych ADMET często posiadają braki (dana cząsteczka nie była badana pod każdym kątem), zaimplementowano `MaskedBCELoss`. Funkcja ta:
1.  Identyfikuje wartości `NaN` w etykietach.
2.  Tworzy maskę binarną.
3.  Oblicza błąd tylko dla istniejących danych, co zapobiega zaburzeniu gradientów przez brakujące informacje.

---





# 5. Config / Kontroler 

Plik `config.py` to kontroler  eksperymentów. Zamiast szukać ustawień głęboko w kodzie modelu, wszystkie najważniejsze parametry zmieniamy w jednym miejscu. Pozwala to na błyskawiczne przełączanie się między różnymi wersjami modelu.


---
## A. Wybór zadań (Co model ma przewidywać?)
Dostepne jest definiowanie dowolnych list zadań oraz kombinacji datasetow.

W konfiguracji zdefiniowano różne listy zadań (`tasks`). Możesz kazać modelowi uczyć się wszystkiego naraz lub skupić się na konkretnych grupach.
My sie skupilismy na tych zadaniach :

* **Klaster Metaboliczny:** Skupienie na tym, jak organizm rozkłada leki (enzymy CYP).
* **Bariery i Transport:** Sprawdzanie, czy lek przejdzie przez jelita lub barierę krew-mózg.
* **Toksyczność:** Przewidywanie, czy związek nie uszkodzi serca lub wątroby.

---
## B. Typy uzywanych danych wejsciowych
Możesz zdecydować, jakich informacji model ma używać do nauki:
* `use_graph = True`: Model analizuje strukturę jak graf (połączenia atomów).
* `use_morgan = True`: Model patrzy na cząsteczkę przez pryzmat konkretnych fragmentów (fingerprinty).
* `use_rdkit = True`: Model dostaje gotowe dane fizykochemiczne (np. masa, polarność).

---

## C. Parametry nauki 
Tu ustawiasz sposob treningu i hiperparametry:
* **Learning Rate (`lr`):** Jak szybko model ma korygować swoje błędy.
* **Epochs:** Ile razy model ma przejrzeć cały zbiór danych podczas treningu.
* **Batch Size:**  Ile cząsteczek model analizuje za jednym razem, zanim zaktualizuje swoją wiedzę.
* **Device:** Na jakiej architekturze trenujemy model



---


# 6. Zarządzanie Danymi i Pipeline Przetwarzania

Projekt opiera się na zaawansowanym systemie pozyskiwania i przetwarzania danych molekularnych, który integruje zewnętrzne bazy danych z lokalnym systemem cache’owania.

## A. Źródło danych: Therapeutics Data Commons (TDC)
Wszystkie dane chemiczne pobierane są automatycznie za pomocą API **TDC**. Projekt korzysta z dwóch głównych grup datasetów:
* **ADME:** Parametry dotyczące wchłaniania, dystrybucji, metabolizmu i wydalania (np. `caco2_wang`, `hia_hou`, `cyp3a4_veith`).
* **Toxicity:** Dane dotyczące toksyczności związków (np. `herg`, `ames`, `dili`).

## B. Proces Ładowania i Łączenia (Data Merging)
Ponieważ model działa w trybie **Multi-Task**, dane z różnych źródeł muszą zostać zsynchronizowane:
1.  **Pobieranie:** Skrypt pobiera wybrane w `Config` zadania.
2.  **Łączenie (Outer Join):** Dane są łączone po kluczu **SMILES** (uproszczony zapis struktury cząsteczki). Jeśli dana cząsteczka była badana pod kątem wchłaniania, ale nie toksyczności, brakujące wartości są oznaczane jako `NaN`.
3.  **Obsługa Splitów:** Projekt rygorystycznie przestrzega podziału danych dostarczanego przez TDC na zbiory: **Train** (treningowy), **Valid** (walidacyjny) oraz **Test** (testowy). Gwarantuje to rzetelność wyników i możliwość ich porównania z literaturą naukową.

## C. Ekstrakcja Cech (Feature Engineering)
Surowy zapis SMILES jest konwertowany na trzy typy reprezentacji:

###  Reprezentacja Grafowa (Molecular Graphs)
Cząsteczka jest traktowana jako graf nieskierowany, gdzie atomy są węzłami, a wiązania krawędziami.

* **Cechy Węzłów (9 atrybutów atomowych):** Dla każdego atomu generowany jest wektor cech zawierający:
    1. **Liczba atomowa:** Identyfikacja pierwiastka.
    2. **Stopień (Degree):** Liczba bezpośrednich sąsiadów w grafie.
    3. **Ładunek formalny:** Stan elektryczny atomu.
    4. **Aromatyczność:** Czy atom jest częścią układu aromatycznego.
    5. **Wartościowość implikowana:** Liczba domyślnych wiązań z wodorem.
    6. **Hybrydyzacja:** Typ hybrydyzacji orbitalnej (zmapowany na liczby całkowite).
    7. **Elektrony rodnikowe:** Liczba niesparowanych elektronów.
    8. **Masa atomowa:** Skalowana (mnożnik 0.01) dla poprawy stabilności numerycznej.
    9. **IsInRing:** Czy atom znajduje się w dowolnym pierścieniu.


* **Krawędzie (Bonds):** Wiązania są mapowane jako pary indeksów atomów. System automatycznie dodaje krawędzie w obie strony (np. $A \to B$ i $B \to A$), co jest standardem w sieciach typu GNN.


###  Morgan Fingerprints (ECFP4)
Jest to reprezentacja oparta na algorytmie kołowym (Circular Fingerprints), która opisuje obecność specyficznych podstruktur chemicznych.

* **Mechanizm:** Wykorzystujemy generator Morgana z promieniem (radius) równym 2, co odpowiada standardowi **ECFP4**. Algorytm analizuje otoczenie każdego atomu w promieniu dwóch wiązań.
* **Wymiar:** Wynikiem jest rzadki wektor o stałej długości **1024 bitów**. Każdy bit informuje o obecności (1) lub braku (0) konkretnego fragmentu strukturalnego w cząsteczce.

###  Deskryptory Fizykochemiczne RDKit
Zestaw **217 parametrów** obliczanych bezpośrednio z pełnej struktury cząsteczki, dostarczający modelowi "twardych" danych numerycznych.

* **Zakres danych:** Obejmuje m.in. masę cząsteczkową, współczynnik podziału n-oktanol/woda ($LogP$), polarną powierzchnię cząsteczki ($TPSA$), liczbę wiązań rotowalnych oraz liczbę donorów/akceptorów wiązań wodorowych.
* **Czyszczenie i stabilizacja:** * Wszystkie wartości ekstremalne lub błędne (`NaN`, `Inf`) są zastępowane zerami lub przycinane (**clipping**) do zakresu $[-10^6, 10^6]$. Zapobiega to eksplozji gradientu w sieci neuronowej.
    * **Normalizacja:** Deskryptory przechodzą przez `StandardScaler`, który centruje dane wokół zera i skaluje je do jednostkowej wariancji. Jest to kluczowe dla "Nogi Wektorowej" (MLP), aby deskryptory o dużych wartościach (np. masa $\approx 500$) nie dominowały nad mniejszymi (np. ładunek $\approx 0.1$).

## D. Strategia Master Cache
Przetwarzanie tysięcy cząsteczek na grafy i deskryptory jest czasochłonne. Kod implementuje system **Master Cache**:

## E. Przygotowanie do Treningu (Loaders)
Gotowe obiekty trafiają do `GNNLoader`, który grupuje cząsteczki w paczki (**Batches**) o rozmiarze zdefiniowanym w konfiguracji (domyślnie 64). System automatycznie dba o to, by dane grafowe były poprawnie przesyłane na kartę graficzną (GPU/CUDA) podczas treningu.



---

# 7. Proces Treningowy i Ewaluacja

Sercem projektu jest potok treningowy (pipeline), który pozwala na rzetelne porównanie różnych architektur i metod uczenia. System został zaprojektowany tak, aby automatycznie przeprowadzać pełny benchmarking.

## A. Pętla Treningowa Multi-Task (MTL)
Główna funkcja treningowa `train_mtl_and_visualize` odpowiada za naukę modelu na wielu zadaniach jednocześnie:
* **Optymalizacja:** Wykorzystuje optymalizator **Adam** z początkowym współczynnikiem uczenia $lr=0.001$.
* **Obsługa NaN:** Dzięki `MaskedBCELoss`, gradienty są obliczane wyłącznie dla zadań, które posiadają etykiety w danej próbce.
* **Scheduler:** System monitoruje $AUROC$ na zbiorze walidacyjnym i automatycznie zmniejsza $lr$ o połowę (`ReduceLROnPlateau`), gdy postęp wyhamowuje.
* **Early Stopping:** Trening jest przerywany, jeśli model nie wykazuje poprawy przez określoną liczbę epok (patience), co zapobiega overfittingowi.

## B. Architektura Eksperymentu 
(`main` i `run_experiments`)
Kod wspiera dwa główne tryby uruchomienia:

* **Tryb `main()`:** 1. Pobiera dane i przygotowuje cache.
    2. Trenuje klasyczny baseline **XGBoost** (Single-Task).
    3. Trenuje model **STL GNN** dla każdego zadania z osobna.
    4. Trenuje finalny model **MTL GNN Hybrid**.
    5. Generuje raporty porównawcze i wykresy końcowe.


* **Tryb `run_experiments()` (Ablation Study):** Uruchamia serię 6 różnych kombinacji wejść (tylko graf, graf + RDKit, hybryda itp.), aby sprawdzić, który zestaw cech jest najbardziej istotny dla predykcji ADMET.

## C. Metryki i Benchmarking
Model jest oceniany pod kątem zdolności do generalizacji na danych testowych przy użyciu następujących narzędzi:
* **AUROC:** Główna metryka dla zadań klasyfikacji ADMET.
* **Macierze Pomyłek (Confusion Matrices):** Generowane osobno dla każdego zadania.
* **Korelacja zadań:** Analiza wizualna pokazująca, jak model przewiduje powiązane ze sobą parametry.

---
### AUROC
* **Dlaczego AUROC?:** Jest to metryka odporna na niezbalansowanie klas (częste w ADMET, gdzie np. tylko 5% związków jest toksycznych).
* **Interpretacja:** Wynik 0.5 oznacza zgadywanie losowe, a 1.0 idealną predykcję.
* **Zbiory rzadkie:** Funkcja `evaluate_per_task` automatycznie pomija zadania, w których zbiór testowy zawiera tylko jedną klasę, co zapobiega błędom obliczeniowym.



### Macierz Korelacji 
* **Korelacja Etykiet (Spearman):** Obliczamy ją na surowych danych, aby wykryć biologiczne powiązania między endpointami (np. jak silnie hamowanie jednego enzymu CYP wiąże się z innym).
* **Korelacja Predykcji (Pearson):** Obliczamy ją na wynikach modelu. Pozwala sprawdzić, czy model "nauczył się" tych samych zależności, które występują w naturze. Jeśli dwa zadania mają wysoką korelację predykcji, model wykorzystuje wspólne cechy chemiczne do ich rozwiązania.

### Macierz Rzadkości (Data Sparsity)
* **Co zawiera:** Pokazuje obecność etykiet dla każdej cząsteczki w zbiorze danych.
* **Cel:** Wizualizacja "dziur" w danych (wartości `NaN`), co uzasadnia konieczność stosowania mechanizmu maskowania w funkcji straty.


---

## D. Porównanie Modeli
Funkcja `plot_model_comparison` tworzy wykresy słupkowe zestawiające wyniki **MTL GNN**, **STL GNN** oraz **XGBoost**. Pozwala to na bezpośrednią weryfikację hipotezy o zysku z uczenia wielozadaniowego dla każdego parametru ADMET z osobna.


---


# 8. Problemy i Ograniczenia Modelu

## A. Problem konfliktu gradientów (MTL)

W Multi-Task Learning gradienty mogą być sprzeczne:

$$
\nabla \mathcal{L}_i \cdot \nabla \mathcal{L}_j < 0
$$

Możliwe rozwiązania:
* ważenie strat ($\lambda_t$)
* task balancing


## B. Ograniczenia 

* brak pełnej informacji 3D o czasteczkach chemicznych
* brak jawnego modelowania dynamiki molekularnej  
* zależność od jakości danych TDC  
* sparsity (dużo NaN)

---




# 9. Planowane Usprawnienia

Projekt jest stale rozwijany. Poniżej przedstawiamy kluczowe funkcjonalności, które mogą znacząco podnieść jakość predykcji ADMET w przyszłości:

## A. Global Virtual Node (Wspólny Wierzchołek)
Obecnie informacje w grafie przepływają tylko między sąsiadami (wiadomości lokalne). Dodanie **Globalnego Wirtualnego Węzła** połączonego ze wszystkimi atomami w cząsteczce pozwoliłoby na:
* Błyskawiczny przepływ informacji między odległymi wierzcholkami.
* Lepszą reprezentację globalnego kontekstu cząsteczki przez model GNN.

## B. Wykorzystanie Atrybutów Krawędzi (Bond Features)
Aktualna implementacja bierze pod uwagę połączenia między atomami, ale nie rozróżnia ich typu. Wprowadzenie cech krawędzi (edge features) pozwoliłoby modelowi rozróżniać:
* Rząd wiązania (pojedyncze, podwójne, potrójne).
* Wiązania aromatyczne.

## C. Mechanizmy Uwagi (Attention Mechanisms)
Zastąpienie standardowej agregacji mechanizmem **Graph Attention (GAT)**, co pozwoliłoby modelowi "skupić się" na najwazniejszych atomach czy wiazaniach

---




