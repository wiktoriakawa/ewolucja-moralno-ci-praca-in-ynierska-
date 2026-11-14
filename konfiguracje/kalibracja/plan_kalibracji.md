# Plan i macierz eksperymentów kalibracyjnych KM3 (issue #27)

**Wersja 1.0 (14.11.2026).** Kryteria wyboru ustalone **przed** uruchomieniem pierwszej serii (K1) – kryterium akceptacji #27. Plan przygotowany z pomocą AI (Claude) na podstawie ustaleń zespołu; decyzje metodyczne: Wiktoria (A/R metodyki eksperymentów), akceptacja Wojtka przez zatwierdzenie PR. Tabele planu trafiają do dokumentu KM3 (p. 4.2, dodatek B). Serie K1, K3 i K4 mają pliki definicji obok (`K1.toml`, `K3.toml`, `K4.toml`); K2 i K5 przygotowuje Wojtek.

## 1. Cel

Dobrać wartości referencyjne parametrów modelu z warstwą ukrytą (D2) tak, aby w eksperymentach docelowych (CA-1–CA-4, Lab04): (a) decyzje agentów były zróżnicowane, a nie prawie deterministyczne (problem z P1, Lab02 p. 4.4.3); (b) selekcja działała bez częstego wymierania i bez zaniku kwalifikacji (P2, Lab02 p. 4.4.4); (c) mutacja utrzymywała różnorodność bez niszczenia odziedziczonych strategii (D11); (d) znany był koszt czasu i danych (NF-06, NF-07).

## 2. Parametry stałe (wszystkie serie)

Konfiguracja bazowa `konfiguracje/bazowa.toml` (Lab02, dodatek A): plansza 64×64, N = 200, 300 kroków na pokolenie, pojemność pól z [1, 5], m = 1, ruch 0, s = 2, c_k = 1, p = 0,5, x = 5, tryb softmax, E0 = 50, w_max = 4, R = 2.

Zmiany wspólne dla KM3: `siec.h = 8` (rejestr decyzji D2; opcjonalna K1b sprawdza h ∈ {0, 4, 8, 16}). Każda seria zmienia tylko parametry z kolumny „Zmieniane”; pozostałe biorą wartości wybrane we wcześniejszych seriach (kolejność K1 → K2 → K3 → K4 → K5).

## 3. Serie

| Seria | Issue, osoba | Pytanie | Zmieniane (wartości) | Stałe (poza bazową) | Pokoleń × kroków | Ziarna | Przebiegów | Miary | Kryterium wyboru (ustalone przed analizą) |
|---|---|---|---|---|---|---|---|---|---|
| K1 | #28, Wiktoria | Czy softmax daje zróżnicowane decyzje? | skala ∈ {5, 10, 50} × τ ∈ {0,5; 1; 2} (`K1.toml`) | h = 8, α = 1,0, T = 50 | 10 × 300 | 401, 402, 403 | 27 | koncentracja decyzji agenta w pokoleniu 0 (mediana udziału najczęstszej akcji; P1: 0,89), entropia akcji populacji (pokolenie 0 i ostatnie 5), odsetek agentów z D ≥ 5, mediana D, udział zbierania na końcu, sd między ziarnami | (1) brak wymarcia w ≥ 2 z 3 ziaren; (2) mediana D w pokoleniu 0 ≥ 5; spośród spełniających – najniższa koncentracja w pokoleniu 0, a przy remisie (różnica < sd) wyższa entropia końcowa; τ = 1 i skala = 5 zostają, jeśli żaden wariant nie poprawia koncentracji o więcej niż sd |
| K1b (opcjonalna) | #28, Wiktoria | Ile neuronów ukrytych? (karta tematu: „analiza efektywności różnych konfiguracji neuronowych”) | h ∈ {0, 4, 8, 16} | skala, τ z K1; α = 1,0; T = 50 | 10 × 300 | 401–403 | 12 | jak K1 + czas na pokolenie | h = 8, chyba że inne h daje niższą koncentrację o więcej niż sd przy czasie ≤ 2× h = 8; bez K1b → h = 8 z D2, porównanie w Lab04 |
| K2 | #29, Wojtek | Gdzie selekcja działa bez wymierania? | T ∈ {30, 50, 70} × α ∈ {0,85; 1,0} (propozycja – ustala Wojtek w #29) | skala, τ, h z K1; E0 = 50 | 30 × 300 | 501, 502, 503 | 18 | przeżywalność, zakwalifikowani, pokolenia z wariantem awaryjnym, wymarcia | ustala Wojtek w #29 (np. 10–50 % zakwalifikowanych, wariant awaryjny < 10 % pokoleń, brak wymarcia) |
| K3 | #30, Wiktoria | Czy mutacja utrzymuje różnorodność? | r ∈ {0,5/L; 1/L; 3/L} × σ ∈ {0,1; 0,2; 0,4} (L = 120 przy h = 8, n_we = 5; `K3.toml`) | skala, τ z K1; T, α z K2 | 50 × 300 | 601, 602, 603 | 27 | różnorodność ρ (średnia odległość euklidesowa genomów) w pokoleniu 0 i ostatnim, iloraz ρ_koniec/ρ_0; korelacja M_wyg rodzic–potomek (pary z D ≥ 5) i liczba par; M_pop; wariant awaryjny | (1) brak wymarcia w ≥ 2 z 3 ziaren; (2) korelacja rodzic–potomek > 0 w każdym ziarnie przy ≥ 30 parach; (3) ρ_koniec/ρ_0 ≥ 0,25 po 50 pokoleniach; spośród spełniających – najwyższa korelacja; przy remisie (< sd) wartości bliższe bazowym (1/L, 0,2) |
| K4 | #30, Wiktoria | Jak lokalność narodzin wpływa na wyniki? | R ∈ {1, 2, 5, 63} (63 = cała plansza dla każdego rodzica; `K4.toml`) | skala, τ z K1; T, α z K2; (r, σ) z K3 | 50 × 300 | 701, 702, 703 | 12 | udział dylematów między rodzeństwem (pokrewieństwo), mediana D (kontakty), M_pop, różnorodność końcowa, korelacja rodzic–potomek | R = 2 (D12) zostaje, chyba że korelacja rodzic–potomek przy R = 63 (narodziny globalne) jest niższa o więcej niż sd (wtedy część „odziedziczalności” to wspólne środowisko – opisać, R referencyjne do decyzji); wynik K4 służy głównie CA-4 |
| K5 | #31, Wojtek | Koszt konfiguracji referencyjnej | – (konfiguracja referencyjna) | wszystkie z K1–K4 | 500 × 300 (NF-06: 200 agentów × 500 pokoleń) | 801 | 1 (+1 powtórzenie NF-03) | czas, szczytowa RAM, rozmiar katalogu, kontrola jakości, NF-03 | NF-06: ≤ 30 min; NF-07: ≤ 8 GB RAM, ≤ 1 GB danych |

Razem (bez K2 i K5): 27 + 12 + 27 + 12 = 78 przebiegów.

## 4. Ziarna i powtórzenia

- 3 ziarna na wariant (gdyby harmonogram się opóźnił – 2, z adnotacją w dokumencie); ziarna rozłączne z dotychczasowymi (próba przeżycia 1–5; pilotaż 101–105, 201–203, 301) i między seriami (4xx, 5xx, 6xx, 7xx, 8xx), żeby żaden wynik nie był „dopasowany” do ziarna z innej serii.
- Wariant × ziarno = jeden przebieg w osobnym katalogu `wyniki/<seria>/<seria>_<wariant>_z<ziarno>/` (np. `wyniki/K1/K1_s10_t0.5_z402/`) z `konfiguracja.json` (konfiguracja, ziarno, commit – wersja kodu).
- Agregacja: średnia po ziarnach ± odchylenie standardowe (w tabelach), linie ziaren + średnia (na wykresach, jak w Lab02 rys. przeżywalności). Niepewność = rozrzut między ziarnami; przy 3 ziarnach nie robimy testów istotności, tylko porównujemy różnice z sd.
- Pokolenia: K1 – 10 (zróżnicowanie decyzji widać w pokoleniu 0 i w pierwszych pokoleniach adaptacji); K3, K4 – 50 (jak P2; różnorodność i odziedziczalność potrzebują wielu pokoleń); K5 – 500 (NF-06).

## 5. Przebiegi zakończone wymarciem

Przebieg, w którym wymarła populacja, liczy się jako „wymarcie” i wchodzi tylko do miar przeżycia; miary z ostatnich pokoleń liczymy z pozostałych ziaren i podajemy, z ilu. Wariant z wymarciem w ≥ 2 z 3 ziaren jest odrzucany. Reguła „wymarcie kończy przebieg” (Lab02) pozostaje (decyzja do rejestru decyzji v0.3).

## 6. Kontrola jakości

Każdy przebieg przed analizą przechodzi `analiza/jakosc.py` (po #26 – z kontrolą nowych kolumn); dwa przebiegi z tym samym ziarnem w K5 muszą dać identyczne CSV (NF-03, `--porownaj`). Przebiegi niezaliczone są wyłączane i wymienione w dodatku B.

## 7. Koszt (pomiary z 11.11.2026 i z serii K1 14.11.2026; Mac, Python 3.14.5, NumPy 2.5.4)

Czas na pokolenie: pomiar 11.11 (konfiguracja bazowa, α = 1,0, różnica czasu przebiegów 10 i 3 pokoleń) – h = 0 ok. 0,22 s, h = 8 ok. 0,33 s. Seria K1 (27 przebiegów × 10 pokoleń, h = 8) trwała 44 s od startu pierwszego do końca ostatniego przebiegu, czyli ok. 0,16 s na pokolenie razem z zapisem. Czas zależy od zachowania agentów, więc do szacunków bierzemy ostrożnie 0,33 s.

Dane: K1 zajęło 394 MB – średnio 14,6 MB na przebieg (7,9–24,7 MB zależnie od wariantu), czyli ok. 1,5 MB na pokolenie, trzy razy więcej niż szacunek z pomiaru przy h = 0 (0,5 MB). 99 % to `dylematy.csv` (wiersz na każdy krok agenta przed sąsiadem). Szacunki niżej przy h = 8: 0,33 s i 1,5 MB na pokolenie.

| Seria | Przebiegów × pokoleń | Czas | Dane |
|---|---|---|---|
| K1 | 27 × 10 | 44 s (pomiar) | 394 MB (pomiar) |
| K1b | 12 × 10 | ok. 1 min (h = 16 niemierzone) | ok. 180 MB |
| K3 | 27 × 50 | ok. 7,5 min | ok. 2,0 GB |
| K4 | 12 × 50 | ok. 3,5 min | ok. 0,9 GB |
| K5 | 1 × 500 (×2) | ok. 3 min (×2) | ok. 0,75 GB (×2) – blisko NF-07 (≤ 1 GB na przebieg) |

K1, K3 i K4 razem to ok. 12 min obliczeń i ok. 3,3 GB danych w `wyniki/` (poza Gitem) – mieści się w jednym wieczorze i na dysku; wąskim gardłem jest analiza i opis, nie obliczenia.

## 8. Wykonanie

Tryb serii (#23, w `main` od PR #41), z katalogu `kod/`:

```bash
PYTHONPATH=src python -m moralnosc seria konfiguracje/kalibracja/K1.toml
```

Stan przebiegów zapisuje `wyniki/<seria>/status.csv`; do analizy idą tylko przebiegi `zakonczony`, pozostałe wymieniamy w dodatku B. Commit, Python i NumPy każdego przebiegu są w jego `konfiguracja.json` (metadane). Każdą serię uruchamiamy raz, do pustego katalogu (po ponownym uruchomieniu przerwany przebieg dostaje status `pominiety` – poprawka zgłoszona w recenzji #38).

Przed K3 i K4 w ich plikach podmieniamy wartości oznaczone „do podmiany” na wybrane w K1 (skala, τ), K2 (T, α) i – dla K4 – K3 (r, σ); podmianę i jej źródło zapisujemy w opisie PR analizy (P6/P7). Wartości r w `K3.toml` są liczone dla L = 120 (h = 8); gdyby K1b zmieniła h, przeliczamy je dla nowego L = 6h + 8(h + 1).

Analiza: `python -m analiza.kalibracja <seria> wyniki/<seria> --wyjscie wyniki/analiza/<seria>` (tabele CSV i LaTeX, wykresy PDF o stałych nazwach).

## 9. Macierz przebiegów (dodatek B dokumentu)

Tabelę „wariant, ziarno, katalog, commit, status, kontrola jakości” składa `analiza/kalibracja.py` ze `status.csv` serii i metadanych `konfiguracja.json` przebiegów; uzupełniamy ją po wykonaniu serii.
