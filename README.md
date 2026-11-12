# Ewolucja moralności sztucznego społeczeństwa

[![Testy](https://github.com/wiktoriakawa/ewolucja-moralnosci-praca-inzynierska/actions/workflows/testy.yml/badge.svg)](https://github.com/wiktoriakawa/ewolucja-moralnosci-praca-inzynierska/actions/workflows/testy.yml)

Projekt pracy inżynierskiej dotyczący symulacji ewolucji zachowań
prospołecznych i egoistycznych w populacji agentów.

Agenci żyją na dwuwymiarowej planszy, poruszają się, zbierają zasoby oraz
wchodzą ze sobą w interakcje. Ich zachowanie jest sterowane przez sieci
neuronowe kodowane w genomach. Selekcja i mutacja pozwalają obserwować zmiany
zachowań w kolejnych pokoleniach.

## Status projektu

Projekt znajduje się na etapie przygotowania kamienia milowego KM3,
obejmującego rozwój modelu i kalibrację jego parametrów.

Obecnie zaimplementowane są:

- reprezentacja stanu symulacji;
- plansza, odrost i zbieranie zasobów;
- rozmieszczanie, ruch i obserwacje agentów;
- liniowa sieć neuronowa i wybór akcji (argmax albo softmax);
- ewolucja: pokolenie zerowe, kwalifikacja, mutacja i nowe pokolenie;
- wczytywanie konfiguracji z pliku TOML;
- dzielenie się zasobami, kradzież i metabolizm;
- pojedynczy krok i przebieg wielu pokoleń;
- rejestrowanie dylematów, akcji i śmierci agentów;
- zapis konfiguracji i metadanych do JSON;
- przyrostowy zapis wyników do plików CSV;
- konfiguracje przebiegów pilotażowych;
- uruchamianie przebiegu jednym poleceniem ze ścieżką do konfiguracji;
- kontrola jakości danych i sprawdzanie powtarzalności przebiegów;
- podstawowa analiza wyników i eksploracja danych pilotażowych;
- automatyczne tworzenie tabel i wykresów;
- testy jednostkowe i integracyjne powyższych elementów.

Najważniejsze elementy planowane w KM3:

- rozszerzenie walidacji konfiguracji;
- uruchamianie serii przebiegów;
- rozbudowa sieci neuronowej i pomiarów;
- kalibracja parametrów modelu.

## Wymagania

- Python 3.12 lub nowszy;
- NumPy;
- pandas do kontroli jakości i analizy danych;
- Matplotlib do tworzenia wykresów;
- pytest do uruchamiania testów.

Projekt został lokalnie sprawdzony z użyciem:

- Python 3.13.2 i 3.14.5;
- NumPy 2.5.4;
- pandas 3.1.0;
- Matplotlib 3.11.2;
- pytest 9.1.1.

## Przygotowanie środowiska

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Zmiana polityki wykonywania dotyczy tylko bieżącego okna PowerShell i przestaje
obowiązywać po jego zamknięciu.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Uruchamianie przebiegu

Program przyjmuje ścieżkę do pliku konfiguracyjnego TOML. Katalog wyników
jest określony przez klucz `zapis.katalog` w tej konfiguracji.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m moralnosc przebieg konfiguracje/pilotaz/P1_z101.toml
```

Linux:

```bash
PYTHONPATH=src python -m moralnosc przebieg konfiguracje/pilotaz/P1_z101.toml
```

Po zakończeniu powstają plik `konfiguracja.json` oraz pliki
`pokolenia.csv`, `agenci.csv` i `dylematy.csv`.

## Analiza wyników

Katalog `analiza` zawiera kontrolę jakości pojedynczego przebiegu oraz
eksplorację danych pilotażowych. Instrukcje uruchomienia i opis tworzonych
plików znajdują się w pliku [`analiza/README.md`](analiza/README.md).

## Uruchamianie testów

Projekt korzysta z układu katalogów `src`, dlatego przed uruchomieniem testów
należy dodać ten katalog do ścieżki importowania.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Linux:

```bash
PYTHONPATH=src python -m pytest -q
```

Pełny zestaw testów jest również uruchamiany automatycznie przez GitHub Actions
dla pull requestów do `main` i po każdym pushu do `main`. Workflow sprawdza
projekt na Pythonie 3.12 oraz 3.14; można go też uruchomić ręcznie z zakładki
**Actions** na GitHubie.

## Struktura repozytorium

```text
src/moralnosc/
├── __init__.py
├── __main__.py      # interfejs uruchamiania przebiegu z pliku TOML
├── stan.py          # struktura stanu symulacji i wspólne stałe
├── srodowisko.py    # plansza, zasoby pól, odrost i zbieranie
├── populacja.py     # rozmieszczanie, ruch i obserwacje agentów
├── siec.py          # sieć neuronowa i wybór akcji
├── ewolucja.py      # pokolenie zerowe, kwalifikacja, mutacja, nowe pokolenie
├── konfiguracja.py  # wczytywanie i minimalna walidacja pliku TOML
├── interakcje.py    # dzielenie, kradzież i metabolizm
├── petla.py         # akcje, kroki, pokolenia i pełny przebieg symulacji
├── pomiar.py        # rejestrowanie pomiarów jednego pokolenia
└── zapis.py         # konfiguracja JSON i przyrostowy zapis wyników CSV

konfiguracje/
├── bazowa.toml        # bazowa konfiguracja modelu
├── generuj_pilotaz.py # generowanie konfiguracji P1–P3
└── pilotaz/           # konfiguracje przebiegów pilotażowych

analiza/
├── jakosc.py          # kontrola jakości i analiza pojedynczego przebiegu
├── eksploracja.py     # eksploracja wyników całego pilotażu
└── README.md          # instrukcja uruchamiania analizy

tests/
├── pomocnicze.py    # sprawdzanie niezmienników stanu
└── test_<moduł>.py  # testy jednostkowe i integracyjne
```

## Konwencje

- Współrzędne mają postać `(wiersz, kolumna)`.
- Plansza ma kształt `H × W`.
- Kierunki są kodowane jako: `0` — północ, `1` — wschód, `2` — południe,
  `3` — zachód.
- Nazwy techniczne w kodzie są zapisywane po polsku bez znaków
  diakrytycznych.
- Komentarze i dokumentacja są zapisywane poprawną polszczyzną.
- Funkcje wykonujące losowania otrzymują jawnie obiekt
  `numpy.random.Generator`.
- Stan planszy i populacji jest przechowywany w tablicach NumPy.

## Zasady wprowadzania zmian

- Każdy moduł powstaje na osobnej gałęzi i trafia do osobnego pull requestu.
- Pull request jest przeglądany przez drugą osobę przed scaleniem.
- Przed scaleniem wszystkie testy dotyczące zmiany muszą przechodzić.
- Zmiany nie są wprowadzane bezpośrednio do gałęzi `main`.
- Pull request powinien dotyczyć jednego spójnego zagadnienia.

## Autorzy

- Wiktoria Kawa
- Wojciech Łukaszyk

Projekt powstaje w ramach pracy inżynierskiej na Wydziale Matematyki i Nauk
Informacyjnych Politechniki Warszawskiej.
