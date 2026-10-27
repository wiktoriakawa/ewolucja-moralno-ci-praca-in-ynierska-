# Ewolucja moralności sztucznego społeczeństwa

Projekt pracy inżynierskiej dotyczący symulacji ewolucji zachowań
prospołecznych i egoistycznych w populacji agentów.

Agenci żyją na dwuwymiarowej planszy, poruszają się, zbierają zasoby oraz
wchodzą ze sobą w interakcje. Ich zachowanie jest sterowane przez sieci
neuronowe kodowane w genomach. Selekcja i mutacja pozwalają obserwować zmiany
zachowań w kolejnych pokoleniach.

## Status projektu

Projekt znajduje się na etapie tworzenia prototypu Lab02.

Obecnie zaimplementowane są:

- reprezentacja stanu symulacji;
- tworzenie planszy i początkowych zasobów;
- odrost i zbieranie zasobów;
- losowe rozmieszczanie agentów;
- ruch agentów i mapa zajętości;
- obserwacje otoczenia;
- minimalne testy powyższych elementów.

Planowane elementy prototypu:

- sieć neuronowa i wybór akcji;
- ewolucja i tworzenie kolejnych pokoleń;
- konfiguracja przebiegów w formacie TOML;
- dzielenie się zasobami i kradzież;
- główna pętla symulacji;
- pomiar zachowań oraz zapis wyników do plików CSV.

## Wymagania

- Python 3.11 lub nowszy;
- NumPy;
- pytest do uruchamiania testów.

Projekt został lokalnie sprawdzony z użyciem:

- Python 3.13.2;
- NumPy 2.5.4;
- pytest 9.1.1.

Biblioteki pandas i Matplotlib będą potrzebne na późniejszym etapie do analizy
i wizualizacji wyników.

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

## Struktura repozytorium

```text
src/moralnosc/
├── __init__.py
├── stan.py          # struktura stanu symulacji i wspólne stałe
├── srodowisko.py    # plansza, zasoby pól, odrost i zbieranie
└── populacja.py     # rozmieszczanie, ruch i obserwacje agentów

tests/
├── pomocnicze.py
├── test_stan.py
├── test_srodowisko.py
└── test_populacja.py
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
