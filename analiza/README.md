# Podstawowa analiza wyników

Moduł `jakosc.py` sprawdza jakość danych jednego przebiegu symulacji,
oblicza podstawowe miary pokoleń i tworzy wykresy. Nie sprawdza założeń
konkretnych przebiegów pilotażowych P1–P3.

## Dane wejściowe

Należy wskazać katalog jednego przebiegu zawierający cztery pliki:

```text
konfiguracja.json
pokolenia.csv
agenci.csv
dylematy.csv
```

Przed uruchomieniem trzeba zainstalować zależności projektu i aktywować
środowisko wirtualne. Polecenia należy wykonywać z katalogu głównego
repozytorium.

## Uruchomienie

Przykład analizy jednego przebiegu w PowerShellu:

```powershell
$env:PYTHONPATH = "src"
python -m analiza.jakosc wyniki/P2_niedobor_z201 `
    --wyjscie wyniki_analizy
```

Argument `--wyjscie` wskazuje katalog bazowy i jest opcjonalny. Bez niego
zostanie użyty katalog `wyniki_analizy`. Wewnątrz skrypt automatycznie tworzy
osobny katalog zawierający nazwę przebiegu oraz datę i czas analizy, np.:

```text
wyniki_analizy/P2_niedobor_z201_2026-11-02_15-30-12/
```

Dzięki temu kolejne uruchomienia nie nadpisują wcześniejszych wyników.

## Wyniki

Skrypt tworzy:

- `kontrola_jakosci.csv` – wynik każdej kontroli wraz ze szczegółami;
- `pokolenia_analiza.csv` – dane pokoleń rozszerzone o `M_pop` i liczbę
  rodziców;
- `przezywalnosc.png` – przeżywalność w kolejnych pokoleniach;
- `moralnosc_populacji.png` – wartość `M_pop` w kolejnych pokoleniach;
- `liczba_rodzicow.png` – liczbę rodziców w kolejnych pokoleniach.

Moralność populacji jest liczona według wzoru:

```text
M_pop = (S - K) / D
```

Gdy w pokoleniu nie wystąpił żaden dylemat (`D = 0`), `M_pop` otrzymuje
wartość `NaN`.

## Kod zakończenia

Program zwraca kod `0`, jeżeli wszystkie kontrole jakości przeszły, oraz
kod `1`, jeżeli wykryto niezgodność. Szczegóły znajdują się wtedy w pliku
`kontrola_jakosci.csv`.
