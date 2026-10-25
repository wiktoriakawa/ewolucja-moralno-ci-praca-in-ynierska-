"""Środowisko symulacji: plansza ze ścianami, zasoby pól, odrost i zbieranie"""

import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import Stan


def utworz_plansze(
    cfg: dict, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tworzy planszę: mapę pojemności pól i początkowe zasoby pól

    Argumenty:
cfg: konfiguracja wczytana z pliku TOML (słownik słowników), potrzebne klucze:
             cfg["plansza"]["H"], cfg["plansza"]["W"]  – liczba wierszy i kolumn,
             cfg["zasoby"]["c_min"], cfg["zasoby"]["c_max"] – zakres pojemności pól.
        rng: generator liczb losowych przebiegu (jeden na cały przebieg, NF-03).

    Zwraca:
        (pojemnosc, zasoby_pol) – dwie tablice float64 o kształcie H×W.
        Pojemność każdego pola jest losowana jednostajnie z [c_min, c_max];
        na starcie pola są pełne, czyli zasoby_pol mają te same wartości co
        pojemnosc (kontrakt, kwestia otwarta 4).
    """
    # Krok 1. Odczytaj H, W, c_min, c_max z cfg (cztery zmienne).

    # Krok 2. Wylosuj pojemność pól: liczby jednostajne z [c_min, c_max],
    #         kształt (H, W), z generatora `rng` – NIE z np.random.
    #         Do sprawdzenia w dokumentacji NumPy: metoda Generator.uniform,
    #         parametry low, high, size.

    # Krok 3. Zasoby na start = pojemność, ale jako OSOBNA tablica.
    #         Pytanie: co by się stało podczas zbierania, gdyby zasoby_pol
    #         i pojemnosc były tą samą tablicą (zasoby_pol = pojemnosc)?

    # Krok 4. Zwróć parę (pojemnosc, zasoby_pol).

    H = cfg["plansza"]["H"]
    W = cfg["plansza"]["W"]
    c_min = cfg["zasoby"]["c_min"]
    c_max = cfg["zasoby"]["c_max"]

    pojemnosc = rng.uniform(c_min, c_max, size=(H, W))
    zasoby_pol = pojemnosc.copy()
    return pojemnosc, zasoby_pol


def odrost(stan: Stan, alfa: float) -> None:
    """Odrost zasobów na wszystkich polach po kroku symulacji

    Każde pole zyskuje alfa jednostek zasobu, ale nie więcej niż jego
    pojemność: zasoby = min(zasoby + alfa, pojemność) – osobno dla każdego pola.

    Argumenty:
        stan: stan symulacji; potrzebne pola stan.zasoby_pol i stan.pojemnosc.
        alfa: odrost na krok (alfa >= 0, klucz zasoby.alfa).

    Zmienia: stan.zasoby_pol (w miejscu). Nic nie zwraca. Nie losuje.
    Woła: pętla, na końcu każdego kroku (kontrakt, sekcja 5, krok 8).
    """
    # Jedna operacja na całej tablicy – bez pętli po polach.
    # Do sprawdzenia w dokumentacji NumPy: np.minimum i jego parametr `out`
    # (pozwala zapisać wynik do istniejącej tablicy, czyli „w miejscu”).
    #
    # Pytanie: czym różni się zapis wyniku do istniejącej tablicy od
    # przypisania stan.zasoby_pol = <nowa tablica>? Kiedy ta różnica
    # mogłaby mieć znaczenie (podpowiedź: ktoś trzyma odwołanie do starej
    # tablicy, np. migawka z początku kroku)?   
    np.minimum(stan.zasoby_pol + alfa, stan.pojemnosc, out=stan.zasoby_pol)

def zbierz(stan: Stan, i: int) -> float:
    """Agent i zbiera cały zasób z pola, na którym stoi (akcja ZBIERZ, kod 4).

    Argumenty:
        stan: stan symulacji; potrzebne pola stan.pozycje, stan.zasoby_pol,
              stan.zasoby_agentow.
        i:    indeks żywego agenta (0 … N−1).

    Zwraca:
        zebraną ilość jako zwykły float Pythona (może wynosić 0, gdy pole jest puste).

    Zmienia: stan.zasoby_pol na polu agenta (na 0) i stan.zasoby_agentow[i]
    (o zebraną ilość). Nie losuje. Suma zasobów agenta i pola się nie zmienia.
    Woła: pętla, przy akcji 4 (kontrakt, sekcja 5, krok 5).
    """
    # Krok 1. Odczytaj wiersz i kolumnę agenta z stan.pozycje[i].
    # Krok 2. Zapamiętaj ilość zasobu na tym polu.
    # Krok 3. Dodaj ją do zasobów agenta i wyzeruj pole.
    #         Pytanie: dlaczego kolejność kroków 2 i 3 ma znaczenie?
    # Krok 4. Zwróć ilość jako float(...).
    #         Pytanie: jaki typ miałaby wartość bez float(...) i czy to problem?
    #
    # Funkcja nie sprawdza, czy agent żyje – to zadanie pętli (kontrakt,
    # sekcja 4.7: walidacja akcji na aktualnym stanie).

    w, k = stan.pozycje[i]
    ilosc = stan.zasoby_pol[w, k]
    stan.zasoby_agentow[i] += ilosc
    stan.zasoby_pol[w, k] = 0.0
    return float(ilosc)
