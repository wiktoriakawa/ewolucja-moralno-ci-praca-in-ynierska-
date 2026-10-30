"""Środowisko symulacji: plansza ze ścianami, zasoby pól, odrost i zbieranie."""

import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import Stan


def utworz_plansze(
    cfg: dict, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tworzy planszę: mapę pojemności pól i początkowe zasoby pól.

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

    H = cfg["plansza"]["H"]
    W = cfg["plansza"]["W"]
    c_min = cfg["zasoby"]["c_min"]
    c_max = cfg["zasoby"]["c_max"]

    pojemnosc = rng.uniform(c_min, c_max, size=(H, W))
    zasoby_pol = pojemnosc.copy()
    return pojemnosc, zasoby_pol


def odrost(stan: Stan, alfa: float) -> None:
    """Odrost zasobów na wszystkich polach po kroku symulacji.

    Każde pole zyskuje alfa jednostek zasobu, ale nie więcej niż jego
    pojemność: zasoby = min(zasoby + alfa, pojemność) – osobno dla każdego pola.

    Argumenty:
        stan: stan symulacji; potrzebne pola stan.zasoby_pol i stan.pojemnosc.
        alfa: odrost na krok (alfa >= 0, klucz zasoby.alfa).

    Zmienia: stan.zasoby_pol (w miejscu). Nic nie zwraca. Nie losuje.
    Woła: pętla, na końcu każdego kroku (kontrakt, sekcja 5, krok 8).
    """
    # Jedna operacja na całej tablicy, bez pętli po polach. Parametr `out` zapisuje
    # wynik do istniejącej tablicy, więc odwołania do stan.zasoby_pol trzymane
    # gdzie indziej pozostają aktualne.
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
    w, k = stan.pozycje[i]
    ilosc = stan.zasoby_pol[w, k]
    stan.zasoby_agentow[i] += ilosc
    stan.zasoby_pol[w, k] = 0.0
    return float(ilosc)
