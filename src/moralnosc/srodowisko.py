"""Środowisko symulacji: plansza ze ścianami, zasoby pól, odrost i zbieranie."""

import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import Stan


def utworz_plansze(
    cfg: dict, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Plansza H×W: (pojemnosc, zasoby_pol).

    Pojemność losowana jednostajnie z [c_min, c_max]; na starcie pola są pełne.
    """

    H = cfg["plansza"]["H"]
    W = cfg["plansza"]["W"]
    c_min = cfg["zasoby"]["c_min"]
    c_max = cfg["zasoby"]["c_max"]

    pojemnosc = rng.uniform(c_min, c_max, size=(H, W))
    zasoby_pol = pojemnosc.copy()
    return pojemnosc, zasoby_pol


def odrost(stan: Stan, alfa: float) -> None:
    """Każde pole zyskuje alfa, najwyżej do swojej pojemności (w miejscu)."""
    # out= zapisuje do istniejącej tablicy – odwołania do niej zostają aktualne
    np.minimum(stan.zasoby_pol + alfa, stan.pojemnosc, out=stan.zasoby_pol)


def zbierz(stan: Stan, i: int) -> float:
    """Agent i zabiera cały zasób z pola pod sobą; zwraca zebraną ilość (może być 0)."""
    w, k = stan.pozycje[i]
    ilosc = stan.zasoby_pol[w, k]
    stan.zasoby_agentow[i] += ilosc
    stan.zasoby_pol[w, k] = 0.0
    return float(ilosc)
