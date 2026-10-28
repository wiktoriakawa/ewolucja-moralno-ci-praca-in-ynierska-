"""Wczytywanie i minimalna walidacja konfiguracji przebiegu."""

import tomllib
from os import PathLike


def wczytaj_konfiguracje(sciezka: str | PathLike[str]) -> dict:
    """Wczytuje konfiguracje przebiegu z pliku TOML.

    Zwraca konfiguracje jako zagniezdzony slownik. Sprawdza warunki
    1 <= N <= H * W oraz p * x - c_k > 0.

    Zglasza:
        AssertionError: gdy jeden ze sprawdzanych warunkow nie jest spelniony.
    """
    with open(sciezka, "rb") as plik:
        cfg = tomllib.load(plik)

    n = cfg["populacja"]["N"]
    h = cfg["plansza"]["H"]
    w = cfg["plansza"]["W"]
    assert 1 <= n <= h * w, "populacja.N musi spelniac warunek 1 <= N <= H * W"

    p = cfg["koszty"]["p"]
    x = cfg["koszty"]["x"]
    c_k = cfg["koszty"]["c_k"]
    assert p * x - c_k > 0, (
        "koszty.p, koszty.x i koszty.c_k musza spelniac warunek p * x - c_k > 0"
    )

    return cfg
