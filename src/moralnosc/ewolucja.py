"""Ewolucja: pokolenie zerowe, kwalifikacja, losowanie rodziców, mutacja,
narodziny potomków w promieniu R i podmiana pokolenia.
"""
import numpy as np
from numpy.typing import NDArray

from moralnosc.populacja import rozmiesc_losowo
from moralnosc.siec import dlugosc_genomu, losuj_genomy
from moralnosc.srodowisko import utworz_plansze
from moralnosc.stan import N_WE, N_WY, PUSTE, Stan, pusty_stan


def pokolenie_zerowe(cfg: dict, rng: np.random.Generator) -> Stan:
    """Stan początkowy: pełna plansza i N żywych agentów z losowymi genomami i zasobami E0.

    Kolejność losowań: plansza → genomy → rozmieszczenie.
    """
    H = cfg["plansza"]["H"]
    W = cfg["plansza"]["W"]
    N = cfg["populacja"]["N"]
    L = dlugosc_genomu(N_WE, N_WY, cfg["siec"]["h"])

    stan = pusty_stan(H=H, W=W, N=N, L=L)
    stan.pojemnosc, stan.zasoby_pol = utworz_plansze(cfg, rng)
    stan.genomy[:] = losuj_genomy(N, L, rng)
    rozmiesc_losowo(stan, rng)

    stan.zasoby_agentow[:] = cfg["ewolucja"]["E0"]
    stan.zywy[:] = True
    stan.ids[:] = np.arange(N)
    stan.nastepny_id = N

    return stan


def kwalifikacja(stan: Stan, T: float) -> tuple[NDArray[np.bool_], bool]:
    """(maska rodziców, awaryjny): żywi z zasobami ≥ T.

    Gdy nikt nie osiąga progu – wszyscy żywi i awaryjny = True.
    Kolumnę kwalifikacja w agenci.csv liczy zapis, nie ta maska.
    """
    maska = stan.zywy & (stan.zasoby_agentow >= T)
    if maska.any():
        return maska, False

    return stan.zywy.copy(), True


def losuj_rodzicow(
    maska: NDArray[np.bool_], n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """n indeksów rodziców z maski – jednostajnie, ze zwracaniem."""
    kandydaci = np.flatnonzero(maska)
    return kandydaci[rng.integers(0, kandydaci.size, size=n)]


def mutuj(
    genomy_rodzicow: NDArray[np.float64],
    r: float,
    sigma: float,
    w_max: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Nowe genomy: każda waga z p. r dostaje szum N(0, sigma), potem przycięcie do ±w_max.

    Rodzice się nie zmieniają.
    """
    maska = rng.random(genomy_rodzicow.shape) < r
    # szum dla wszystkich wag – liczba losowań nie zależy od maski
    szum = rng.normal(0.0, sigma, size=genomy_rodzicow.shape)
    potomkowie = genomy_rodzicow + maska * szum  # `+` daje nową tablicę (`+=` zmieniłoby rodziców)
    np.clip(potomkowie, -w_max, w_max, out=potomkowie)
    return potomkowie


def rozmiesc_potomkow(
    pozycje_rodzicow: NDArray[np.int64],
    R: int,
    H: int,
    W: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(pozycje n×2, zajetosc H×W) potomków na wolnych polach w kwadracie o promieniu R wokół rodzica.

    Brak wolnego pola → promień rośnie o 1. n > H·W → ValueError.
    """
    n = pozycje_rodzicow.shape[0]
    if n > H * W:
        raise ValueError(f"{n} potomków nie zmieści się na planszy {H}×{W}")

    pozycje = np.empty((n, 2), dtype=np.int64)
    zajetosc = np.full((H, W), PUSTE, dtype=np.int64)

    # losowa kolejność: pierwszeństwo wyboru pola nie zależy od numeru potomka
    for i in rng.permutation(n):
        w, k = pozycje_rodzicow[i]
        promien = R
        while True:
            # okno przycięte do planszy, górna granica wyłączna
            w0, w1 = max(0, w - promien), min(H, w + promien + 1)
            k0, k1 = max(0, k - promien), min(W, k + promien + 1)
            wolne = np.argwhere(zajetosc[w0:w1, k0:k1] == PUSTE)
            if wolne.shape[0] > 0:
                break
            # kończy się, bo n ≤ H·W
            promien += 1
        dw, dk = wolne[rng.integers(wolne.shape[0])]
        pozycje[i] = (w0 + dw, k0 + dk)  # z układu okna do układu planszy
        zajetosc[w0 + dw, k0 + dk] = i
    return pozycje, zajetosc


def nowe_pokolenie(
    stan: Stan, maska: NDArray[np.bool_], cfg: dict, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Zastępuje całą populację N potomkami; zwraca indeksy rodziców.

    Plansza przechodzi bez odnawiania. r = "auto" → 1/L.
    Losuje: rodzice → mutacja → kolejność potomków → pola → kierunki.
    """
    N, L = stan.genomy.shape
    H, W = stan.zajetosc.shape
    ew = cfg["ewolucja"]
    r = 1.0 / L if ew["r"] == "auto" else ew["r"]

    # najpierw wszystko ze starego stanu...
    rodzice = losuj_rodzicow(maska, N, rng)
    genomy = mutuj(stan.genomy[rodzice], r, ew["sigma"], ew["w_max"], rng)
    pozycje, zajetosc = rozmiesc_potomkow(stan.pozycje[rodzice], ew["R"], H, W, rng)
    kierunki = rng.integers(0, 4, size=N)

    # ...potem nadpisanie przez [:] (odwołania do tablic zostają aktualne)
    stan.rodzic[:] = stan.ids[rodzice]  # stare id – przed nadaniem nowych
    stan.ids[:] = stan.nastepny_id + np.arange(N)
    stan.nastepny_id += N
    stan.genomy[:] = genomy
    stan.pozycje[:] = pozycje
    stan.zajetosc[:] = zajetosc
    stan.kierunki[:] = kierunki
    stan.zasoby_agentow[:] = ew["E0"]  # bez dziedziczenia bogactwa
    stan.zywy[:] = True
    stan.pokolenie += 1
    return rodzice


def roznorodnosc(genomy: NDArray[np.float64]) -> float:
    """Średnia odległość euklidesowa między genomami (pary i < j); N < 2 → 0.0.

    Stała długość genomu (D2) – wartości porównywalne między pokoleniami;
    spadek do 0 = populacja klonów.
    """
    N = genomy.shape[0]
    if N < 2:
        return 0.0
    # ||a − b||² = ||a||² + ||b||² − 2·a·b – bez tymczasowej tablicy N×N×L
    kwadraty = (genomy ** 2).sum(axis=1)
    d2 = kwadraty[:, None] + kwadraty[None, :] - 2.0 * genomy @ genomy.T
    np.clip(d2, 0.0, None, out=d2)   # zaokrąglenia mogą dać minimalnie < 0
    i, j = np.triu_indices(N, k=1)
    return float(np.sqrt(d2[i, j]).mean())
