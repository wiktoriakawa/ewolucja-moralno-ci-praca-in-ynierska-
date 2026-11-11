"""Populacja i percepcja: rozmieszczenie agentów, ruch, pole przed agentem, obserwacje."""

import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import KIERUNKI, PUSTE, Stan


def rozmiesc_losowo(stan: Stan, rng: np.random.Generator) -> None:
    """N agentów na losowych, różnych polach, z losowymi kierunkami (N ≤ H·W)."""
    H = stan.zajetosc.shape[0]
    W = stan.zajetosc.shape[1]
    N = stan.pozycje.shape[0]

    numery_pol = rng.choice(H * W, size=N, replace=False)
    wiersze, kolumny = np.divmod(numery_pol, W)
    stan.pozycje[:, 0] = wiersze
    stan.pozycje[:, 1] = kolumny
    stan.zajetosc.fill(PUSTE)
    stan.zajetosc[wiersze, kolumny] = np.arange(N)
    stan.kierunki[:] = rng.integers(0, 4, size=N)


def ruch(stan: Stan, i: int, kierunek: int) -> bool:
    """Ruch agenta i o jedno pole; False przy ścianie albo zajętym celu.

    Kierunek patrzenia zmienia się zawsze, nawet gdy ruch się nie uda.
    """
    stan.kierunki[i] = kierunek
    cel = stan.pozycje[i] + KIERUNKI[kierunek]
    if not (0 <= cel[0] < stan.zajetosc.shape[0] and 0 <= cel[1] < stan.zajetosc.shape[1]):
        return False
    if stan.zajetosc[cel[0], cel[1]] != PUSTE:
        return False
    stan.zajetosc[stan.pozycje[i, 0], stan.pozycje[i, 1]] = PUSTE
    stan.zajetosc[cel[0], cel[1]] = i
    stan.pozycje[i] = cel
    return True


def usun_martwych(stan: Stan, maska: NDArray[np.bool_]) -> None:
    """Oznacza agentów z maski jako martwych i zdejmuje ich z planszy.

    Tablice nie zmieniają rozmiaru – martwy zostaje w nich do końca pokolenia.
    """
    # tylko żywi: pozycja martwego jest nieaktualna, mogłaby skasować sąsiada
    maska = maska & stan.zywy
    stan.zywy[maska] = False
    stan.zajetosc[stan.pozycje[maska, 0], stan.pozycje[maska, 1]] = PUSTE


def pole_przed(stan: Stan) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """(cele N×2, w_planszy N) – pole przed każdym agentem; przy ścianie poza planszą."""
    H, W = stan.zajetosc.shape
    cele = stan.pozycje + KIERUNKI[stan.kierunki]
    w_planszy = (0 <= cele[:, 0]) & (cele[:, 0] < H) & (0 <= cele[:, 1]) & (cele[:, 1] < W)
    return cele, w_planszy


def obserwacje(stan: Stan, skala: float) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Wejścia sieci całej populacji: (obs N×5, przed N).

    Kolumny obs jak stałe OBS_* w stan.py; zasoby dzielone przez skala,
    obecność sąsiada (kol. 3) to 0/1. przed – indeks agenta przed agentem albo PUSTE.
    """
    cele, w_planszy = pole_przed(stan)
    # clip tylko po to, żeby indeksować bez błędu; wartości spoza planszy zeruje where
    bezpieczne = np.clip(cele, 0, [stan.zajetosc.shape[0] - 1, stan.zajetosc.shape[1] - 1])
    zasoby_przed = np.where(w_planszy, stan.zasoby_pol[bezpieczne[:, 0], bezpieczne[:, 1]], 0.0)
    przed = np.where(w_planszy, stan.zajetosc[bezpieczne[:, 0], bezpieczne[:, 1]], PUSTE)
    obecny = (przed >= 0).astype(np.float64)
    zasoby_przed_agent = np.where(obecny, stan.zasoby_agentow[np.maximum(przed, 0)], 0.0)
    zasoby_wlasne = stan.zasoby_agentow / skala
    zasoby_pod = stan.zasoby_pol[stan.pozycje[:, 0], stan.pozycje[:, 1]] / skala
    obs = np.column_stack(
        (zasoby_wlasne, zasoby_pod, zasoby_przed / skala, obecny, zasoby_przed_agent / skala)
    )
    return obs, przed
