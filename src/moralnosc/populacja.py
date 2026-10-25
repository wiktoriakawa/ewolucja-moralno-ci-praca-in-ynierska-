"""Populacja i percepcja: rozmieszczenie agentów, ruch, pole przed agentem, obserwacje
"""
import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import KIERUNKI, PUSTE, Stan


def rozmiesc_losowo(stan: Stan, rng: np.random.Generator) -> None:
    """Rozmieszcza N agentów na losowych, różnych polach i losuje ich kierunki.

    Argumenty:
        stan: stan z ustalonymi rozmiarami (H×W z mapy zajętości, N z tablicy pozycji),
              np. z pusty_stan; N ≤ H·W (sprawdza konfiguracja).
        rng:  generator przebiegu.

    Nic nie zwraca, zmienia stan.pozycje, stan.zajetosc i stan.kierunki.    
    
    """

    H = stan.zajetosc.shape[0]
    W = stan.zajetosc.shape[1]
    N = stan.pozycje.shape[0] # liczba agentów

    numery_pol = rng.choice(H * W, size=N, replace=False) # losowe, różne pola 
    wiersze, kolumny = np.divmod(numery_pol, W)
    stan.pozycje[:, 0] = wiersze
    stan.pozycje[:, 1] = kolumny
    stan.zajetosc.fill(PUSTE)
    stan.zajetosc[wiersze, kolumny] = np.arange(N)
    stan.kierunki[:] = rng.integers(0, 4, size=N)



def ruch(stan: Stan, i: int, kierunek: int) -> bool:
    """Próba ruchu agenta i o jedno pole w zadanym kierunku (akcje 0–3).

    Argumenty: stan, indeks agenta i kierunek ruchu (0 = N, 1 = E, 2 = S, 3 = W).
    Zwraca: True, jeśli agent się przesunął; False, jeśli cel był poza planszą (ściana) albo zajęty przez innego agenta.
    Zmienia: stan.pozycje[i], stan.zajetosc. Nie zmienia kierunku patrzenia agenta.
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
    """Oznacza agentów z maski jako martwych i zdejmuje ich z mapy zajętości.
    Argumenty: stan symulacji i maska N wartości logicznych – True dla agentów do usunięcia.

    Zmienia: stan.zywy, stan.zajetosc. Tablice agentów NIE zmieniają rozmiaru
    – martwy agent zostaje w tablicach do końca pokolenia 
    """

    stan.zywy[maska] = False
    stan.zajetosc[stan.pozycje[maska, 0], stan.pozycje[maska, 1]] = PUSTE


def pole_przed(stan: Stan) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Współrzędne pola przed każdym agentem i informacja, czy leży w planszy.

    Zwraca:
        cele:      tablica N×2 (int64) – pozycja + KIERUNKI[kierunek] dla każdego
                   agenta; przy ścianie zawiera współrzędne spoza planszy (−1, H albo W);
        w_planszy: tablica N wartości logicznych – True, gdy cel leży w planszy.
    """

    cele = stan.pozycje + KIERUNKI[stan.kierunki]
    w_planszy = (0 <= cele[:, 0]) & (cele[:, 0] < stan.zajetosc.shape[0]) & (0 <= cele[:, 1]) & (cele[:, 1] < stan.zajetosc.shape[1])
    return cele, w_planszy


def obserwacje(stan: Stan, skala: float) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Wejścia sieci dla całej populacji naraz.
    Argumenty:
        stan:  stan symulacji.
        skala: dzielnik normalizujący zasoby (klucz obserwacje.skala, domyślnie c_max).

    Zwraca:
        obs:   tablica N×5 (float64), kolumny jak stałe OBS_* w stan.py:
               0 – zasoby własne agenta,
               1 – zasoby pola, na którym stoi,
               2 – zasoby pola przed nim (0 przy ścianie),
               3 – obecność agenta przed nim (1.0 albo 0.0),
               4 – zasoby agenta przed nim (0, gdy nikogo nie ma);
               kolumny 0, 1, 2 i 4 są podzielone przez skala, kolumna 3 – nie;
        przed: tablica N (int64) – indeks agenta stojącego przed agentem albo PUSTE.
    """

    cele, w_planszy = pole_przed(stan)
    bezpieczne = np.clip(cele, 0, [stan.zajetosc.shape[0] - 1, stan.zajetosc.shape[1] - 1])
    zasoby_przed = np.where(w_planszy, stan.zasoby_pol[bezpieczne[:, 0], bezpieczne[:, 1]], 0.0)
    przed = np.where(w_planszy, stan.zajetosc[bezpieczne[:, 0], bezpieczne[:, 1]], PUSTE)
    obecny = (przed >= 0).astype(np.float64)
    zasoby_przed_agent = np.where(obecny, stan.zasoby_agentow[np.maximum(przed, 0)], 0.0)
    zasoby_wlasne = stan.zasoby_agentow / skala
    zasoby_pod = stan.zasoby_pol[stan.pozycje[:, 0], stan.pozycje[:, 1]] / skala
    obs = np.column_stack((zasoby_wlasne, zasoby_pod, zasoby_przed / skala, obecny, zasoby_przed_agent / skala))
    return obs, przed
