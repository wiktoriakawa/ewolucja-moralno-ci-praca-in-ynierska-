"""Sieć neuronowa agentów i wybór akcji.

h = 0 – sieć liniowa (Lab02): y = x·W + b, L = (n_we + 1)·n_wy = 48.
h ≥ 1 – warstwa ukryta z tanh:  y = tanh(x·W1 + b1)·W2 + b2,
        L = (n_we + 1)·h + (h + 1)·n_wy  (h = 8 → 120).

Układ genomu (wiersz `genomy`, macierze wierszami jak w `reshape`):
    h = 0:  [ W (n_we×n_wy) | b ]
    h ≥ 1:  [ W1 (n_we×h) | b1 | W2 (h×n_wy) | b2 ]

n_we bierzemy z obs.shape[1] (pamięć #25 nie wymaga zmian), n_wy = N_WY.
"""
import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import N_WY


def dlugosc_genomu(n_we: int, n_wy: int, h: int) -> int:
    """Liczba wag i biasów sieci; h = 0 to sieć liniowa. h < 0 → ValueError."""
    if h < 0:
        raise ValueError(f"Liczba neuronów ukrytych h musi być >= 0, jest {h}")
    # osobna gałąź: ogólny wzór dla h = 0 dałby n_wy zamiast (n_we + 1)·n_wy
    if h == 0:
        return (n_we + 1) * n_wy
    return (n_we + 1) * h + (h + 1) * n_wy


def losuj_genomy(n: int, L: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Genomy pokolenia zerowego: n×L z N(0, 1)."""
    return rng.standard_normal((n, L))


def rozloz_genom(
    genomy: NDArray[np.float64], n_we: int, h: int
) -> list[NDArray[np.float64]]:
    """Dzieli genomy N×L na bloki: [W, b] (h = 0) albo [W1, b1, W2, b2].

    Bloki są widokami na `genomy`. Zła długość L → ValueError (inaczej genom
    z warstwą ukrytą dałby się po cichu odczytać jako liniowy).
    """
    N, L = genomy.shape
    n_wy = N_WY
    oczekiwana = dlugosc_genomu(n_we, n_wy, h)
    if L != oczekiwana:
        raise ValueError(
            f"Genom ma długość L = {L}, a sieć o n_we = {n_we}, h = {h} "
            f"potrzebuje L = {oczekiwana}"
        )

    if h == 0:
        W = genomy[:, :n_we * n_wy].reshape(N, n_we, n_wy)
        b = genomy[:, n_we * n_wy:]
        return [W, b]

    p1 = n_we * h          # koniec W1
    p2 = p1 + h            # koniec b1
    p3 = p2 + h * n_wy     # koniec W2; dalej b2
    W1 = genomy[:, :p1].reshape(N, n_we, h)
    b1 = genomy[:, p1:p2]
    W2 = genomy[:, p2:p3].reshape(N, h, n_wy)
    b2 = genomy[:, p3:]
    return [W1, b1, W2, b2]


def wyjscia(
    obs: NDArray[np.float64], genomy: NDArray[np.float64], h: int
) -> NDArray[np.float64]:
    """Wyjścia sieci wszystkich agentów naraz: obs N×n_we → y N×n_wy.

    h bez wartości domyślnej – celowo, żeby nie pomylić architektury.
    """
    n_we = obs.shape[1]
    bloki = rozloz_genom(genomy, n_we, h)

    # einsum "ij,ijk->ik": każdy agent i mnoży swoje wejścia przez własną macierz
    if h == 0:
        W, b = bloki
        return np.einsum("ij,ijk->ik", obs, W) + b

    W1, b1, W2, b2 = bloki
    ukryte = np.tanh(np.einsum("ij,ijk->ik", obs, W1) + b1)   # N×h
    return np.einsum("ij,ijk->ik", ukryte, W2) + b2           # N×n_wy


def wybierz_akcje(
    y: NDArray[np.float64], tryb: str, tau: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Jedna akcja (0–7, int64) na agenta z wyjść y.

    "argmax" – największe wyjście; "softmax" – losowanie ∝ exp(y / tau),
    tylko ten tryb zużywa rng. Inny tryb → ValueError.
    """
    y = np.asarray(y)
    if tryb == "argmax":
        akcje = np.argmax(y, axis=1)
    elif tryb == "softmax":
        y = (y - np.max(y, axis=1, keepdims=True)) / tau
        p = np.exp(y)
        p /= np.sum(p, axis=1, keepdims=True)
        u = rng.random(y.shape[0])
        akcje = np.sum(np.cumsum(p, axis=1) < u[:, None], axis=1)
        # suma skumulowana może wyjść minimalnie < 1 – indeks poza zakresem
        akcje = np.minimum(akcje, y.shape[1] - 1)
    else:
        raise ValueError(f"Nieznany tryb {tryb!r}")
    return akcje.astype(np.int64)
