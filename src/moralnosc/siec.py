"""Sieć neuronowa agentów i wybór akcji.

Na ten moment sieć jest liniowa: 5 wejść (obserwacje) → 8 wyjść (po jednym na
akcję), bez warstwy ukrytej. Genom agenta = wektor wszystkich wag sieci razem
z biasami, długość L = 48.
"""
import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import N_WE, N_WY


def dlugosc_genomu(n_we: int, n_wy: int) -> int:
    """Długość genomu sieci liniowej: (n_we + 1) · n_wy – wagi plus biasy.
    """
    return (n_we + 1) * n_wy


def losuj_genomy(n: int, L: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Genomy pokolenia zerowego: tablica n×L z rozkładu normalnego N(0, 1)
    n - liczba agentów, L - długość genomu (liczba wag + biasów)
    """
    return rng.standard_normal((n, L))


def wyjscia(obs: NDArray[np.float64], genomy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wyjścia sieci wszystkich agentów naraz: y[i] = obs[i] · W[i] + b[i]
    """
    N = obs.shape[0]
    W = genomy[:, :N_WE * N_WY].reshape(N, N_WE, N_WY)
    b = genomy[:, N_WE * N_WY:]
    y = np.einsum("ij,ijk->ik", obs, W) + b
    return y


def wybierz_akcje(
    y: NDArray[np.float64], tryb: str, tau: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Zamienia wyjścia sieci na jedną akcję na agenta

    Argumenty:
        y:    N×8 – wyjścia sieci
        tryb: "argmax" – akcja o największym wyjściu (agent deterministyczny)
              "softmax" – akcja losowana z prawdopodobieństwem ∝ exp(y / tau)
        tau:  temperatura (> 0), używana tylko w trybie softmax;
        rng:  generator przebiegu – używany TYLKO w trybie softmax (N losowań).

    Zwraca: tablica N kodów akcji 0–7 (int64).
    Inny tryb niż "argmax" i "softmax" → ValueError z nazwą trybu.
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
        # Zabezpieczenie przed błędem zaokrąglenia: suma skumulowana może wyjść
        # minimalnie poniżej 1, a wtedy indeks wypadłby poza ostatnią akcję.
        akcje = np.minimum(akcje, y.shape[1] - 1)
    else:
        raise ValueError(f"Nieznany tryb {tryb!r}")
    return akcje.astype(np.int64)
