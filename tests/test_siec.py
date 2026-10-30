"""Testy modułu siec.py – testy minimalne na Lab02
Testy statystyczne softmaksu (małe i duże tau) – etap 4.
"""
import numpy as np
import pytest

from moralnosc import siec
from moralnosc.stan import N_WE, N_WY, NIC, OBS_ZASOBY_POD, ZBIERZ



def test_znane_wagi_i_bias_wybieraja_oczekiwane_akcje():
    L = siec.dlugosc_genomu(N_WE, N_WY)
    assert L == 48

    genomy = np.zeros((2, L))
    # agent 0: waga z wejścia „zasoby pod agentem” do akcji ZBIERZ = 1.0
    # (waga z wejścia j do akcji k leży pod indeksem j·N_WY + k)
    genomy[0, OBS_ZASOBY_POD * N_WY + ZBIERZ] = 1.0
    # agent 1: tylko bias akcji NIC = 1.0 (biasy zaczynają się od indeksu N_WE·N_WY = 40)
    genomy[1, N_WE * N_WY + NIC] = 1.0
    # obaj widzą to samo: pod sobą zasób 0.75 (po normalizacji)
    obs = np.array([[0.5, 0.75, 0.0, 0.0, 0.0],
                    [0.5, 0.75, 0.0, 0.0, 0.0]])

    y = siec.wyjscia(obs, genomy)

    np.testing.assert_allclose(y[0], [0, 0, 0, 0, 0.75, 0, 0, 0])
    np.testing.assert_allclose(y[1], [0, 0, 0, 0, 0, 0, 0, 1.0])
    # ta sama obserwacja, inne genomy → inne decyzje: każdy agent liczy własną sieć
    akcje = siec.wybierz_akcje(y, "argmax", tau=1.0, rng=np.random.default_rng(0))
    assert list(akcje) == [ZBIERZ, NIC]



@pytest.mark.parametrize("tryb", ["argmax", "softmax"])
def test_ksztalty_i_zakres_akcji(tryb):
    N = 6
    rng = np.random.default_rng(3)
    L = siec.dlugosc_genomu(N_WE, N_WY)
    genomy = siec.losuj_genomy(N, L, rng)
    obs = rng.random((N, N_WE))
    y = siec.wyjscia(obs, genomy)
    akcje = siec.wybierz_akcje(y, tryb, 1.0, rng)
    assert genomy.shape == (N, L)
    assert y.shape == (N, N_WY)
    assert akcje.shape == (N,)
    assert akcje.dtype == np.int64
    assert np.all((akcje >= 0) & (akcje < N_WY))

def test_softmax_to_samo_ziarno_te_same_akcje():
    y = np.random.default_rng(42).random((50, 8))
    akcje1 = siec.wybierz_akcje(y, "softmax", 1.0, np.random.default_rng(123))
    akcje2 = siec.wybierz_akcje(y, "softmax", 1.0, np.random.default_rng(123))
    assert np.array_equal(akcje1, akcje2)
