"""Testy siec.py: sieć liniowa (h = 0, regresja Lab02) i warstwa ukryta (#24).

Testy statystyczne softmaksu – etap 4.
"""
import numpy as np
import pytest

from moralnosc import siec
from moralnosc.stan import N_WE, N_WY, NIC, OBS_ZASOBY_POD, ZBIERZ


# --- sieć liniowa (h = 0) ----------------------------------------------------------------

def test_znane_wagi_i_bias_wybieraja_oczekiwane_akcje():
    L = siec.dlugosc_genomu(N_WE, N_WY, 0)
    assert L == 48

    genomy = np.zeros((2, L))
    # agent 0: waga „zasoby pod agentem” → ZBIERZ (indeks j·N_WY + a)
    genomy[0, OBS_ZASOBY_POD * N_WY + ZBIERZ] = 1.0
    # agent 1: tylko bias NIC (biasy od indeksu 40)
    genomy[1, N_WE * N_WY + NIC] = 1.0
    obs = np.array([[0.5, 0.75, 0.0, 0.0, 0.0],
                    [0.5, 0.75, 0.0, 0.0, 0.0]])

    y = siec.wyjscia(obs, genomy, 0)

    np.testing.assert_allclose(y[0], [0, 0, 0, 0, 0.75, 0, 0, 0])
    np.testing.assert_allclose(y[1], [0, 0, 0, 0, 0, 0, 0, 1.0])
    # ta sama obserwacja, inne genomy → inne decyzje
    akcje = siec.wybierz_akcje(y, "argmax", tau=1.0, rng=np.random.default_rng(0))
    assert list(akcje) == [ZBIERZ, NIC]


@pytest.mark.parametrize("tryb", ["argmax", "softmax"])
def test_ksztalty_i_zakres_akcji(tryb):
    N = 6
    rng = np.random.default_rng(3)
    L = siec.dlugosc_genomu(N_WE, N_WY, 0)
    genomy = siec.losuj_genomy(N, L, rng)
    obs = rng.random((N, N_WE))
    y = siec.wyjscia(obs, genomy, 0)
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


# --- długość genomu ----------------------------------------------------------------------

def test_dlugosc_genomu_wedlug_wzoru():
    assert siec.dlugosc_genomu(5, 8, 0) == 48
    assert siec.dlugosc_genomu(5, 8, 2) == 36    # 6·2 + 3·8
    assert siec.dlugosc_genomu(5, 8, 8) == 120   # 6·8 + 9·8
    assert siec.dlugosc_genomu(6, 8, 8) == 128   # z pamięcią (D2)
    with pytest.raises(ValueError):
        siec.dlugosc_genomu(5, 8, -1)


# --- warstwa ukryta: przykład liczony ręcznie --------------------------------------------

def test_warstwa_ukryta_znane_wagi():
    """h = 2, L = 36: W1 (0–9), b1 (10–11), W2 (12–27), b2 (28–35)."""
    h = 2
    L = siec.dlugosc_genomu(N_WE, N_WY, h)
    assert L == 36

    genomy = np.zeros((2, L))
    # agent 0: wejście 1 → neuron 0 (waga 1), neuron 0 → ZBIERZ (waga 2), bias NIC = 0.5
    genomy[0, OBS_ZASOBY_POD * h + 0] = 1.0                      # W1[1, 0]
    genomy[0, (N_WE + 1) * h + 0 * N_WY + ZBIERZ] = 2.0          # W2[0, 4]
    genomy[0, (N_WE + 1) * h + h * N_WY + NIC] = 0.5             # b2[7]
    # agent 1: tylko bias NIC
    genomy[1, (N_WE + 1) * h + h * N_WY + NIC] = 1.0
    obs = np.array([[0.5, 0.75, 0.0, 0.0, 0.0],
                    [0.5, 0.75, 0.0, 0.0, 0.0]])

    y = siec.wyjscia(obs, genomy, h)

    # y[ZBIERZ] = 2·tanh(0.75) ≈ 1.270298
    oczekiwane_0 = np.zeros(N_WY)
    oczekiwane_0[ZBIERZ] = 2 * np.tanh(0.75)
    oczekiwane_0[NIC] = 0.5
    np.testing.assert_allclose(y[0], oczekiwane_0, rtol=1e-12)
    assert y[0, ZBIERZ] == pytest.approx(1.2702979, abs=1e-6)
    np.testing.assert_allclose(y[1], [0, 0, 0, 0, 0, 0, 0, 1.0])
    akcje = siec.wybierz_akcje(y, "argmax", tau=1.0, rng=np.random.default_rng(0))
    assert list(akcje) == [ZBIERZ, NIC]


# --- dekodowanie genomu ------------------------------------------------------------------

def test_rozloz_genom_ksztalty_i_polozenie_wag():
    """Każda waga ma inną wartość; położenie sprawdzane pętlami wprost ze wzorów indeksów."""
    N, h = 3, 4
    L = siec.dlugosc_genomu(N_WE, N_WY, h)
    genomy = np.arange(N * L, dtype=float).reshape(N, L)

    W1, b1, W2, b2 = siec.rozloz_genom(genomy, N_WE, h)

    assert W1.shape == (N, N_WE, h)
    assert b1.shape == (N, h)
    assert W2.shape == (N, h, N_WY)
    assert b2.shape == (N, N_WY)
    for i in range(N):
        for k in range(h):
            for j in range(N_WE):
                assert W1[i, j, k] == genomy[i, j * h + k]
            assert b1[i, k] == genomy[i, N_WE * h + k]
            for a in range(N_WY):
                assert W2[i, k, a] == genomy[i, (N_WE + 1) * h + k * N_WY + a]
        for a in range(N_WY):
            assert b2[i, a] == genomy[i, (N_WE + 1) * h + h * N_WY + a]
    for blok in (W1, b1, W2, b2):
        assert np.shares_memory(blok, genomy)   # widoki, nie kopie

    # h = 0: [W, b]
    L0 = siec.dlugosc_genomu(N_WE, N_WY, 0)
    genomy0 = np.arange(N * L0, dtype=float).reshape(N, L0)
    bloki0 = siec.rozloz_genom(genomy0, N_WE, 0)
    assert len(bloki0) == 2
    W, b = bloki0
    assert W.shape == (N, N_WE, N_WY)
    assert b.shape == (N, N_WY)
    for i in range(N):
        for a in range(N_WY):
            for j in range(N_WE):
                assert W[i, j, a] == genomy0[i, j * N_WY + a]
            assert b[i, a] == genomy0[i, N_WE * N_WY + a]


def test_rozloz_genom_odrzuca_zla_dlugosc():
    genomy = np.zeros((2, 35))   # o jeden za krótki dla h = 2
    with pytest.raises(ValueError, match="36"):
        siec.rozloz_genom(genomy, N_WE, 2)
    with pytest.raises(ValueError):
        siec.wyjscia(np.zeros((2, N_WE)), genomy, 2)

    # L = 120 (h = 8) czytane jako liniowe – 120 dzieli się przez 6
    genomy_h8 = np.zeros((2, siec.dlugosc_genomu(N_WE, N_WY, 8)))
    with pytest.raises(ValueError):
        siec.rozloz_genom(genomy_h8, N_WE, 0)
    with pytest.raises(ValueError):
        siec.wyjscia(np.zeros((2, N_WE)), genomy_h8, 0)

    # i odwrotnie: L = 48 czytane z h = 8
    genomy_liniowe = np.zeros((2, siec.dlugosc_genomu(N_WE, N_WY, 0)))
    with pytest.raises(ValueError):
        siec.wyjscia(np.zeros((2, N_WE)), genomy_liniowe, 8)


# --- własności warstwy ukrytej -----------------------------------------------------------

@pytest.mark.parametrize("tryb", ["argmax", "softmax"])
@pytest.mark.parametrize("h", [1, 8])
def test_ksztalty_z_warstwa_ukryta(h, tryb):
    N = 6
    rng = np.random.default_rng(3)
    L = siec.dlugosc_genomu(N_WE, N_WY, h)
    genomy = siec.losuj_genomy(N, L, rng)
    obs = rng.random((N, N_WE))
    y = siec.wyjscia(obs, genomy, h)
    akcje = siec.wybierz_akcje(y, tryb, 1.0, rng)
    assert genomy.shape == (N, L)
    assert y.shape == (N, N_WY)
    assert y.dtype == np.float64
    assert akcje.shape == (N,)
    assert akcje.dtype == np.int64
    assert np.all((akcje >= 0) & (akcje < N_WY))


def test_bias_neuronu_ukrytego_przechodzi_przez_tanh():
    """obs = 0 → y[a] = tanh(b1[k])·W2[k, a]; łapie pominięcie b1."""
    h = 2
    L = siec.dlugosc_genomu(N_WE, N_WY, h)
    genomy = np.zeros((1, L))
    genomy[0, N_WE * h + 1] = 0.5                              # b1[1]
    genomy[0, (N_WE + 1) * h + 1 * N_WY + ZBIERZ] = 3.0        # W2[1, ZBIERZ]

    y = siec.wyjscia(np.zeros((1, N_WE)), genomy, h)

    oczekiwane = np.zeros(N_WY)
    oczekiwane[ZBIERZ] = 3.0 * np.tanh(0.5)
    np.testing.assert_allclose(y[0], oczekiwane, rtol=1e-12)


def test_zerowe_W2_daje_same_biasy_wyjsc():
    """W2 = 0 → y = b2 niezależnie od obserwacji."""
    N, h = 4, 3
    rng = np.random.default_rng(5)
    L = siec.dlugosc_genomu(N_WE, N_WY, h)
    genomy = siec.losuj_genomy(N, L, rng)
    poczatek_W2 = (N_WE + 1) * h
    koniec_W2 = poczatek_W2 + h * N_WY
    genomy[:, poczatek_W2:koniec_W2] = 0.0
    b2 = genomy[:, koniec_W2:]

    obs1 = rng.random((N, N_WE))
    obs2 = rng.random((N, N_WE))
    assert not np.allclose(obs1, obs2)

    np.testing.assert_allclose(siec.wyjscia(obs1, genomy, h), b2)
    np.testing.assert_allclose(siec.wyjscia(obs2, genomy, h), b2)


def test_wyjscia_ograniczone_przez_tanh():
    """|y| ≤ Σ_k |W2[k, a]| + |b2[a]| nawet przy ogromnych obs (dla h = 0 nie zachodzi)."""
    N, h = 5, 4
    rng = np.random.default_rng(7)
    L = siec.dlugosc_genomu(N_WE, N_WY, h)
    genomy = siec.losuj_genomy(N, L, rng)
    obs = 1e6 * rng.random((N, N_WE))

    y = siec.wyjscia(obs, genomy, h)

    _, _, W2, b2 = siec.rozloz_genom(genomy, N_WE, h)
    ograniczenie = np.abs(W2).sum(axis=1) + np.abs(b2)   # suma po neuronach
    assert ograniczenie.shape == (N, N_WY)
    assert np.all(np.abs(y) <= ograniczenie + 1e-12)
