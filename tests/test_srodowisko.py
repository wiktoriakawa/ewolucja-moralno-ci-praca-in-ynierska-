import numpy as np
import pytest

from moralnosc import srodowisko
from moralnosc.stan import pusty_stan
from pomocnicze import sprawdz_niezmienniki

H, W = 3, 4


@pytest.fixture
def cfg():
    return {"plansza": {"H": H, "W": W}, "zasoby": {"c_min": 1.0, "c_max": 4.0}}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def stan():
    """Plansza 3×4 wpisana ręcznie (testy nie zależą od losowania), agent na (1, 2)."""
    s = pusty_stan(H=H, W=W, N=1, L=48)
    s.pojemnosc = np.full((H, W), 4.0)
    s.zasoby_pol = np.array([
        [0.0, 1.0, 3.9, 4.0],
        [2.0, 0.5, 3.0, 0.0],   # agent na (1, 2), zasób 3.0
        [4.0, 4.0, 1.0, 2.5],
    ])
    s.pozycje[0] = (1, 2)
    s.zajetosc[1, 2] = 0
    s.zywy[0] = True
    s.zasoby_agentow[0] = 5.0
    sprawdz_niezmienniki(s)
    return s


def test_utworz_plansze_ksztalt_zakres_i_pelne_pola(cfg, rng):
    pojemnosc, zasoby_pol = srodowisko.utworz_plansze(cfg, rng)

    assert pojemnosc.shape == (H, W)
    assert zasoby_pol.shape == (H, W)
    assert pojemnosc.dtype == np.float64
    assert np.all((pojemnosc >= 1.0) & (pojemnosc <= 4.0))
    assert np.array_equal(zasoby_pol, pojemnosc)             # pola pełne
    assert not np.shares_memory(zasoby_pol, pojemnosc)       # osobne tablice


def test_odrost_nie_przekracza_pojemnosci(stan):
    alfa = 1.5

    srodowisko.odrost(stan, alfa)

    assert np.all(stan.zasoby_pol <= stan.pojemnosc)
    assert stan.zasoby_pol[0, 0] == pytest.approx(alfa)   # 0.0 + 1.5
    assert stan.zasoby_pol[0, 2] == pytest.approx(4.0)    # 3.9 + 1.5 → 4.0


def test_zbierz_zachowuje_sume_i_zeruje_pole(stan):
    suma_przed = stan.zasoby_agentow[0] + stan.zasoby_pol[1, 2]

    ilosc = srodowisko.zbierz(stan, 0)

    assert stan.zasoby_pol[1, 2] == 0
    assert stan.zasoby_agentow[0] + stan.zasoby_pol[1, 2] == pytest.approx(suma_przed)
    assert ilosc == pytest.approx(3.0)
    assert isinstance(ilosc, float)
    sprawdz_niezmienniki(stan)
