import numpy as np
import pytest

from moralnosc import srodowisko
from moralnosc.stan import pusty_stan
from pomocnicze import sprawdz_niezmienniki

H, W = 3, 4  # mała plansza 3×4 – wyniki da się policzyć w pamięci


@pytest.fixture
def cfg():
    """Minimalna konfiguracja: tylko klucze, których potrzebuje środowisko."""
    return {"plansza": {"H": H, "W": W}, "zasoby": {"c_min": 1.0, "c_max": 4.0}}


@pytest.fixture
def rng():
    """Generator z ustalonym ziarnem – każdy test dostaje świeży."""
    return np.random.default_rng(12345)


@pytest.fixture
def stan():
    """Stan 3×4 z ręcznie ułożoną planszą i jednym agentem na polu (1, 2).

    Plansza jest wpisana ręcznie, a nie z utworz_plansze – dzięki temu testy
    odrostu i zbierania nie zależą od losowania i od tego, czy utworz_plansze
    już działa.
    """
    s = pusty_stan(H=H, W=W, N=1, L=48)
    s.pojemnosc = np.full((H, W), 4.0)
    s.zasoby_pol = np.array([
        [0.0, 1.0, 3.9, 4.0],
        [2.0, 0.5, 3.0, 0.0],   # agent stoi na polu (1, 2) z zasobem 3.0
        [4.0, 4.0, 1.0, 2.5],
    ])
    s.pozycje[0] = (1, 2)
    s.zajetosc[1, 2] = 0
    s.zywy[0] = True
    s.zasoby_agentow[0] = 5.0
    sprawdz_niezmienniki(s)  # sam stan testowy też musi być poprawny
    return s


def test_utworz_plansze_ksztalt_zakres_i_pelne_pola(cfg, rng):
    # przygotuj: cfg i rng dostarczają fixture powyżej

    # wykonaj
    pojemnosc, zasoby_pol = srodowisko.utworz_plansze(cfg, rng)

    # sprawdź
    assert pojemnosc.shape == (H, W)
    assert zasoby_pol.shape == (H, W)
    assert pojemnosc.dtype == np.float64
    assert np.all((pojemnosc >= 1.0) & (pojemnosc <= 4.0))   # zakres [c_min, c_max]
    assert np.array_equal(zasoby_pol, pojemnosc)             # na starcie pola pełne
    assert not np.shares_memory(zasoby_pol, pojemnosc)       # ale to dwie osobne tablice



def test_odrost_nie_przekracza_pojemnosci(stan):
    # przygotuj
    alfa = 1.5

    # wykonaj
    srodowisko.odrost(stan, alfa)

    # sprawdź
    assert np.all(stan.zasoby_pol <= stan.pojemnosc)
    assert stan.zasoby_pol[0, 0] == pytest.approx(alfa)   # puste pole: 0.0 + 1.5
    assert stan.zasoby_pol[0, 2] == pytest.approx(4.0)    # 3.9 + 1.5 = 5.4, przycięte do 4.0



def test_zbierz_zachowuje_sume_i_zeruje_pole(stan):
    # przygotuj
    suma_przed = stan.zasoby_agentow[0] + stan.zasoby_pol[1, 2]   # 5.0 + 3.0

    # wykonaj
    ilosc = srodowisko.zbierz(stan, 0)

    # sprawdź
    assert stan.zasoby_pol[1, 2] == 0
    assert stan.zasoby_agentow[0] + stan.zasoby_pol[1, 2] == pytest.approx(suma_przed)
    assert ilosc == pytest.approx(3.0)
    assert isinstance(ilosc, float)
    sprawdz_niezmienniki(stan)