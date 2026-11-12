"""Testy populacja.py."""
import numpy as np
import pytest

from moralnosc import populacja
from moralnosc.stan import POLNOC, POLUDNIE, PUSTE, ZACHOD, pusty_stan
from pomocnicze import sprawdz_niezmienniki

H, W = 3, 4


@pytest.fixture
def stan():
    """Plansza 3×4, pojemność 4.0, dwóch agentów (A0 = 2.0, A1 = 6.0):

            kol 0   kol 1   kol 2   kol 3
      w 0    0.0    [A0]v    2.0     3.0     A0 na (0, 1), pod nim 1.0, patrzy na S
      w 1    4.0    [A1]<    1.5     2.5     A1 na (1, 1), pod nim 0.5, patrzy na W
      w 2    3.5     2.2     1.0     2.0
    """
    s = pusty_stan(H=H, W=W, N=2, L=48)
    s.pojemnosc = np.full((H, W), 4.0)
    s.zasoby_pol = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 0.5, 1.5, 2.5],
        [3.5, 2.2, 1.0, 2.0],
    ])
    s.pozycje[:] = [[0, 1], [1, 1]]
    s.kierunki[:] = [POLUDNIE, ZACHOD]
    s.zajetosc[0, 1] = 0
    s.zajetosc[1, 1] = 1
    s.zywy[:] = True
    s.zasoby_agentow[:] = [2.0, 6.0]
    sprawdz_niezmienniki(s)
    return s


# --- obserwacje ----------------------------------------------------------------------

def test_obserwacje_agenta_zwroconego_do_sasiada(stan):
    obs, przed = populacja.obserwacje(stan, skala=4.0)

    assert obs.shape == (2, 5)
    assert obs.dtype == np.float64

    # A0 patrzy na A1: [własne, pod sobą, przed sobą, obecność, zasoby A1]
    assert przed[0] == 1
    np.testing.assert_allclose(obs[0], [2.0 / 4, 1.0 / 4, 0.5 / 4, 1.0, 6.0 / 4])

    # A1 patrzy na puste pole (1, 0)
    assert przed[1] == PUSTE
    np.testing.assert_allclose(obs[1], [6.0 / 4, 0.5 / 4, 4.0 / 4, 0.0, 0.0])


# --- rozmiesc_losowo -----------------------------------------------------------------

@pytest.mark.parametrize("N", [5, H * W])   # też pełna plansza
def test_rozmiesc_losowo_bez_kolizji(N):
    s = pusty_stan(H=H, W=W, N=N, L=48)
    rng = np.random.default_rng(1)
    populacja.rozmiesc_losowo(s, rng)
    s.zywy[:] = True
    sprawdz_niezmienniki(s)


# --- ruch ----------------------------------------------------------------------------

def test_ruch_w_sciane_i_na_zajete_pole(stan):
    assert not populacja.ruch(stan, 0, POLNOC)          # ściana
    assert stan.kierunki[0] == POLNOC
    assert tuple(stan.pozycje[0]) == (0, 1)

    assert not populacja.ruch(stan, 1, POLNOC)          # zajęte przez A0
    assert stan.kierunki[1] == POLNOC
    assert tuple(stan.pozycje[1]) == (1, 1)

    sprawdz_niezmienniki(stan)


def test_ruch_na_wolne_pole_zachowuje_niezmienniki(stan):
    A1 = 1
    assert populacja.ruch(stan, A1, POLUDNIE)
    assert tuple(stan.pozycje[A1]) == (2, 1)
    assert stan.zajetosc[1, 1] == PUSTE
    assert stan.zajetosc[2, 1] == A1
    sprawdz_niezmienniki(stan)


def test_usun_martwych_czysci_zajetosc_i_zostawia_zywych(stan):
    stan.zasoby_agentow[0] = 0.0
    maska = stan.zywy & (stan.zasoby_agentow <= 0)

    populacja.usun_martwych(stan, maska)

    assert not stan.zywy[0]
    assert stan.zajetosc[0, 1] == PUSTE
    assert stan.zywy[1]
    assert stan.zajetosc[1, 1] == 1
    sprawdz_niezmienniki(stan)


def test_usun_martwych_pomija_juz_martwego_agenta(stan):
    # A0 umiera, A1 wchodzi na jego pole; pozycja A0 nadal wskazuje (0, 1)
    populacja.usun_martwych(stan, np.array([True, False]))
    assert populacja.ruch(stan, 1, POLNOC)

    # ponowne usunięcie A0 nie może zdjąć A1
    populacja.usun_martwych(stan, np.array([True, False]))

    assert stan.zajetosc[0, 1] == 1
    assert stan.zywy.tolist() == [False, True]
    sprawdz_niezmienniki(stan)


# --- obserwacje przy ścianie ---------------------------------------------------------

def test_obserwacje_przy_scianie(stan):
    # pułapka: wiersz −1 to w NumPy wiersz 2 (zasób 2.2) – ma wyjść 0
    stan.kierunki[0] = POLNOC
    obs, przed = populacja.obserwacje(stan, skala=4.0)
    assert przed[0] == PUSTE
    np.testing.assert_allclose(obs[0, 2:], [0.0, 0.0, 0.0])
