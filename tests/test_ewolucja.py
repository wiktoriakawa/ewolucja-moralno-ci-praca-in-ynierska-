"""Testy ewolucja.py."""
import numpy as np
import pytest

from moralnosc import ewolucja
from moralnosc.stan import POLNOC, PUSTE, pusty_stan
from pomocnicze import sprawdz_niezmienniki

H, W, N, L = 3, 4, 4, 48
T = 3.0                   # próg kwalifikacji


@pytest.fixture
def stan():
    """Koniec pokolenia, plansza 3×4, ids 10–40:

            kol 0   kol 1   kol 2   kol 3
      w 0   [A0]     .       .      [A1]      A0: 5.0 (≥ T)     A1: 2.0 (< T)
      w 1    .       .       .       .
      w 2   [A2]     .       .       .        A2: 3.0 (= T)     A3: martwy, 9.0

    Genom agenta i to same wartości i – widać, czyj genom dostał potomek.
    """
    s = pusty_stan(H=H, W=W, N=N, L=L)
    s.pojemnosc = np.full((H, W), 4.0)
    s.zasoby_pol = np.full((H, W), 1.0)
    s.pozycje[:] = [[0, 0], [0, 3], [2, 0], [2, 3]]
    s.kierunki[:] = POLNOC
    s.zywy[:] = [True, True, True, False]
    for i in (0, 1, 2):  # martwy A3 poza mapą
        s.zajetosc[s.pozycje[i, 0], s.pozycje[i, 1]] = i
    s.zasoby_agentow[:] = [5.0, 2.0, 3.0, 9.0]
    s.ids[:] = [10, 20, 30, 40]  # celowo różne od indeksów
    s.nastepny_id = 41
    s.genomy[:] = np.arange(N)[:, None]
    sprawdz_niezmienniki(s)
    return s


@pytest.fixture
def cfg():
    return {"ewolucja": {"r": "auto", "sigma": 0.2, "w_max": 5.0, "R": 1, "E0": 10.0}}


# --- mutuj ---------------------------------------------------------------------------

def test_mutuj_przy_r_zero_kopiuje_genom():
    rng = np.random.default_rng(0)
    rodzice = rng.uniform(-1.0, 1.0, size=(3, L))   # w [−w_max, w_max] – bez przycięcia
    kopia = rodzice.copy()

    potomkowie = ewolucja.mutuj(rodzice, r=0.0, sigma=0.5, w_max=5.0, rng=rng)

    np.testing.assert_array_equal(potomkowie, rodzice)
    assert not np.shares_memory(potomkowie, rodzice)    # nowa tablica
    np.testing.assert_array_equal(rodzice, kopia)       # wejście nietknięte


# --- kwalifikacja --------------------------------------------------------------------

def test_kwalifikacja_prog_wlacznie_i_bez_martwych(stan):
    maska, awaryjny = ewolucja.kwalifikacja(stan, T)

    # A2 ma dokładnie T – tak; martwy A3 – nie, mimo najwięcej zasobów
    assert maska.tolist() == [True, False, True, False]
    assert awaryjny is False


def test_kwalifikacja_awaryjna_gdy_nikt_nie_osiaga_progu(stan):
    maska, awaryjny = ewolucja.kwalifikacja(stan, 6.0)

    # wszyscy żywi, ale nie martwy A3
    assert maska.tolist() == [True, True, True, False]
    assert awaryjny is True
    assert not np.shares_memory(maska, stan.zywy)


# --- nowe_pokolenie ------------------------------------------------------------------

def test_nowe_pokolenie_rodzice_tylko_z_zakwalifikowanych(stan, cfg):
    cfg["ewolucja"]["r"] = 0.0   # genom potomka = genom rodzica
    maska, _ = ewolucja.kwalifikacja(stan, T)  # tylko A0 i A2
    rng = np.random.default_rng(7)
    stare_ids = stan.ids.copy()

    rodzice = ewolucja.nowe_pokolenie(stan, maska, cfg, rng)

    assert rodzice.shape == (N,)
    assert set(rodzice.tolist()) <= {0, 2}

    assert stan.ids.tolist() == [41, 42, 43, 44]
    assert stan.nastepny_id == 45
    # rodzice – indeksy starego pokolenia; stan.rodzic – trwałe id
    np.testing.assert_array_equal(stan.rodzic, stare_ids[rodzice])
    assert stan.zywy.all()
    np.testing.assert_allclose(stan.zasoby_agentow, 10.0)
    assert stan.pokolenie == 1

    np.testing.assert_array_equal(stan.genomy, np.repeat(rodzice[:, None], L, axis=1))
    sprawdz_niezmienniki(stan)


# --- rozmiesc_potomkow ---------------------------------------------------------------

def test_rozmiesc_potomkow_na_pelnej_planszy():
    # 12 potomków na 12 polach, rodzice w rogu – przy R = 1 promień musi rosnąć
    pozycje_rodzicow = np.zeros((H * W, 2), dtype=np.int64)
    rng = np.random.default_rng(1)

    pozycje, zajetosc = ewolucja.rozmiesc_potomkow(pozycje_rodzicow, 1, H, W, rng)

    assert sorted(zajetosc.ravel().tolist()) == list(range(H * W))
    np.testing.assert_array_equal(zajetosc[pozycje[:, 0], pozycje[:, 1]], np.arange(H * W))


# --- pokolenie_zerowe ----------------------------------------------------------------

def test_pokolenie_zerowe_spelnia_niezmienniki():
    cfg = {
        "plansza": {"H": H, "W": W},
        "populacja": {"N": 5},
        "zasoby": {"c_min": 1.0, "c_max": 4.0},
        "ewolucja": {"E0": 10.0},
        "siec": {"h": 0},
    }

    s = ewolucja.pokolenie_zerowe(cfg, np.random.default_rng(0))

    assert s.genomy.shape == (5, L)
    assert np.any(s.genomy != 0)  # wylosowane, nie zera z pusty_stan
    assert s.zywy.all()
    np.testing.assert_allclose(s.zasoby_agentow, 10.0)
    assert s.ids.tolist() == [0, 1, 2, 3, 4]
    assert np.all(s.rodzic == PUSTE)
    assert (s.nastepny_id, s.pokolenie) == (5, 0)
    np.testing.assert_array_equal(s.zasoby_pol, s.pojemnosc)  # pola pełne
    sprawdz_niezmienniki(s)


def test_pokolenie_zerowe_z_warstwa_ukryta_ma_dluzszy_genom():
    cfg = {
        "plansza": {"H": H, "W": W},
        "populacja": {"N": 5},
        "zasoby": {"c_min": 1.0, "c_max": 4.0},
        "ewolucja": {"E0": 10.0},
        "siec": {"h": 8},
    }

    s = ewolucja.pokolenie_zerowe(cfg, np.random.default_rng(0))

    assert s.genomy.shape == (5, 120)   # 6·8 + 9·8
    assert np.any(s.genomy != 0)
    sprawdz_niezmienniki(s)
    # plansza losowana przed genomami – nie zależy od h
    s_lin = ewolucja.pokolenie_zerowe({**cfg, "siec": {"h": 0}}, np.random.default_rng(0))
    np.testing.assert_array_equal(s.pojemnosc, s_lin.pojemnosc)
