"""Testy modułu ewolucja.py – testy minimalne na Lab02 (instrukcja, krok 4).
"""
import numpy as np
import pytest

from moralnosc import ewolucja
from moralnosc.stan import POLNOC, PUSTE, pusty_stan
from pomocnicze import sprawdz_niezmienniki

H, W, N, L = 3, 4, 4, 48  # mała plansza 3×4, czterech agentów, genom sieci liniowej
T = 3.0                   # próg kwalifikacji


@pytest.fixture
def stan():
    """Koniec pokolenia na planszy 3×4, czterech agentów (ids 0–3):

            kol 0   kol 1   kol 2   kol 3
      w 0   [A0]     .       .      [A1]      A0: 5.0 (≥ T)     A1: 2.0 (< T)
      w 1    .       .       .       .
      w 2   [A2]     .       .       .        A2: 3.0 (= T)     A3: martwy, 9.0

    A3 zginął w trakcie pokolenia (stał na (2, 3)), więc nie ma go na mapie
    zajętości. Ma najwięcej zasobów, ale martwy nie może zostać rodzicem.
    Genom agenta i składa się z samych wartości i – łatwo sprawdzić, czyj
    genom dostał potomek.
    """
    s = pusty_stan(H=H, W=W, N=N, L=L)
    s.pojemnosc = np.full((H, W), 4.0)
    s.zasoby_pol = np.full((H, W), 1.0)
    s.pozycje[:] = [[0, 0], [0, 3], [2, 0], [2, 3]]
    s.kierunki[:] = POLNOC
    s.zywy[:] = [True, True, True, False]
    for i in (0, 1, 2):  # martwego A3 nie wpisujemy na mapę
        s.zajetosc[s.pozycje[i, 0], s.pozycje[i, 1]] = i
    s.zasoby_agentow[:] = [5.0, 2.0, 3.0, 9.0]
    s.ids[:] = np.arange(N)
    s.nastepny_id = N
    s.genomy[:] = np.arange(N)[:, None]  # wiersz i: 48 razy wartość i
    sprawdz_niezmienniki(s)  # sam stan testowy też musi być poprawny
    return s


@pytest.fixture
def cfg():
    """Minimalna konfiguracja: tylko klucze, których potrzebuje nowe_pokolenie."""
    return {"ewolucja": {"r": "auto", "sigma": 0.2, "w_max": 5.0, "R": 1, "E0": 10.0}}


# --- mutuj ---------------------------------------------------------------------------

def test_mutuj_przy_r_zero_kopiuje_genom():
    # przygotuj: wagi wewnątrz [−w_max, w_max], żeby przycięcie niczego nie zmieniło
    rng = np.random.default_rng(0)
    rodzice = rng.uniform(-1.0, 1.0, size=(3, L))
    kopia = rodzice.copy()

    # wykonaj
    potomkowie = ewolucja.mutuj(rodzice, r=0.0, sigma=0.5, w_max=5.0, rng=rng)

    # sprawdź
    np.testing.assert_array_equal(potomkowie, rodzice)  # bez mutacji – ten sam genom
    assert not np.shares_memory(potomkowie, rodzice)    # ale w nowej tablicy
    np.testing.assert_array_equal(rodzice, kopia)       # wejście nietknięte


# --- kwalifikacja --------------------------------------------------------------------

def test_kwalifikacja_prog_wlacznie_i_bez_martwych(stan):
    # wykonaj
    maska, awaryjny = ewolucja.kwalifikacja(stan, T)

    # sprawdź: A0 (5.0) i A2 (dokładnie T) – tak; A1 (2.0) – nie;
    # A3 – nie, bo martwy, choć ma najwięcej zasobów
    assert maska.tolist() == [True, False, True, False]
    assert awaryjny is False


def test_kwalifikacja_awaryjna_gdy_nikt_nie_osiaga_progu(stan):
    # wykonaj: próg 6.0 jest wyższy niż zasoby wszystkich żywych
    maska, awaryjny = ewolucja.kwalifikacja(stan, 6.0)

    # sprawdź: rodzicami mogą być wszyscy żywi, ale nadal nie martwy A3
    assert maska.tolist() == [True, True, True, False]
    assert awaryjny is True
    # maska to kopia, a nie ta sama tablica co stan.zywy
    assert not np.shares_memory(maska, stan.zywy)


# --- nowe_pokolenie ------------------------------------------------------------------

def test_nowe_pokolenie_rodzice_tylko_z_zakwalifikowanych(stan, cfg):
    # przygotuj: r = 0, żeby genom potomka był dokładnie genomem rodzica
    cfg["ewolucja"]["r"] = 0.0
    maska, _ = ewolucja.kwalifikacja(stan, T)  # rodzicami mogą być tylko A0 i A2
    rng = np.random.default_rng(7)

    # wykonaj
    rodzice = ewolucja.nowe_pokolenie(stan, maska, cfg, rng)

    # sprawdź rodziców
    assert rodzice.shape == (N,)
    assert set(rodzice.tolist()) <= {0, 2}

    # sprawdź potomków: nowe identyfikatory, rodowód, zasoby startowe, numer pokolenia
    assert stan.ids.tolist() == [4, 5, 6, 7]
    assert stan.nastepny_id == 8
    np.testing.assert_array_equal(stan.rodzic, rodzice)  # w fixture stare ids = indeksy
    assert stan.zywy.all()
    np.testing.assert_allclose(stan.zasoby_agentow, 10.0)
    assert stan.pokolenie == 1

    # genom agenta i w fixture to same wartości i, więc potomek k ma same wartości rodzice[k]
    np.testing.assert_array_equal(stan.genomy, np.repeat(rodzice[:, None], L, axis=1))
    sprawdz_niezmienniki(stan)


# --- rozmiesc_potomkow ---------------------------------------------------------------

def test_rozmiesc_potomkow_na_pelnej_planszy():
    # przygotuj: 12 potomków na 12 polach, wszyscy rodzice w rogu (0, 0) –
    # kwadrat o promieniu R = 1 ma tylko 4 pola, więc promień musi rosnąć
    pozycje_rodzicow = np.zeros((H * W, 2), dtype=np.int64)
    rng = np.random.default_rng(1)

    # wykonaj
    pozycje, zajetosc = ewolucja.rozmiesc_potomkow(pozycje_rodzicow, 1, H, W, rng)

    # sprawdź: każde pole zajął inny potomek, a mapa zgadza się z pozycjami
    assert sorted(zajetosc.ravel().tolist()) == list(range(H * W))
    np.testing.assert_array_equal(zajetosc[pozycje[:, 0], pozycje[:, 1]], np.arange(H * W))


# --- pokolenie_zerowe ----------------------------------------------------------------

def test_pokolenie_zerowe_spelnia_niezmienniki():
    # przygotuj: minimalna konfiguracja – tylko klucze, których potrzebuje funkcja
    cfg = {
        "plansza": {"H": H, "W": W},
        "populacja": {"N": 5},
        "zasoby": {"c_min": 1.0, "c_max": 4.0},
        "ewolucja": {"E0": 10.0},
    }

    # wykonaj
    s = ewolucja.pokolenie_zerowe(cfg, np.random.default_rng(0))

    # sprawdź
    assert s.genomy.shape == (5, L)
    assert np.any(s.genomy != 0)  # genomy wylosowane, a nie zera z pusty_stan
    assert s.zywy.all()
    np.testing.assert_allclose(s.zasoby_agentow, 10.0)
    assert s.ids.tolist() == [0, 1, 2, 3, 4]
    assert np.all(s.rodzic == PUSTE)
    assert (s.nastepny_id, s.pokolenie) == (5, 0)
    np.testing.assert_array_equal(s.zasoby_pol, s.pojemnosc)  # plansza pełna na starcie
    sprawdz_niezmienniki(s)