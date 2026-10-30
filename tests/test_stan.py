"""Testy modułu stan.py: stałe, klasa Stan i funkcja pusty_stan."""
import numpy as np
import pytest

from moralnosc import stan as s
from pomocnicze import sprawdz_niezmienniki

H, W, N, L = 3, 4, 2, 48  # mała plansza 3×4, dwóch agentów, genom sieci liniowej


@pytest.fixture
def pusty():
    """Świeży pusty stan dla każdego testu (testy nie wpływają na siebie nawzajem)."""
    return s.pusty_stan(H=H, W=W, N=N, L=L)


# --- kierunki -------------------------------------------------------------

def test_kierunki_ksztalt_i_typ():
    assert s.KIERUNKI.shape == (4, 2)
    assert s.KIERUNKI.dtype == np.int64


def test_kazdy_kierunek_to_krok_o_jedno_pole():
    # w każdym wierszu dokładnie jedna współrzędna zmienia się o 1
    assert np.all(np.abs(s.KIERUNKI).sum(axis=1) == 1)


def test_przeciwne_kierunki_sumuja_sie_do_zera():
    assert np.all(s.KIERUNKI[s.POLNOC] + s.KIERUNKI[s.POLUDNIE] == 0)
    assert np.all(s.KIERUNKI[s.WSCHOD] + s.KIERUNKI[s.ZACHOD] == 0)


def test_kierunki_tylko_do_odczytu():
    with pytest.raises(ValueError):
        s.KIERUNKI[0, 0] = 5


# --- kody akcji i rozmiar sieci --------------------------------------------

def test_kody_akcji_unikalne_i_w_zakresie():
    kody = [s.RUCH_N, s.RUCH_E, s.RUCH_S, s.RUCH_W, s.ZBIERZ, s.DZIEL, s.KRADNIJ, s.NIC]
    assert sorted(kody) == list(range(s.LICZBA_AKCJI))


def test_kody_ruchu_rowne_kodom_kierunkow():
    assert (s.RUCH_N, s.RUCH_E, s.RUCH_S, s.RUCH_W) == (s.POLNOC, s.WSCHOD, s.POLUDNIE, s.ZACHOD)


def test_rozmiar_sieci_i_kolumny_obserwacji():
    assert s.N_WY == s.LICZBA_AKCJI
    kolumny = [s.OBS_ZASOBY_WLASNE, s.OBS_ZASOBY_POD, s.OBS_ZASOBY_PRZED,
               s.OBS_OBECNOSC_SASIADA, s.OBS_ZASOBY_SASIADA]
    assert sorted(kolumny) == list(range(s.N_WE))


# --- pusty_stan ------------------------------------------------------------

def test_pusty_stan_ksztalty(pusty):
    assert pusty.pojemnosc.shape == (H, W)
    assert pusty.zasoby_pol.shape == (H, W)
    assert pusty.zajetosc.shape == (H, W)
    assert pusty.pozycje.shape == (N, 2)
    for tablica in (pusty.kierunki, pusty.zasoby_agentow, pusty.zywy, pusty.ids, pusty.rodzic):
        assert tablica.shape == (N,)
    assert pusty.genomy.shape == (N, L)


def test_pusty_stan_typy(pusty):
    for tablica in (pusty.pojemnosc, pusty.zasoby_pol, pusty.zasoby_agentow, pusty.genomy):
        assert tablica.dtype == np.float64
    for tablica in (pusty.zajetosc, pusty.pozycje, pusty.kierunki, pusty.ids, pusty.rodzic):
        assert tablica.dtype == np.int64
    assert pusty.zywy.dtype == np.bool_


def test_pusty_stan_wartosci_poczatkowe(pusty):
    assert np.all(pusty.zajetosc == s.PUSTE)   # wszystkie pola puste
    assert np.all(pusty.ids == s.PUSTE)        # identyfikatory jeszcze nienadane
    assert np.all(pusty.rodzic == s.PUSTE)     # pokolenie zerowe nie ma rodziców
    assert not pusty.zywy.any()                # nikt jeszcze nie stoi na planszy
    assert (pusty.nastepny_id, pusty.pokolenie, pusty.krok) == (0, 0, 0)


def test_pusty_stan_spelnia_niezmienniki(pusty):
    sprawdz_niezmienniki(pusty)


def test_pusty_stan_tylko_argumenty_nazwane():
    with pytest.raises(TypeError):
        s.pusty_stan(H, W, N, L)


# --- klasa Stan ------------------------------------------------------------

def test_stan_blokuje_literowki_w_nazwach_pol(pusty):
    with pytest.raises(AttributeError):
        pusty.zasoby_pl = np.zeros(N)   # literówka: zasoby_pl zamiast zasoby_pol


def test_stan_wymaga_pol_podanych_z_nazwa():
    with pytest.raises(TypeError):
        s.Stan(np.zeros((H, W)))


# --- sprawdzenie samego pomocnika ------------------------------------------

def test_pomocnik_wykrywa_agenta_spoza_mapy_zajetosci(pusty):
    # żywy agent, którego nie ma na mapie zajętości – niezmiennik 1 musi to wykryć
    pusty.zywy[0] = True
    with pytest.raises(AssertionError):
        sprawdz_niezmienniki(pusty)
