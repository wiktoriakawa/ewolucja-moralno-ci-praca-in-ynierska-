"""Testy mechanicznej wersji pętli, bez pomiaru i zapisu wyników."""

import numpy as np
import pytest

from moralnosc import petla
from moralnosc.stan import (
    DZIEL,
    KRADNIJ,
    NIC,
    N_WE,
    N_WY,
    POLNOC,
    PUSTE,
    WSCHOD,
    ZACHOD,
    ZBIERZ,
    pusty_stan,
)
from pomocnicze import sprawdz_niezmienniki


def ustaw_akcje(stan, akcje):
    """Ustawia biasy sieci tak, aby argmax wybrał podane akcje agentów."""
    stan.genomy.fill(0.0)
    poczatek_biasow = N_WE * N_WY
    for i, akcja in enumerate(akcje):
        stan.genomy[i, poczatek_biasow + akcja] = 1.0


@pytest.fixture
def cfg():
    """Minimalna konfiguracja potrzebna funkcjom krok i wykonaj_akcje."""
    return {
        "obserwacje": {"skala": 4.0},
        "siec": {"tryb": "argmax", "tau": 1.0},
        "koszty": {
            "ruch": 0.25,
            "s": 1.0,
            "c_k": 0.5,
            "p": 1.0,
            "x": 1.5,
            "m": 1.0,
        },
        "zasoby": {"alfa": 0.5},
    }


@pytest.fixture
def stan():
    """Plansza 2x3 z dwoma żyjącymi agentami patrzącymi na siebie."""
    s = pusty_stan(H=2, W=3, N=2, L=48)
    s.pojemnosc[:] = 4.0
    s.zasoby_pol[:] = [[0.0, 1.0, 3.8], [2.0, 2.5, 4.0]]
    s.pozycje[:] = [[0, 0], [0, 1]]
    s.kierunki[:] = [WSCHOD, ZACHOD]
    s.zajetosc[0, 0] = 0
    s.zajetosc[0, 1] = 1
    s.zasoby_agentow[:] = [2.0, 0.5]
    s.zywy[:] = True
    s.ids[:] = [10, 20]
    s.nastepny_id = 21
    ustaw_akcje(s, [NIC, NIC])
    sprawdz_niezmienniki(s)
    return s


def test_krok_nalicza_metabolizm_usuwa_martwych_odrasta_i_zwieksza_numer(
    stan, cfg
):
    """Koniec kroku ma kolejność: metabolizm, śmierć, odrost, numer kroku."""
    zasoby_pol_przed = stan.zasoby_pol.copy()
    rng = np.random.default_rng(0)

    ktos_przezyl = petla.krok(stan, cfg, rng)

    assert ktos_przezyl
    np.testing.assert_allclose(stan.zasoby_agentow, [1.0, -0.5])
    assert stan.zywy.tolist() == [True, False]
    assert stan.zajetosc[0, 0] == 0
    assert stan.zajetosc[0, 1] == PUSTE
    np.testing.assert_allclose(
        stan.zasoby_pol,
        np.minimum(zasoby_pol_przed + cfg["zasoby"]["alfa"], stan.pojemnosc),
    )
    assert stan.krok == 1
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_nalicza_koszt_nieudanej_proby_ruchu(stan, cfg):
    """Zablokowana próba ruchu zmienia kierunek i nadal kosztuje."""
    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=PUSTE,
        akcja=POLNOC,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )
    assert wynik == "porazka"
    assert kwota == 0.0
    assert tuple(stan.pozycje[0]) == (0, 0)
    assert stan.kierunki[0] == POLNOC
    assert stan.zasoby_agentow[0] == pytest.approx(1.75)
    assert stan.zajetosc[0, 0] == 0
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_zbiera_zasoby_i_zwraca_kwote(stan, cfg):
    """Zbieranie przenosi cały zasób pola do agenta i zwraca zebraną kwotę."""
    stan.zasoby_pol[0, 0] = 1.5
    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=PUSTE,
        akcja=ZBIERZ,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )
    assert wynik == "wykonana"
    assert kwota == pytest.approx(1.5)
    assert stan.zasoby_agentow[0] == pytest.approx(3.5)
    assert stan.zasoby_pol[0, 0] == 0.0
    assert tuple(stan.pozycje[0]) == (0, 0)
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_dzieli_z_aktualnym_celem(stan, cfg):
    """Dzielenie przekazuje pełną kwotę s agentowi nadal stojącemu przed dawcą."""
    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=1,
        akcja=DZIEL,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )
    assert wynik == "sukces"
    assert kwota == pytest.approx(1.0)
    np.testing.assert_allclose(stan.zasoby_agentow, [1.0, 1.5])
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_nie_przenosi_dzielenia_na_nowego_agenta(cfg):
    """Dzielenie nie przenosi się na agenta, który zastąpił pierwotny cel."""
    stan = pusty_stan(H=2, W=3, N=3, L=48)
    stan.pojemnosc[:] = 4.0
    stan.zywy[:] = True
    stan.zasoby_agentow[:] = [2.0, 0.5, 3.0]
    stan.ids[:] = [10, 20, 30]
    stan.nastepny_id = 31

    stan.pozycje[:] = [
        [0, 0],  # A0
        [1, 0],  # A1 odszedł z pola przed A0
        [0, 1],  # A2 zajął jego miejsce
    ]
    stan.kierunki[:] = [WSCHOD, POLNOC, ZACHOD]

    stan.zajetosc.fill(PUSTE)
    stan.zajetosc[0, 0] = 0
    stan.zajetosc[1, 0] = 1
    stan.zajetosc[0, 1] = 2
    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=1,
        akcja=DZIEL,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )

    assert wynik == "niewykonana"
    assert kwota == 0.0
    np.testing.assert_allclose(stan.zasoby_agentow, [2.0, 0.5, 3.0])
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_nie_nalicza_kosztu_kradziezy_gdy_cel_odszedl(
    stan, cfg
):
    """Kradzież bez pierwotnego celu nie jest próbą i nie kosztuje."""
    stan.pozycje[1] = [1, 1]
    stan.zajetosc[0, 1] = PUSTE
    stan.zajetosc[1, 1] = 1
    zasoby_przed = stan.zasoby_agentow.copy()

    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=1,
        akcja=KRADNIJ,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )

    assert wynik == "niewykonana"
    assert kwota == 0.0
    np.testing.assert_allclose(stan.zasoby_agentow, zasoby_przed)
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_nic_zwraca_wykonana_bez_zmiany_zasobow(stan, cfg):
    """Akcja NIC zostawia zasoby agentów bez zmian."""
    zasoby_przed = stan.zasoby_agentow.copy()

    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=1,
        akcja=NIC,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )

    assert wynik == "wykonana"
    assert kwota == 0.0
    np.testing.assert_allclose(stan.zasoby_agentow, zasoby_przed)
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_odrzuca_nieznany_kod(stan, cfg):
    """Nieznany kod akcji powoduje czytelny wyjątek."""
    with pytest.raises(ValueError, match="Nieznany kod akcji"):
        petla.wykonaj_akcje(
            stan,
            i=0,
            cel=PUSTE,
            akcja=999,
            cfg=cfg,
            rng=np.random.default_rng(0),
        )


def test_krok_zwraca_false_gdy_wszyscy_umieraja(stan, cfg):
    """Krok zwraca False, gdy po metabolizmie nie zostaje żywy agent."""
    stan.zasoby_agentow[:] = [1.0, 0.5]

    ktos_przezyl = petla.krok(stan, cfg, np.random.default_rng(0))

    assert ktos_przezyl is False
    assert stan.zywy.tolist() == [False, False]
    assert np.all(stan.zajetosc == PUSTE)
    assert stan.krok == 1
    sprawdz_niezmienniki(stan)
