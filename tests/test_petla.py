"""Testy mechanicznej wersji pętli, bez pomiaru i zapisu wyników."""

import numpy as np
import pytest

from moralnosc import ewolucja, petla
from moralnosc.pomiar import (
    PRZYCZYNA_KONIEC_POKOLENIA,
    PRZYCZYNA_WYCZERPANIE,
    PRZYCZYNA_ZYJE,
    pusty_pomiar,
)
from moralnosc.stan import (
    DZIEL,
    KRADNIJ,
    NIC,
    NIEWYKONANA,
    N_WE,
    N_WY,
    POLNOC,
    PORAZKA,
    PUSTE,
    SUKCES,
    WSCHOD,
    WYKONANA,
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
def cfg_przebiegu():
    """Mała kompletna konfiguracja przebiegu: 3 pokolenia po 2 kroki."""
    return {
        "przebieg": {"ziarno": 67, "pokolenia": 3, "kroki": 2},
        "plansza": {"H": 3, "W": 3},
        "populacja": {"N": 2},
        "zasoby": {"c_min": 1.0, "c_max": 2.0, "alfa": 1.0},
        "obserwacje": {"skala": 2.0},
        "koszty": {
            "ruch": 0.0,
            "s": 1.0,
            "c_k": 0.5,
            "p": 1.0,
            "x": 1.0,
            "m": 0.1,
        },
        "siec": {"tryb": "argmax", "tau": 1.0},
        "ewolucja": {
            "T": 0.0,
            "E0": 100.0,
            "r": 0.0,
            "sigma": 0.0,
            "w_max": 4.0,
            "R": 1,
        },
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
    assert wynik == PORAZKA
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
    assert wynik == WYKONANA
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
    assert wynik == SUKCES
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

    assert wynik == NIEWYKONANA
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

    assert wynik == NIEWYKONANA
    assert kwota == 0.0
    np.testing.assert_allclose(stan.zasoby_agentow, zasoby_przed)
    sprawdz_niezmienniki(stan)


def test_wykonaj_akcje_udana_kradziez_zwraca_kwote(stan, cfg):
    """Udana kradzież pobiera koszt próby i przenosi kwotę x od celu."""
    stan.zasoby_agentow[:] = [2.0, 3.0]

    wynik, kwota = petla.wykonaj_akcje(
        stan,
        i=0,
        cel=1,
        akcja=KRADNIJ,
        cfg=cfg,
        rng=np.random.default_rng(0),
    )

    assert wynik == SUKCES
    assert kwota == pytest.approx(1.5)
    np.testing.assert_allclose(stan.zasoby_agentow, [3.0, 1.5])
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

    assert wynik == WYKONANA
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


def test_krok_rejestruje_akcje_i_dylematy_z_migawki(stan, cfg):
    """Krok przekazuje do pomiaru akcje, wyniki i zasoby z początku kroku."""
    stan.zasoby_agentow[:] = [3.0, 2.0]
    ustaw_akcje(stan, [DZIEL, NIC])
    pomiar = pusty_pomiar(N=2)

    petla.krok(stan, cfg, np.random.default_rng(0), pomiar=pomiar)

    assert pomiar.akcje_licznik.sum() == 2
    assert pomiar.akcje_licznik[DZIEL] == 1
    assert pomiar.akcje_licznik[NIC] == 1
    np.testing.assert_array_equal(pomiar.S, [1, 0])
    np.testing.assert_array_equal(pomiar.K, [0, 0])
    np.testing.assert_array_equal(pomiar.N_o, [0, 1])
    np.testing.assert_array_equal(pomiar.D, [1, 1])
    np.testing.assert_array_equal(pomiar.S + pomiar.K + pomiar.N_o, pomiar.D)

    zdarzenie_dzielenia = next(z for z in pomiar.zdarzenia if z["id"] == 10)
    assert zdarzenie_dzielenia["id_celu"] == 20
    assert zdarzenie_dzielenia["zasoby_agenta"] == pytest.approx(3.0)
    assert zdarzenie_dzielenia["zasoby_celu"] == pytest.approx(2.0)
    assert zdarzenie_dzielenia["akcja"] == DZIEL
    assert zdarzenie_dzielenia["klasa"] == "dzielenie"
    assert zdarzenie_dzielenia["wynik"] == SUKCES
    assert zdarzenie_dzielenia["kwota"] == pytest.approx(1.0)


def test_krok_rejestruje_smierc_przed_zwiekszeniem_numeru_kroku(stan, cfg):
    """Agent umierający po metabolizmie otrzymuje numer bieżącego kroku."""
    pomiar = pusty_pomiar(N=2)

    petla.krok(stan, cfg, np.random.default_rng(0), pomiar=pomiar)

    np.testing.assert_array_equal(pomiar.krok_smierci, [-1, 0])
    np.testing.assert_array_equal(
        pomiar.przyczyna_smierci,
        [PRZYCZYNA_ZYJE, PRZYCZYNA_WYCZERPANIE],
    )
    assert stan.zywy.tolist() == [True, False]
    assert stan.krok == 1


def test_wykonaj_pokolenie_konczy_sie_po_wymarciu(cfg_przebiegu):
    """Wymarcie kończy pokolenie przed limitem kroków i bez rodziców."""
    cfg_przebiegu["ewolucja"]["E0"] = 1.0
    cfg_przebiegu["koszty"]["m"] = 100.0
    cfg_przebiegu["przebieg"]["kroki"] = 5
    rng = np.random.default_rng(cfg_przebiegu["przebieg"]["ziarno"])
    stan = ewolucja.pokolenie_zerowe(cfg_przebiegu, rng)

    pomiar, maska, awaryjny, przetrwal = petla.wykonaj_pokolenie(
        stan, cfg_przebiegu, rng
    )

    assert przetrwal is False
    assert awaryjny is False
    assert not maska.any()
    assert stan.krok == 1
    assert not stan.zywy.any()
    assert pomiar.akcje_licznik.sum() == cfg_przebiegu["populacja"]["N"]
    np.testing.assert_array_equal(pomiar.krok_smierci, [0, 0])
    np.testing.assert_array_equal(
        pomiar.przyczyna_smierci,
        [PRZYCZYNA_WYCZERPANIE, PRZYCZYNA_WYCZERPANIE],
    )


def test_uruchom_przebieg_wykonuje_pokolenia_i_callback(cfg_przebiegu):
    """Callback widzi każde zakończone pokolenie przed utworzeniem potomków."""
    migawki = []

    def zapamietaj(stan, pomiar, maska, awaryjny):
        migawki.append(
            {
                "pokolenie": stan.pokolenie,
                "krok": stan.krok,
                "ids": stan.ids.copy(),
                "akcje": int(pomiar.akcje_licznik.sum()),
                "przyczyny": pomiar.przyczyna_smierci.copy(),
                "rodzice": int(maska.sum()),
                "awaryjny": awaryjny,
            }
        )

    stan = petla.uruchom_przebieg(cfg_przebiegu, po_pokoleniu=zapamietaj)

    assert [m["pokolenie"] for m in migawki] == [0, 1, 2]
    assert [m["krok"] for m in migawki] == [2, 2, 2]
    assert [m["akcje"] for m in migawki] == [4, 4, 4]
    assert [m["rodzice"] for m in migawki] == [2, 2, 2]
    assert [m["awaryjny"] for m in migawki] == [False, False, False]
    np.testing.assert_array_equal(migawki[0]["ids"], [0, 1])
    np.testing.assert_array_equal(migawki[1]["ids"], [2, 3])
    np.testing.assert_array_equal(migawki[2]["ids"], [4, 5])
    for migawka in migawki:
        np.testing.assert_array_equal(
            migawka["przyczyny"],
            [PRZYCZYNA_KONIEC_POKOLENIA, PRZYCZYNA_KONIEC_POKOLENIA],
        )
    assert stan.pokolenie == 2
    assert stan.krok == 2
