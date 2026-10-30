"""Testy kontenera Pomiar i funkcji pusty_pomiar."""

import numpy as np
import pytest

from moralnosc.pomiar import (
    KOLUMNY_DYLEMATY,
    PRZYCZYNA_KONIEC_POKOLENIA,
    PRZYCZYNA_WYCZERPANIE,
    PRZYCZYNA_ZYJE,
    Pomiar,
    pusty_pomiar,
    status_dylematu,
    zarejestruj,
    zarejestruj_smierc,
)
from moralnosc.stan import (
    DZIEL,
    KRADNIJ,
    LICZBA_AKCJI,
    NIC,
    PUSTE,
    SUKCES,
    pusty_stan,
)


N = 3


@pytest.fixture
def pomiar():
    """Świeży pusty pomiar dla trzech agentów."""
    return pusty_pomiar(N=N)


@pytest.fixture
def stan():
    """Minimalny stan potrzebny do rejestrowania dylematów."""
    s = pusty_stan(H=2, W=2, N=N, L=48)
    s.zywy[:] = [True, True, False]
    s.ids[:] = [10, 20, 30]
    s.pokolenie = 2
    s.krok = 7
    return s


def test_pusty_pomiar_ma_wlasciwe_ksztalty(pomiar):
    """Tablice agentów mają długość N, a licznik akcji osobne pole na każdą akcję."""
    for tablica in (
        pomiar.S,
        pomiar.K,
        pomiar.N_o,
        pomiar.D,
        pomiar.krok_smierci,
        pomiar.przyczyna_smierci,
    ):
        assert tablica.shape == (N,)

    assert pomiar.akcje_licznik.shape == (LICZBA_AKCJI,)


def test_pusty_pomiar_ma_calkowite_typy_tablic(pomiar):
    """Wszystkie liczniki i kody są przechowywane jako liczby całkowite int64."""
    for tablica in (
        pomiar.S,
        pomiar.K,
        pomiar.N_o,
        pomiar.D,
        pomiar.krok_smierci,
        pomiar.przyczyna_smierci,
        pomiar.akcje_licznik,
    ):
        assert tablica.dtype == np.int64


def test_pusty_pomiar_ma_wlasciwe_wartosci_poczatkowe(pomiar):
    """Nowe pokolenie zaczyna bez decyzji, zdarzeń i zarejestrowanych śmierci."""
    assert np.all(pomiar.S == 0)
    assert np.all(pomiar.K == 0)
    assert np.all(pomiar.N_o == 0)
    assert np.all(pomiar.D == 0)
    assert np.all(pomiar.krok_smierci == -1)
    assert np.all(pomiar.przyczyna_smierci == PRZYCZYNA_ZYJE)
    assert np.all(pomiar.akcje_licznik == 0)
    assert pomiar.zdarzenia == []


def test_puste_pomiary_maja_niezalezne_listy_zdarzen():
    """Zdarzenie dodane w jednym pokoleniu nie trafia do innego pomiaru."""
    pierwszy = pusty_pomiar(N=N)
    drugi = pusty_pomiar(N=N)

    pierwszy.zdarzenia.append({"akcja": 5})

    assert pierwszy.zdarzenia == [{"akcja": 5}]
    assert drugi.zdarzenia == []
    assert pierwszy.zdarzenia is not drugi.zdarzenia


def test_pomiar_blokuje_tworzenie_nowych_pol(pomiar):
    """Pomiar nie pozwala dynamicznie tworzyć nowych, niezdefiniowanych pól."""
    with pytest.raises(AttributeError):
        # Slots chroni przed utworzeniem nowego pola, np. wskutek literówki.
        pomiar.nowe_pole = np.zeros(N, dtype=np.int64)


def test_pomiar_wymaga_pol_podanych_z_nazwa():
    """Każdy argument konstruktora Pomiar wymaga jawnego podania nazwy pola."""
    tablica = np.zeros(N, dtype=np.int64)

    with pytest.raises(TypeError):
        Pomiar(
            tablica,
            tablica,
            tablica,
            tablica,
            tablica,
            tablica,
            np.zeros(LICZBA_AKCJI, dtype=np.int64),
            [],
        )


def test_pusty_pomiar_wymaga_argumentu_nazwanego():
    """Liczbę agentów przekazuje się jawnie jako N, aby uniknąć pomyłek."""
    with pytest.raises(TypeError):
        pusty_pomiar(N)


def test_status_dylematu_wykrywa_agenta_spelniajacego_warunki(stan):
    """Dylemat wymaga żywego agenta, celu i odpowiednich zasobów migawki."""
    przed = np.array([1, PUSTE, 0], dtype=np.int64)
    zasoby_start = np.array([2.0, 3.0, 5.0])

    dylemat = status_dylematu(stan, przed, zasoby_start, s=1.0)

    assert dylemat.tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("zasoby_start", "oczekiwany"),
    [
        ([0.5, 3.0, 5.0], False),
        ([2.0, 0.0, 5.0], False),
        ([1.0, 3.0, 5.0], True),
    ],
)
def test_status_dylematu_sprawdza_zasoby_agenta_i_celu(
    stan, zasoby_start, oczekiwany
):
    """Agent musi mieć co najmniej s, a jego cel więcej niż zero zasobów."""
    przed = np.array([1, PUSTE, PUSTE], dtype=np.int64)

    dylemat = status_dylematu(
        stan,
        przed,
        np.array(zasoby_start, dtype=np.float64),
        s=1.0,
    )

    assert bool(dylemat[0]) is oczekiwany


def test_zarejestruj_dzielenie_i_pelne_zdarzenie(pomiar, stan):
    """Dzielenie w dylemacie zwiększa S i zapisuje dane z migawki."""
    zasoby_start = np.array([2.0, 3.0, 0.0])

    zarejestruj(
        pomiar,
        stan,
        zasoby_start,
        i=0,
        j=1,
        akcja=DZIEL,
        dylemat_i=True,
        wynik=SUKCES,
        kwota=1.0,
    )

    assert pomiar.S.tolist() == [1, 0, 0]
    assert pomiar.D.tolist() == [1, 0, 0]
    assert pomiar.akcje_licznik[DZIEL] == 1
    assert tuple(pomiar.zdarzenia[0]) == KOLUMNY_DYLEMATY
    assert pomiar.zdarzenia[0] == {
        "pokolenie": 2,
        "krok": 7,
        "id": 10,
        "id_celu": 20,
        "zasoby_agenta": 2.0,
        "zasoby_celu": 3.0,
        "akcja": DZIEL,
        "klasa": "dzielenie",
        "wynik": SUKCES,
        "kwota": 1.0,
    }


@pytest.mark.parametrize(
    ("akcja", "nazwa_licznika", "klasa"),
    [
        (KRADNIJ, "K", "kradziez"),
        (NIC, "N_o", "obojetna"),
    ],
)
def test_zarejestruj_klasyfikuje_kradziez_i_akcje_obojetna(
    pomiar, stan, akcja, nazwa_licznika, klasa
):
    """Kradzież zwiększa K, a każda inna niż dzielenie i kradzież zwiększa N_o."""
    zasoby_start = np.array([2.0, 3.0, 0.0])

    zarejestruj(
        pomiar,
        stan,
        zasoby_start,
        i=0,
        j=1,
        akcja=akcja,
        dylemat_i=True,
        wynik=SUKCES,
        kwota=0.0,
    )

    assert getattr(pomiar, nazwa_licznika)[0] == 1
    assert pomiar.D[0] == 1
    assert pomiar.zdarzenia[0]["klasa"] == klasa


def test_zarejestruj_poza_dylematem_zlicza_tylko_akcje(pomiar, stan):
    """Każda intencja trafia do licznika akcji, ale nie każda jest dylematem."""
    zarejestruj(
        pomiar,
        stan,
        np.array([2.0, 3.0, 0.0]),
        i=0,
        j=PUSTE,
        akcja=NIC,
        dylemat_i=False,
        wynik=SUKCES,
        kwota=0.0,
    )

    assert pomiar.akcje_licznik[NIC] == 1
    assert pomiar.D[0] == 0
    assert pomiar.zdarzenia == []


def test_zarejestruj_zachowuje_niezmiennik_licznikow(pomiar, stan):
    """Każdy dylemat zwiększa dokładnie jeden z liczników S, K albo N_o."""
    zasoby_start = np.array([2.0, 3.0, 0.0])
    for akcja in (DZIEL, KRADNIJ, NIC):
        zarejestruj(
            pomiar,
            stan,
            zasoby_start,
            i=0,
            j=1,
            akcja=akcja,
            dylemat_i=True,
            wynik=SUKCES,
            kwota=0.0,
        )

    np.testing.assert_array_equal(pomiar.S + pomiar.K + pomiar.N_o, pomiar.D)


def test_zarejestruj_smierc_zapisuje_krok_i_przyczyne(pomiar):
    """Śmierć przez wyczerpanie zasobów otrzymuje indeks bieżącego kroku."""
    maska = np.array([False, True, False])

    zarejestruj_smierc(pomiar, maska, krok=4, przyczyna=PRZYCZYNA_WYCZERPANIE)

    assert pomiar.krok_smierci.tolist() == [-1, 4, -1]
    assert pomiar.przyczyna_smierci.tolist() == [
        PRZYCZYNA_ZYJE,
        PRZYCZYNA_WYCZERPANIE,
        PRZYCZYNA_ZYJE,
    ]


def test_zarejestruj_smierc_nie_nadpisuje_wczesniejszego_wpisu(pomiar):
    """Oznaczenie końca pokolenia nie zastępuje wcześniej zapisanej śmierci."""
    zarejestruj_smierc(
        pomiar,
        np.array([False, True, False]),
        krok=4,
        przyczyna=PRZYCZYNA_WYCZERPANIE,
    )

    zarejestruj_smierc(
        pomiar,
        np.array([True, True, True]),
        krok=10,
        przyczyna=PRZYCZYNA_KONIEC_POKOLENIA,
    )

    assert pomiar.krok_smierci.tolist() == [10, 4, 10]
    assert pomiar.przyczyna_smierci.tolist() == [
        PRZYCZYNA_KONIEC_POKOLENIA,
        PRZYCZYNA_WYCZERPANIE,
        PRZYCZYNA_KONIEC_POKOLENIA,
    ]
