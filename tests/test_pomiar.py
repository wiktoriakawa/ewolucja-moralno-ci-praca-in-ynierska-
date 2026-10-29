"""Testy kontenera Pomiar i funkcji pusty_pomiar."""

import numpy as np
import pytest

from moralnosc.pomiar import Pomiar, pusty_pomiar
from moralnosc.stan import LICZBA_AKCJI


N = 3


@pytest.fixture
def pomiar():
    """Świeży pusty pomiar dla trzech agentów."""
    return pusty_pomiar(N=N)


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
    assert np.all(pomiar.przyczyna_smierci == 0)
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
