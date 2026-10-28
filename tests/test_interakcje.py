"""Testy minimalne interakcji i metabolizmu na Lab02."""

import numpy as np
import pytest

from moralnosc import interakcje
from moralnosc.stan import pusty_stan


def utworz_stan(zasoby, zywi=None):
    """Tworzy minimalny stan agentow potrzebny do testow interakcji."""
    n = len(zasoby)
    stan = pusty_stan(H=1, W=n, N=n, L=48)
    stan.zasoby_agentow[:] = zasoby
    stan.zywy[:] = True if zywi is None else zywi
    return stan


def test_dzielenie_przenosi_pelna_kwote():
    """Agent może przekazać s także wtedy, gdy ma dokładnie s zasobów."""
    stan = utworz_stan([3.0, 1.0])

    wykonano = interakcje.wykonaj_dzielenie(stan, i=0, j=1, s=3.0)

    assert wykonano
    np.testing.assert_allclose(stan.zasoby_agentow, [0.0, 4.0])


def test_dzielenie_nie_przenosi_czesciowej_kwoty():
    """Brak pełnej kwoty s oznacza niewykonanie dzielenia bez częściowego daru."""
    stan = utworz_stan([2.0, 1.0])

    wykonano = interakcje.wykonaj_dzielenie(stan, i=0, j=1, s=3.0)

    assert not wykonano
    np.testing.assert_allclose(stan.zasoby_agentow, [2.0, 1.0])


def test_udana_kradziez_nalicza_koszt_i_przenosi_kwote():
    """Udana próba najpierw kosztuje c_k, a następnie przenosi x zasobów."""
    stan = utworz_stan([5.0, 4.0])
    rng = np.random.default_rng(0)

    sukces, kwota = interakcje.wykonaj_kradziez(
        stan, i=0, j=1, c_k=1.0, p=1.0, x=3.0, rng=rng
    )

    assert sukces
    assert kwota == pytest.approx(3.0)
    np.testing.assert_allclose(stan.zasoby_agentow, [7.0, 1.0])


def test_udana_kradziez_zabiera_tylko_dostepne_zasoby():
    """Kradzież nie może zabrać więcej zasobów, niż aktualnie ma ofiara."""
    stan = utworz_stan([5.0, 2.0])
    rng = np.random.default_rng(0)

    sukces, kwota = interakcje.wykonaj_kradziez(
        stan, i=0, j=1, c_k=1.0, p=1.0, x=3.0, rng=rng
    )

    assert sukces
    assert kwota == pytest.approx(2.0)
    np.testing.assert_allclose(stan.zasoby_agentow, [6.0, 0.0])


def test_udana_kradziez_nie_przenosi_ujemnych_zasobow():
    """Ujemne zasoby ofiary są traktowane jako brak zasobów do kradzieży."""
    stan = utworz_stan([5.0, -1.0])
    rng = np.random.default_rng(0)

    sukces, kwota = interakcje.wykonaj_kradziez(
        stan, i=0, j=1, c_k=1.0, p=1.0, x=3.0, rng=rng
    )

    assert sukces
    assert kwota == pytest.approx(0.0)
    np.testing.assert_allclose(stan.zasoby_agentow, [4.0, -1.0])


def test_nieudana_kradziez_nalicza_koszt_bez_transferu():
    """Losowa porażka nadal kosztuje c_k, ale nie powoduje transferu."""
    stan = utworz_stan([5.0, 4.0])
    rng = np.random.default_rng(0)

    sukces, kwota = interakcje.wykonaj_kradziez(
        stan, i=0, j=1, c_k=1.0, p=0.5, x=3.0, rng=rng
    )

    assert not sukces
    assert kwota == pytest.approx(0.0)
    np.testing.assert_allclose(stan.zasoby_agentow, [4.0, 4.0])


def test_metabolizm_dotyczy_tylko_zywych_agentow():
    """Metabolizm obciąża tylko żywych i nie zmienia flag stanu życia."""
    stan = utworz_stan([3.0, 2.0, 1.0], zywi=[True, False, True])

    interakcje.nalicz_koszty(stan, m=1.5)

    np.testing.assert_allclose(stan.zasoby_agentow, [1.5, 2.0, -0.5])
    assert np.array_equal(stan.zywy, [True, False, True])
