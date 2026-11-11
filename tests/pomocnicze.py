"""Funkcje pomocnicze do testów."""
import numpy as np

from moralnosc.stan import PUSTE, Stan


def sprawdz_niezmienniki(stan: Stan) -> None:
    """Niezmienniki stanu (kontrakt, 3.2); naruszenie → AssertionError z opisem."""
    H, W = stan.zajetosc.shape
    zywi = np.flatnonzero(stan.zywy)
    wiersze = stan.pozycje[zywi, 0]
    kolumny = stan.pozycje[zywi, 1]

    # 0. żywi na planszy (indeks −1 „zawinąłby” bez błędu)
    assert np.all((wiersze >= 0) & (wiersze < H) & (kolumny >= 0) & (kolumny < W)), \
        "żywy agent poza planszą"

    # 1. pole żywego agenta wskazuje na niego
    assert np.array_equal(stan.zajetosc[wiersze, kolumny], zywi), \
        "mapa zajętości nie zgadza się z pozycjami żywych agentów"

    # 2. zajętych pól tyle, ilu żywych
    assert np.count_nonzero(stan.zajetosc != PUSTE) == zywi.size, \
        "liczba zajętych pól różni się od liczby żywych agentów"

    # 3. 0 ≤ zasoby pól ≤ pojemność
    assert np.all(stan.zasoby_pol >= 0), "ujemne zasoby pola"
    assert np.all(stan.zasoby_pol <= stan.pojemnosc), "zasoby pola powyżej pojemności"
