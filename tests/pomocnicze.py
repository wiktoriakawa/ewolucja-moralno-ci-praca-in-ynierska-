"""Funkcje pomocnicze do testów (nie są częścią programu)."""
import numpy as np

from moralnosc.stan import PUSTE, Stan


def sprawdz_niezmienniki(stan: Stan) -> None:
    """Sprawdza niezmienniki stanu z kontraktu (sekcja 3.2).

    Przy naruszeniu zgłasza AssertionError z opisem, który niezmiennik nie działa.
    """
    H, W = stan.zajetosc.shape
    zywi = np.flatnonzero(stan.zywy)  # indeksy żywych agentów
    wiersze = stan.pozycje[zywi, 0]
    kolumny = stan.pozycje[zywi, 1]

    # 0. Żywi agenci stoją na planszy (indeks −1 nie zgłosiłby błędu, tylko „zawinął”).
    assert np.all((wiersze >= 0) & (wiersze < H) & (kolumny >= 0) & (kolumny < W)), \
        "żywy agent poza planszą"

    # 1. Pole każdego żywego agenta wskazuje w mapie zajętości właśnie na niego.
    assert np.array_equal(stan.zajetosc[wiersze, kolumny], zywi), \
        "mapa zajętości nie zgadza się z pozycjami żywych agentów"

    # 2. Zajętych pól jest dokładnie tyle, ilu jest żywych (brak martwych na mapie i kolizji).
    assert np.count_nonzero(stan.zajetosc != PUSTE) == zywi.size, \
        "liczba zajętych pól różni się od liczby żywych agentów"

    # 3. Zasoby pól mieszczą się między zerem a pojemnością.
    assert np.all(stan.zasoby_pol >= 0), "ujemne zasoby pola"
    assert np.all(stan.zasoby_pol <= stan.pojemnosc), "zasoby pola powyżej pojemności"