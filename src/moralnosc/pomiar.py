"""Dane pomiarowe zbierane podczas jednego pokolenia symulacji.

Moduł oddziela dane służące analizie od stanu symulowanego świata. Klasa
``Pomiar`` przechowuje liczniki agentów, informacje o ich śmierci oraz
zdarzenia, które zostaną później zapisane do pliku ``dylematy.csv``.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import DZIEL, KRADNIJ, LICZBA_AKCJI, PUSTE, Stan


PRZYCZYNA_ZYJE = 0
PRZYCZYNA_WYCZERPANIE = 1
PRZYCZYNA_KONIEC_POKOLENIA = 2

KOLUMNY_DYLEMATY = (
    "pokolenie",
    "krok",
    "id",
    "id_celu",
    "zasoby_agenta",
    "zasoby_celu",
    "akcja",
    "klasa",
    "wynik",
    "kwota",
)


@dataclass(eq=False, slots=True, kw_only=True)
class Pomiar:
    """Dane zebrane dla wszystkich agentów w jednym pokoleniu.

    Indeks ``i`` oznacza tego samego agenta co indeks ``i`` w tablicach
    obiektu ``Stan``. Kontener nie zmienia przebiegu symulacji i nie wykonuje
    losowań. Jest uzupełniany przez funkcje rejestrujące podczas kolejnych
    kroków, a po zakończeniu pokolenia służy do przygotowania plików CSV.

    ``eq=False`` zapobiega niejednoznacznemu porównywaniu tablic NumPy przez
    operator ``==``. ``slots=True`` pozwala szybko wykryć literówki w nazwach
    pól, a ``kw_only=True`` wymaga podawania pól konstruktora z nazwami.
    """

    # --- decyzje agentów w sytuacjach dylematu ---
    S: NDArray[np.int64]  # N, liczba intencji dzielenia
    K: NDArray[np.int64]  # N, liczba intencji kradzieży
    # N, liczba intencji akcji obojętnych 0–4 i 7 w dylematach.
    N_o: NDArray[np.int64]
    D: NDArray[np.int64]  # N, liczba wszystkich napotkanych dylematów

    # --- zakończenie życia agentów w pokoleniu ---
    # Kroki są indeksowane od 0. Śmierć w K krokach ma numer 0…K−1, agent
    # dożywający końca pokolenia otrzymuje K, a brak wpisu ma wartość -1.
    krok_smierci: NDArray[np.int64]
    # PRZYCZYNA_ZYJE, PRZYCZYNA_WYCZERPANIE albo PRZYCZYNA_KONIEC_POKOLENIA.
    przyczyna_smierci: NDArray[np.int64]

    # --- dane zbiorcze i szczegółowe ---
    # Osiem elementów: jeden licznik dla każdego kodu akcji od 0 do 7.
    akcje_licznik: NDArray[np.int64]
    # Każdy słownik opisuje jedną decyzję podjętą w sytuacji dylematu.
    # Pole nie pojawia się w repr, ponieważ lista może zawierać wiele zdarzeń.
    zdarzenia: list[dict[str, object]] = field(repr=False)


def pusty_pomiar(*, N: int) -> Pomiar:
    """Tworzy pusty pomiar dla ``N`` agentów na początku pokolenia.

    Liczniki decyzji, dylematów i akcji zaczynają się od zera. Krok śmierci
    ma wartość ``-1``, ponieważ żaden agent jeszcze nie umarł, natomiast kod
    przyczyny śmierci wynosi ``0``, czyli agent nadal żyje. Lista zdarzeń jest
    tworzona od nowa przy każdym wywołaniu, więc pomiary różnych pokoleń nie
    współdzielą wpisów.

    Funkcja niczego nie losuje i nie modyfikuje obiektu ``Stan``.
    """
    return Pomiar(
        S=np.zeros(N, dtype=np.int64),
        K=np.zeros(N, dtype=np.int64),
        N_o=np.zeros(N, dtype=np.int64),
        D=np.zeros(N, dtype=np.int64),
        krok_smierci=np.full(N, -1, dtype=np.int64),
        przyczyna_smierci=np.full(N, PRZYCZYNA_ZYJE, dtype=np.int64),
        akcje_licznik=np.zeros(LICZBA_AKCJI, dtype=np.int64),
        zdarzenia=[],
    )


def status_dylematu(
    stan: Stan,
    przed: NDArray[np.int64],
    zasoby_start: NDArray[np.float64],
    s: float,
) -> NDArray[np.bool_]:
    """Wyznacza dylematy na podstawie migawki z początku kroku.

    Żywy agent znajduje się w dylemacie, gdy miał przed sobą innego agenta,
    posiadał co najmniej ``s`` zasobów, a jego cel miał dodatnie zasoby.
    Indeksy celów pochodzą z ``przed``, natomiast zasoby obu stron z niezmiennej
    kopii ``zasoby_start``. Funkcja nie modyfikuje stanu ani nie losuje.
    """
    dylemat = np.zeros(stan.zywy.shape, dtype=np.bool_)
    kandydaci = stan.zywy & (przed != PUSTE) & (zasoby_start >= s)
    indeksy = np.flatnonzero(kandydaci)
    dylemat[indeksy] = zasoby_start[przed[indeksy]] > 0
    return dylemat


def zarejestruj(
    pomiar: Pomiar,
    stan: Stan,
    zasoby_start: NDArray[np.float64],
    i: int,
    j: int,
    akcja: int,
    dylemat_i: bool | np.bool_,
    wynik: str,
    kwota: float,
) -> None:
    """Rejestruje intencję agenta oraz szczegóły decyzji w dylemacie.

    Każda wybrana akcja zwiększa licznik zbiorczy. Jeżeli decyzja została
    podjęta w dylemacie, funkcja zwiększa ``D`` i dokładnie jeden z liczników
    ``S``, ``K`` lub ``N_o`` oraz zapisuje zdarzenie przeznaczone do
    ``dylematy.csv``. Zasoby w zdarzeniu pochodzą z początku kroku.
    """
    pomiar.akcje_licznik[akcja] += 1
    if not dylemat_i:
        return

    pomiar.D[i] += 1
    if akcja == DZIEL:
        pomiar.S[i] += 1
        klasa = "dzielenie"
    elif akcja == KRADNIJ:
        pomiar.K[i] += 1
        klasa = "kradziez"
    else:
        pomiar.N_o[i] += 1
        klasa = "obojetna"

    wartosci = (
        stan.pokolenie,
        stan.krok,
        int(stan.ids[i]),
        int(stan.ids[j]),
        float(zasoby_start[i]),
        float(zasoby_start[j]),
        int(akcja),
        klasa,
        wynik,
        float(kwota),
    )
    zdarzenie = dict(zip(KOLUMNY_DYLEMATY, wartosci, strict=True))
    pomiar.zdarzenia.append(zdarzenie)


def zarejestruj_smierc(
    pomiar: Pomiar,
    maska: NDArray[np.bool_],
    krok: int,
    przyczyna: int,
) -> None:
    """Zapisuje krok i przyczynę śmierci wskazanych agentów.

    Wpis powstaje tylko dla agentów bez wcześniejszej przyczyny śmierci, dzięki
    czemu późniejsze wywołanie nie nadpisze informacji już zarejestrowanej.
    Funkcja nie modyfikuje stanu symulacji ani nie losuje.
    """
    bez_wpisu = pomiar.przyczyna_smierci == PRZYCZYNA_ZYJE
    do_zapisu = maska & bez_wpisu
    pomiar.krok_smierci[do_zapisu] = krok
    pomiar.przyczyna_smierci[do_zapisu] = przyczyna
