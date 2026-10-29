"""Dane pomiarowe zbierane podczas jednego pokolenia symulacji.

Moduł oddziela dane służące analizie od stanu symulowanego świata. Klasa
``Pomiar`` przechowuje liczniki agentów, informacje o ich śmierci oraz
zdarzenia, które zostaną później zapisane do pliku ``dylematy.csv``.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from moralnosc.stan import LICZBA_AKCJI


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
    N_o: NDArray[np.int64]  # N, liczba intencji wyboru akcji obojętnej
    D: NDArray[np.int64]  # N, liczba wszystkich napotkanych dylematów

    # --- zakończenie życia agentów w pokoleniu ---
    # Wartość -1 oznacza, że krok śmierci nie został jeszcze zarejestrowany.
    krok_smierci: NDArray[np.int64]
    # 0 = żyje, 1 = wyczerpanie zasobów, 2 = koniec pokolenia.
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
        przyczyna_smierci=np.zeros(N, dtype=np.int64),
        akcje_licznik=np.zeros(LICZBA_AKCJI, dtype=np.int64),
        zdarzenia=[],
    )
