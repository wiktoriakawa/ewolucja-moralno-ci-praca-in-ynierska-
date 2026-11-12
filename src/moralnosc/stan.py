"""Stan symulacji i wspólne stałe projektu (bez logiki symulacji)."""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# --- kierunki ---

POLNOC = 0
WSCHOD = 1
POLUDNIE = 2
ZACHOD = 3

KIERUNKI = np.array(
    [
        [-1, 0],  # POLNOC
        [0, 1],   # WSCHOD
        [1, 0],   # POLUDNIE
        [0, -1],  # ZACHOD
    ],
    dtype=np.int64,
)
KIERUNKI.flags.writeable = False


# --- akcje ---
RUCH_N = POLNOC
RUCH_E = WSCHOD
RUCH_S = POLUDNIE
RUCH_W = ZACHOD
ZBIERZ = 4   # cały zasób z pola pod agentem
DZIEL = 5    # kwota s dla agenta przed sobą (prospołeczna)
KRADNIJ = 6  # próba kradzieży od agenta przed sobą (egoistyczna)
NIC = 7      # w dylemacie obojętna
LICZBA_AKCJI = 8

# wyniki akcji
SUKCES = "sukces"
PORAZKA = "porazka"
NIEWYKONANA = "niewykonana"
WYKONANA = "wykonana"


# --- sieć ---
N_WE = 5             # wejścia (kolumny obserwacji, bez biasu)
N_WY = LICZBA_AKCJI  # jedno wyjście na akcję

# kolumny macierzy obserwacji N×5
OBS_ZASOBY_WLASNE = 0
OBS_ZASOBY_POD = 1        # pole, na którym stoi agent
OBS_ZASOBY_PRZED = 2      # pole przed agentem (0 przy ścianie)
OBS_OBECNOSC_SASIADA = 3  # 1/0
OBS_ZASOBY_SASIADA = 4    # 0, gdy nikogo nie ma

# „brak”: puste pole, brak sąsiada, brak rodzica, nienadany id
PUSTE = -1


@dataclass(eq=False, slots=True, kw_only=True)
class Stan:
    """Stan symulacji; indeks i to ten sam agent we wszystkich tablicach.

    eq=False – == na tablicach NumPy kończy się błędem; slots=True – literówka
    w nazwie pola zgłasza błąd; kw_only=True – pola tylko z nazwą.
    """

    # --- plansza (H×W) ---
    pojemnosc: NDArray[np.float64]       # stała w przebiegu
    zasoby_pol: NDArray[np.float64]      # 0 <= zasoby_pol <= pojemnosc
    zajetosc: NDArray[np.int64]          # indeks agenta albo PUSTE

    # --- agenci (N) ---
    pozycje: NDArray[np.int64]           # N×2, (wiersz, kolumna)
    kierunki: NDArray[np.int64]          # 0–3
    zasoby_agentow: NDArray[np.float64]
    zywy: NDArray[np.bool_]
    ids: NDArray[np.int64]               # unikalne w całym przebiegu
    rodzic: NDArray[np.int64]            # PUSTE w pokoleniu zerowym
    genomy: NDArray[np.float64] = field(repr=False)  # N×L

    # --- liczniki ---
    nastepny_id: int = 0
    pokolenie: int = 0
    krok: int = 0                        # w bieżącym pokoleniu, od 0


def pusty_stan(*, H: int, W: int, N: int, L: int) -> Stan:
    """Pusty stan o właściwych kształtach: nikt nie żyje ani nie stoi na planszy.

    Dane wpisują utworz_plansze, losuj_genomy i rozmiesc_losowo. Nie losuje.
    """
    return Stan(
        pojemnosc=np.zeros((H, W), dtype=np.float64),
        zasoby_pol=np.zeros((H, W), dtype=np.float64),
        zajetosc=np.full((H, W), PUSTE, dtype=np.int64),
        pozycje=np.zeros((N, 2), dtype=np.int64),
        kierunki=np.zeros(N, dtype=np.int64),
        zasoby_agentow=np.zeros(N, dtype=np.float64),
        zywy=np.zeros(N, dtype=np.bool_),
        ids=np.full(N, PUSTE, dtype=np.int64),
        rodzic=np.full(N, PUSTE, dtype=np.int64),
        genomy=np.zeros((N, L), dtype=np.float64),
    )
