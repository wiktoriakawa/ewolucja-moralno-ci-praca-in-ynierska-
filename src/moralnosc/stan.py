"""Stan symulacji i wspólne stałe projektu.

Moduł nie zawiera logiki symulacji – tylko stałe (kierunki, kody akcji,
rozmiar sieci), opis stanu symulacji (klasa Stan) i funkcję tworzącą pusty stan.
Zgodny z kontraktem interfejsów, sekcje 2–3.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

#przemieszczanie sie

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
KIERUNKI.flags.writeable = False  # tylko do odczytu – zapis zgłosi błąd


RUCH_N = POLNOC
RUCH_E = WSCHOD
RUCH_S = POLUDNIE
RUCH_W = ZACHOD
ZBIERZ = 4   # zbieranie całego zasobu z pola pod agentem
DZIEL = 5    # przekazanie kwoty s agentowi przed sobą – akcja prospołeczna
KRADNIJ = 6  # próba kradzieży od agenta przed sobą – akcja egoistyczna
NIC = 7      # pozostanie w miejscu – świadoma akcja, w dylemacie obojętna
LICZBA_AKCJI = 8


# rozmiar sieci neuronowej: N_WE wejść, N_WY wyjść (bez biasu), 1 warstwa ukryta
N_WE = 5             # liczba wejść (kolumny macierzy obserwacji, bez biasu)
N_WY = LICZBA_AKCJI  # jedno wyjście na każdą akcję – obie wartości nie mogą się rozjechać

# Numery kolumn macierzy obserwacji (N×5), zgodnie z kontraktem, sekcja 4.4.
OBS_ZASOBY_WLASNE = 0     # zasoby agenta
OBS_ZASOBY_POD = 1        # zasoby pola, na którym stoi agent
OBS_ZASOBY_PRZED = 2      # zasoby pola przed agentem (0 przy ścianie)
OBS_OBECNOSC_SASIADA = 3  # 1, gdy przed agentem stoi inny agent, inaczej 0
OBS_ZASOBY_SASIADA = 4    # zasoby agenta przed agentem (0, gdy go nie ma)

# Wartość oznaczająca „brak”: puste pole w mapie zajętości, brak agenta przed
# sobą, brak rodzica (pokolenie zerowe) i jeszcze nienadany identyfikator.
PUSTE = -1


#klasa stanu symulacji


@dataclass(eq=False, slots=True, kw_only=True)
class Stan:
    """Stan symulacji w jednej chwili.

    Indeks i oznacza tego samego agenta we wszystkich tablicach agentów.
    Oznaczenia kształtów: H×W – plansza, N – liczba agentów, L – długość genomu.

    eq=False     – bez porównywania przez ==, które dla tablic NumPy kończy się błędem;
    slots=True   – literówka w nazwie pola (np. stan.zasoby_pl = ...) zgłasza błąd;
    kw_only=True – pola podaje się tylko z nazwą, więc nie da się pomylić kolejności.
    """

    # --- plansza ---
    pojemnosc: NDArray[np.float64]       # H×W, pojemność pól c_max(x, y); stała w przebiegu
    zasoby_pol: NDArray[np.float64]      # H×W, bieżące zasoby pól; 0 <= zasoby_pol <= pojemnosc
    zajetosc: NDArray[np.int64]          # H×W, indeks agenta stojącego na polu albo PUSTE

    # --- agenci ---
    pozycje: NDArray[np.int64]           # N×2, (wiersz, kolumna)
    kierunki: NDArray[np.int64]          # N, kierunek patrzenia: POLNOC…ZACHOD (0–3)
    zasoby_agentow: NDArray[np.float64]  # N, bieżące zasoby agentów
    zywy: NDArray[np.bool_]              # N, czy agent żyje
    ids: NDArray[np.int64]               # N, identyfikator unikalny w całym przebiegu
    rodzic: NDArray[np.int64]            # N, identyfikator rodzica; PUSTE w pokoleniu zerowym
    genomy: NDArray[np.float64] = field(repr=False)  # N×L, wagi sieci (ukryte przy print)

    # --- liczniki ---
    nastepny_id: int = 0                 # kolejny wolny identyfikator
    pokolenie: int = 0                   # numer pokolenia (0 = pokolenie zerowe)
    krok: int = 0                        # numer kroku w bieżącym pokoleniu


def pusty_stan(*, H: int, W: int, N: int, L: int) -> Stan:
    """Tworzy pusty stan: plansza H×W, N agentów, genomy długości L.

    Argumenty podaje się tylko z nazwą (gwiazdka w sygnaturze), np.
    pusty_stan(H=64, W=64, N=200, L=48), żeby nie pomylić H, W i N.

    Tablice mają właściwe kształty i typy, ale nie zawierają jeszcze danych:
    wypełniają je utworz_plansze, losuj_genomy i rozmiesc_losowo (kontrakt,
    sekcja 5). Nikt nie jest jeszcze żywy ani rozmieszczony, więc niezmienniki
    stanu są spełnione od początku. Funkcja niczego nie losuje.
    """
    return Stan(
        pojemnosc=np.zeros((H, W), dtype=np.float64),
        zasoby_pol=np.zeros((H, W), dtype=np.float64),
        zajetosc=np.full((H, W), PUSTE, dtype=np.int64),  # wszystkie pola puste
        pozycje=np.zeros((N, 2), dtype=np.int64),
        kierunki=np.zeros(N, dtype=np.int64),
        zasoby_agentow=np.zeros(N, dtype=np.float64),
        zywy=np.zeros(N, dtype=np.bool_),  # False: agenci jeszcze nie stoją na planszy
        ids=np.full(N, PUSTE, dtype=np.int64),     # identyfikatory nadaje inicjalizacja
        rodzic=np.full(N, PUSTE, dtype=np.int64),  # pokolenie zerowe nie ma rodziców
        genomy=np.zeros((N, L), dtype=np.float64),
    )