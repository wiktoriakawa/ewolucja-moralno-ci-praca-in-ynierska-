"""Ewolucja: pokolenie zerowe, kwalifikacja do reprodukcji, losowanie rodziców,
mutacja, narodziny potomków w promieniu R i podmiana pokolenia
"""
import numpy as np
from numpy.typing import NDArray

from moralnosc.populacja import rozmiesc_losowo
from moralnosc.siec import dlugosc_genomu, losuj_genomy
from moralnosc.srodowisko import utworz_plansze
from moralnosc.stan import N_WE, N_WY, PUSTE, Stan, pusty_stan


def pokolenie_zerowe(cfg: dict, rng: np.random.Generator) -> Stan:
    """Tworzy stan początkowy przebiegu: planszę i pokolenie zerowe.
    Argumenty:
        cfg: konfiguracja; potrzebne klucze plansza.H, plansza.W, populacja.N,
             zasoby.c_min, zasoby.c_max, ewolucja.E0.
        rng: generator przebiegu.

    Zwraca: stan z pełną planszą i N agentami – losowe genomy, pozycje
    i kierunki, zasoby E0, wszyscy żywi, ids = 0…N−1, rodzic = PUSTE,
    nastepny_id = N, pokolenie = 0.
    Losuje w kolejności z kontraktu: plansza → genomy → rozmieszczenie.
    """
    H = cfg["plansza"]["H"]
    W = cfg["plansza"]["W"]
    N = cfg["populacja"]["N"]
    L = dlugosc_genomu(N_WE, N_WY)

    stan = pusty_stan(H=H, W=W, N=N, L=L)
    stan.pojemnosc, stan.zasoby_pol = utworz_plansze(cfg, rng)
    stan.genomy[:] = losuj_genomy(N, L, rng)
    rozmiesc_losowo(stan, rng)

    stan.zasoby_agentow[:] = cfg["ewolucja"]["E0"]
    stan.zywy[:] = True
    stan.ids[:] = np.arange(N)
    stan.nastepny_id = N

    return stan


def kwalifikacja(stan: Stan, T: float) -> tuple[NDArray[np.bool_], bool]:
    """Wybiera agentów, którzy mogą zostać rodzicami.

    Argumenty:
        stan: stan na koniec pokolenia.
        T:    próg zasobów (klucz ewolucja.T).

    Zwraca:
        maska:    N wartości logicznych – True dla żywych z zasobami ≥ T;
                  gdy nikt nie osiąga progu – True dla wszystkich żywych;
        awaryjny: True, gdy zadziałał wariant awaryjny.

    Nie zmienia stanu. Nie losuje. Gdy nikt nie żyje, maska jest cała False –
    wtedy pętla kończy przebieg i nie woła nowe_pokolenie.
    Uwaga: kolumna kwalifikacja w agenci.csv to zawsze żywy ∧ zasoby ≥ T,
    także w pokoleniu awaryjnym – liczy ją zapis, nie ta maska.
    """
    maska = stan.zywy & (stan.zasoby_agentow >= T)  # & działa element po elemencie, `and` – nie
    if maska.any():
        return maska, False

    return stan.zywy.copy(), True


def losuj_rodzicow(
    maska: NDArray[np.bool_], n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Losuje rodziców n potomków spośród agentów z maski.

    Argumenty:
        maska: N wartości logicznych z kwalifikacja (co najmniej jedna True).
        n:     liczba potomków.
        rng:   generator przebiegu (n losowań).

    Zwraca: tablica n indeksów rodziców (int64) – losowanie ze zwracaniem,
    z równym prawdopodobieństwem, więc jeden agent może wystąpić wiele razy.
    Przy masce całej False rng.integers zgłasza ValueError.
    """
    kandydaci = np.flatnonzero(maska)
    return kandydaci[rng.integers(0, kandydaci.size, size=n)]


def mutuj(
    genomy_rodzicow: NDArray[np.float64],
    r: float,
    sigma: float,
    w_max: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Tworzy genomy potomków przez mutację genomów rodziców.

    Każda waga z prawdopodobieństwem r dostaje szum z rozkładu N(0, sigma);
    potem wszystkie wagi są przycinane do przedziału [−w_max, w_max].

    Argumenty:
        genomy_rodzicow: n×L – wiersz k to genom rodzica potomka k.
        r:     prawdopodobieństwo mutacji pojedynczej wagi (0 ≤ r ≤ 1).
        sigma: odchylenie standardowe szumu.
        w_max: największa dopuszczalna wartość bezwzględna wagi.
        rng:   generator przebiegu.

    Zwraca: NOWĄ tablicę n×L; genomy_rodzicow się nie zmieniają.
    Losuje: najpierw maska mutacji (n×L), potem szum (n×L) – kolejność z kontraktu.
    """
    maska = rng.random(genomy_rodzicow.shape) < r  # True – ta waga mutuje
    # Szum losujemy dla wszystkich wag, także tych bez mutacji: liczba losowań
    # nie zależy wtedy od wyniku maski.
    szum = rng.normal(0.0, sigma, size=genomy_rodzicow.shape)
    potomkowie = genomy_rodzicow + maska * szum  # `+` tworzy nową tablicę; `+=` zmieniłoby rodziców
    # Przycinane są wszystkie wagi, także niezmutowane – ma to znaczenie tylko
    # przy małym w_max (wagi pokolenia zerowego pochodzą z N(0, 1)).
    np.clip(potomkowie, -w_max, w_max, out=potomkowie)
    return potomkowie


def rozmiesc_potomkow(
    pozycje_rodzicow: NDArray[np.int64],
    R: int,
    H: int,
    W: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Rozmieszcza potomków na wolnych polach w pobliżu rodziców.

    Potomek rodzi się na losowym wolnym polu kwadratu o promieniu R wokół
    pozycji rodzica (kwadrat przycięty do planszy). Gdy w kwadracie nie ma
    wolnego pola, promień rośnie o 1, aż wolne pole się znajdzie.

    Argumenty:
        pozycje_rodzicow: n×2 – wiersz i to pozycja rodzica potomka i.
        R:    promień narodzin (klucz ewolucja.R, R ≥ 1).
        H, W: rozmiary planszy.
        rng:  generator przebiegu.

    Zwraca:
        pozycje:  n×2 (int64) – pozycje potomków;
        zajetosc: H×W (int64) – nowa mapa zajętości: indeks potomka albo PUSTE.

    Losuje: permutacja kolejności potomków, potem jedno losowanie pola na potomka.
    Gdy n > H·W – ValueError (potomkowie się nie zmieszczą).
    """
    n = pozycje_rodzicow.shape[0]
    if n > H * W:
        raise ValueError(f"{n} potomków nie zmieści się na planszy {H}×{W}")

    pozycje = np.empty((n, 2), dtype=np.int64)
    zajetosc = np.full((H, W), PUSTE, dtype=np.int64)

    # Pętla jest tu potrzebna: każdy potomek zajmuje pole następnym.
    # Losowa kolejność – pierwszeństwo wyboru pola nie zależy od numeru potomka.
    for i in rng.permutation(n):
        w, k = pozycje_rodzicow[i]
        promien = R
        while True:
            # Granice okna przycięte do planszy; górna granica jest wyłączna, jak w wycinkach.
            w0, w1 = max(0, w - promien), min(H, w + promien + 1)
            k0, k1 = max(0, k - promien), min(W, k + promien + 1)
            wolne = np.argwhere(zajetosc[w0:w1, k0:k1] == PUSTE)  # współrzędne wewnątrz okna
            if wolne.shape[0] > 0:
                break
            # Pętla się kończy, bo n ≤ H·W: najpóźniej okno obejmie całą planszę,
            # a na niej zawsze jest wolne pole dla kolejnego potomka.
            promien += 1
        dw, dk = wolne[rng.integers(wolne.shape[0])]
        pozycje[i] = (w0 + dw, k0 + dk)  # z układu okna z powrotem do układu planszy
        zajetosc[w0 + dw, k0 + dk] = i   # i, nie numer w kolejności: zajetosc[pozycje[i]] == i
    return pozycje, zajetosc


def nowe_pokolenie(
    stan: Stan, maska: NDArray[np.bool_], cfg: dict, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Zastępuje całą populację N potomkami (pokolenia rozłączne, WF-08).

    Argumenty:
        stan:  stan na koniec pokolenia.
        maska: maska rodziców z kwalifikacja (co najmniej jedna True).
        cfg:   konfiguracja; potrzebne klucze ewolucja.r (liczba albo "auto" = 1/L),
               ewolucja.sigma, ewolucja.w_max, ewolucja.R, ewolucja.E0.
        rng:   generator przebiegu.

    Zwraca: tablica N indeksów rodziców (indeksy z kończącego się pokolenia) – do logów.

    Zmienia: wszystkie pola agentów, zajetosc, nastepny_id, pokolenie.
    Nie zmienia: pojemnosc i zasoby_pol (plansza przechodzi do kolejnego
    pokolenia – do decyzji, czy ją odnawiać) ani krok (zeruje go pętla).
    Losuje: rodzice → maska mutacji → szum → kolejność potomków → pola → kierunki.
    """
    N, L = stan.genomy.shape
    H, W = stan.zajetosc.shape
    ew = cfg["ewolucja"]
    r = 1.0 / L if ew["r"] == "auto" else ew["r"]

    # Najpierw wszystko liczymy ze starego stanu, ...
    rodzice = losuj_rodzicow(maska, N, rng)
    # indeksowanie tablicą indeksów (stan.genomy[rodzice]) daje kopię, nie widok
    genomy = mutuj(stan.genomy[rodzice], r, ew["sigma"], ew["w_max"], rng)
    pozycje, zajetosc = rozmiesc_potomkow(stan.pozycje[rodzice], ew["R"], H, W, rng)
    kierunki = rng.integers(0, 4, size=N)

    # ... a dopiero potem nadpisujemy stan. Zapis przez [:] trafia do istniejących
    # tablic, więc odwołania do nich trzymane gdzie indziej pozostają aktualne.
    stan.rodzic[:] = stan.ids[rodzice]  # STARE identyfikatory rodziców – przed nadaniem nowych
    stan.ids[:] = stan.nastepny_id + np.arange(N)
    stan.nastepny_id += N
    stan.genomy[:] = genomy
    stan.pozycje[:] = pozycje
    stan.zajetosc[:] = zajetosc
    stan.kierunki[:] = kierunki
    stan.zasoby_agentow[:] = ew["E0"]  # bez dziedziczenia bogactwa
    stan.zywy[:] = True
    stan.pokolenie += 1
    return rodzice
