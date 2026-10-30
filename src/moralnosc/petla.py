"""Mechaniczna obsługa pojedynczego kroku symulacji."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from moralnosc.ewolucja import kwalifikacja, nowe_pokolenie, pokolenie_zerowe
from moralnosc.interakcje import nalicz_koszty, wykonaj_dzielenie, wykonaj_kradziez
from moralnosc.pomiar import (
    PRZYCZYNA_KONIEC_POKOLENIA,
    PRZYCZYNA_WYCZERPANIE,
    Pomiar,
    pusty_pomiar,
    status_dylematu,
    zarejestruj,
    zarejestruj_smierc,
)
from moralnosc.populacja import obserwacje, ruch, usun_martwych
from moralnosc.siec import wybierz_akcje, wyjscia
from moralnosc.srodowisko import odrost, zbierz
from moralnosc.stan import (
    DZIEL,
    KIERUNKI,
    KRADNIJ,
    NIC,
    NIEWYKONANA,
    PORAZKA,
    PUSTE,
    RUCH_N,
    RUCH_W,
    SUKCES,
    WYKONANA,
    ZBIERZ,
    Stan,
)


# Typ funkcji wywoływanej po zakończeniu każdego pokolenia. Konkretną
# funkcję, np. zapisującą wyniki do CSV, przekazujemy jako `po_pokoleniu`.
ObslugaPokolenia = Callable[[Stan, Pomiar, NDArray[np.bool_], bool], None]


def _cel_nadal_przed(stan: Stan, i: int, cel: int) -> bool:
    """Sprawdza, czy cel z migawki nadal stoi przed agentem i."""
    if cel == PUSTE:
        return False

    pole = stan.pozycje[i] + KIERUNKI[stan.kierunki[i]]
    h, w = stan.zajetosc.shape
    if not (0 <= pole[0] < h and 0 <= pole[1] < w):
        return False

    return bool(stan.zajetosc[pole[0], pole[1]] == cel)


def wykonaj_akcje(
    stan: Stan,
    i: int,
    cel: int,
    akcja: int,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[str, float]:
    """Wykonuje akcję agenta na aktualnym stanie i zwraca wynik oraz kwotę."""
    if RUCH_N <= akcja <= RUCH_W:
        stan.zasoby_agentow[i] -= cfg["koszty"]["ruch"]
        wykonano = ruch(stan, i, akcja)
        return (SUKCES if wykonano else PORAZKA), 0.0

    if akcja == ZBIERZ:
        return WYKONANA, zbierz(stan, i)

    if akcja == DZIEL:
        if not _cel_nadal_przed(stan, i, cel):
            return NIEWYKONANA, 0.0
        wykonano = wykonaj_dzielenie(stan, i, cel, cfg["koszty"]["s"])
        kwota = cfg["koszty"]["s"] if wykonano else 0.0
        return (SUKCES if wykonano else NIEWYKONANA), float(kwota)

    if akcja == KRADNIJ:
        if not _cel_nadal_przed(stan, i, cel):
            return NIEWYKONANA, 0.0
        sukces, kwota = wykonaj_kradziez(
            stan,
            i,
            cel,
            c_k=cfg["koszty"]["c_k"],
            p=cfg["koszty"]["p"],
            x=cfg["koszty"]["x"],
            rng=rng,
        )
        return (SUKCES if sukces else PORAZKA), kwota

    if akcja == NIC:
        return WYKONANA, 0.0

    raise ValueError(f"Nieznany kod akcji {akcja!r}")


def krok(
    stan: Stan,
    cfg: dict,
    rng: np.random.Generator,
    *,
    pomiar: Pomiar | None = None,
) -> bool:
    """Wykonuje jeden krok symulacji i zwraca, czy któryś agent przeżył.

    Gdy podano ``pomiar``, rejestruje intencje i wyniki akcji na podstawie
    migawki z początku kroku oraz zgony spowodowane wyczerpaniem zasobów.
    Brak pomiaru pozwala uruchamiać samą mechanikę, np. w próbie przeżycia.
    """
    obs, przed = obserwacje(stan, cfg["obserwacje"]["skala"])
    if pomiar is not None:
        zasoby_start = stan.zasoby_agentow.copy()
        dylemat = status_dylematu(stan, przed, zasoby_start, cfg["koszty"]["s"])

    y = wyjscia(obs, stan.genomy)
    akcje = wybierz_akcje(y, cfg["siec"]["tryb"], cfg["siec"]["tau"], rng)

    zywi = np.flatnonzero(stan.zywy)
    for i in rng.permutation(zywi):
        wynik, kwota = wykonaj_akcje(stan, i, przed[i], akcje[i], cfg, rng)
        if pomiar is not None:
            zarejestruj(
                pomiar,
                stan,
                zasoby_start,
                i=int(i),
                j=int(przed[i]),
                akcja=int(akcje[i]),
                dylemat_i=dylemat[i],
                wynik=wynik,
                kwota=kwota,
            )

    nalicz_koszty(stan, cfg["koszty"]["m"])
    martwi = stan.zywy & (stan.zasoby_agentow <= 0)
    if pomiar is not None:
        zarejestruj_smierc(
            pomiar,
            martwi,
            krok=stan.krok,
            przyczyna=PRZYCZYNA_WYCZERPANIE,
        )
    usun_martwych(stan, martwi)
    odrost(stan, cfg["zasoby"]["alfa"])
    stan.krok += 1
    return bool(stan.zywy.any())


def wykonaj_pokolenie(
    stan: Stan,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[Pomiar, NDArray[np.bool_], bool, bool]:
    """Wykonuje jedno pokolenie i przygotowuje dane do zapisu.

    Zwraca pomiar, maskę zakwalifikowanych rodziców, informację o użyciu
    wariantu awaryjnego oraz informację, czy ktoś dożył końca pokolenia.
    Po wymarciu maska rodziców jest pusta, a wariant awaryjny ma wartość
    ``False``.
    """
    stan.krok = 0
    pomiar = pusty_pomiar(N=stan.zywy.size)

    for _ in range(cfg["przebieg"]["kroki"]):
        if not krok(stan, cfg, rng, pomiar=pomiar):
            pusta_maska = np.zeros(stan.zywy.shape, dtype=np.bool_)
            return pomiar, pusta_maska, False, False

    maska, awaryjny = kwalifikacja(stan, cfg["ewolucja"]["T"])
    zarejestruj_smierc(
        pomiar,
        stan.zywy,
        krok=stan.krok,
        przyczyna=PRZYCZYNA_KONIEC_POKOLENIA,
    )
    return pomiar, maska, awaryjny, True


def uruchom_przebieg(
    cfg: dict,
    *,
    po_pokoleniu: ObslugaPokolenia | None = None,
) -> Stan:
    """Uruchamia kolejne pokolenia jednego deterministycznego przebiegu.

    Opcjonalna funkcja ``po_pokoleniu`` otrzymuje stan przed utworzeniem
    potomków, pomiar, maskę rodziców i informację o wariancie awaryjnym.
    Docelowo w tym miejscu zostanie podłączony zapis wyników do CSV.
    Przebieg kończy się po zadanej liczbie pokoleń albo po wymarciu.
    """
    rng = np.random.default_rng(cfg["przebieg"]["ziarno"])
    stan = pokolenie_zerowe(cfg, rng)
    liczba_pokolen = cfg["przebieg"]["pokolenia"]

    for numer_pokolenia in range(liczba_pokolen):
        pomiar, maska, awaryjny, przetrwal = wykonaj_pokolenie(stan, cfg, rng)

        if po_pokoleniu is not None:
            po_pokoleniu(stan, pomiar, maska, awaryjny)

        if not przetrwal or numer_pokolenia == liczba_pokolen - 1:
            break

        nowe_pokolenie(stan, maska, cfg, rng)

    return stan
