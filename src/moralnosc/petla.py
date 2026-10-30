"""Mechaniczna obsługa pojedynczego kroku symulacji."""

import numpy as np

from moralnosc.interakcje import nalicz_koszty, wykonaj_dzielenie, wykonaj_kradziez
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


def krok(stan: Stan, cfg: dict, rng: np.random.Generator) -> bool:
    """Wykonuje jeden krok symulacji i zwraca, czy któryś agent przeżył."""
    obs, przed = obserwacje(stan, cfg["obserwacje"]["skala"])
    y = wyjscia(obs, stan.genomy)
    akcje = wybierz_akcje(y, cfg["siec"]["tryb"], cfg["siec"]["tau"], rng)

    zywi = np.flatnonzero(stan.zywy)
    for i in rng.permutation(zywi):
        wykonaj_akcje(stan, i, przed[i], akcje[i], cfg, rng)

    nalicz_koszty(stan, cfg["koszty"]["m"])
    martwi = stan.zywy & (stan.zasoby_agentow <= 0)
    usun_martwych(stan, martwi)
    odrost(stan, cfg["zasoby"]["alfa"])
    stan.krok += 1
    return bool(stan.zywy.any())
