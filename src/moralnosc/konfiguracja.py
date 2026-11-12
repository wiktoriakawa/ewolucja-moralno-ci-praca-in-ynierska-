"""Wczytywanie, uzupełnianie i walidacja konfiguracji przebiegu."""

import math
import tomllib
from os import PathLike


class BladKonfiguracji(ValueError):
    """Błąd wskazujący niepoprawny parametr konfiguracji modelu."""


DOZWOLONE_POLA = {
    "przebieg": {"ziarno", "pokolenia", "kroki"},
    "plansza": {"H", "W"},
    "populacja": {"N"},
    "zasoby": {"c_min", "c_max", "alfa"},
    "obserwacje": {"skala"},
    "koszty": {"m", "ruch", "s", "c_k", "p", "x"},
    "siec": {"tryb", "tau", "h"},
    "ewolucja": {"T", "E0", "r", "sigma", "w_max", "R"},
    "zapis": {"katalog"},
}

# Wartości przewidziane w kontrakcie. ``skala`` zależy od c_max, a ``tau``
# jest uzupełniane osobno, ponieważ jest wymagane tylko dla trybu softmax.
DOMYSLNE_WARTOSCI = {
    ("populacja", "N"): 200,
    ("koszty", "ruch"): 0.0,
    ("siec", "h"): 0,
    ("ewolucja", "r"): "auto",
}


def _blad(parametr: str, przyczyna: str) -> None:
    raise BladKonfiguracji(f"{parametr}: {przyczyna}")


def _sprawdz_strukture(cfg: dict) -> None:
    """Sprawdza sekcje, nazwy pól i obecność parametrów bez wartości domyślnej."""
    nieznane_sekcje = sorted(set(cfg) - set(DOZWOLONE_POLA))
    if nieznane_sekcje:
        _blad(nieznane_sekcje[0], "nieznana sekcja konfiguracji")

    for sekcja, pola in DOZWOLONE_POLA.items():
        if sekcja not in cfg:
            _blad(sekcja, "brak wymaganej sekcji")
        if type(cfg[sekcja]) is not dict:
            _blad(sekcja, "sekcja musi być tabelą TOML")

        nieznane_pola = sorted(set(cfg[sekcja]) - pola)
        if nieznane_pola:
            _blad(f"{sekcja}.{nieznane_pola[0]}", "nieznany parametr")

        domyslne = {pole for nazwa, pole in DOMYSLNE_WARTOSCI if nazwa == sekcja}
        opcjonalne = domyslne | ({"skala"} if sekcja == "obserwacje" else set())
        opcjonalne |= {"tau"} if sekcja == "siec" else set()
        brakujace = sorted(pola - opcjonalne - set(cfg[sekcja]))
        if brakujace:
            _blad(f"{sekcja}.{brakujace[0]}", "brak wymaganego parametru")


def _uzupelnij_domyslne(cfg: dict) -> None:
    for (sekcja, pole), wartosc in DOMYSLNE_WARTOSCI.items():
        cfg[sekcja].setdefault(pole, wartosc)
    cfg["obserwacje"].setdefault("skala", cfg["zasoby"]["c_max"])


def _int(cfg: dict, sekcja: str, pole: str) -> int:
    wartosc = cfg[sekcja][pole]
    if type(wartosc) is not int:
        _blad(f"{sekcja}.{pole}", "wartość musi być liczbą całkowitą")
    return wartosc


def _float(cfg: dict, sekcja: str, pole: str) -> float:
    wartosc = cfg[sekcja][pole]
    if type(wartosc) is not float:
        _blad(f"{sekcja}.{pole}", "wartość musi być liczbą zmiennoprzecinkową")
    if not math.isfinite(wartosc):
        _blad(f"{sekcja}.{pole}", "wartość musi być skończona")
    return wartosc


def _tekst(cfg: dict, sekcja: str, pole: str) -> str:
    wartosc = cfg[sekcja][pole]
    if type(wartosc) is not str:
        _blad(f"{sekcja}.{pole}", "wartość musi być tekstem")
    return wartosc


def waliduj_konfiguracje(cfg: dict) -> dict:
    """Waliduje konfigurację i uzupełnia jej udokumentowane wartości domyślne.

    Zwraca ten sam słownik po uzupełnieniu wartości domyślnych. Pierwszy
    napotkany błąd zgłasza jako ``BladKonfiguracji`` z nazwą parametru.
    """
    if type(cfg) is not dict:
        _blad("konfiguracja", "główny obiekt musi być tabelą TOML")

    _sprawdz_strukture(cfg)
    _uzupelnij_domyslne(cfg)

    ziarno = _int(cfg, "przebieg", "ziarno")
    pokolenia = _int(cfg, "przebieg", "pokolenia")
    kroki = _int(cfg, "przebieg", "kroki")
    if ziarno < 0:
        _blad("przebieg.ziarno", "wartość musi być >= 0")
    if pokolenia < 1:
        _blad("przebieg.pokolenia", "wartość musi być >= 1")
    if kroki < 1:
        _blad("przebieg.kroki", "wartość musi być >= 1")

    H = _int(cfg, "plansza", "H")
    W = _int(cfg, "plansza", "W")
    N = _int(cfg, "populacja", "N")
    if H < 2:
        _blad("plansza.H", "wartość musi być >= 2")
    if W < 2:
        _blad("plansza.W", "wartość musi być >= 2")
    if not 1 <= N <= H * W:
        _blad("populacja.N", "musi spełniać warunek 1 <= N <= plansza.H * plansza.W")

    c_min = _float(cfg, "zasoby", "c_min")
    c_max = _float(cfg, "zasoby", "c_max")
    alfa = _float(cfg, "zasoby", "alfa")
    if c_min < 0:
        _blad("zasoby.c_min", "wartość musi być >= 0")
    if c_max < c_min:
        _blad("zasoby.c_max", "wartość musi być >= zasoby.c_min")
    if alfa < 0:
        _blad("zasoby.alfa", "wartość musi być >= 0")

    skala = _float(cfg, "obserwacje", "skala")
    if skala <= 0:
        _blad("obserwacje.skala", "wartość musi być > 0")

    m = _float(cfg, "koszty", "m")
    ruch = _float(cfg, "koszty", "ruch")
    s = _float(cfg, "koszty", "s")
    c_k = _float(cfg, "koszty", "c_k")
    p = _float(cfg, "koszty", "p")
    x = _float(cfg, "koszty", "x")
    if m <= 0:
        _blad("koszty.m", "wartość musi być > 0")
    if ruch < 0:
        _blad("koszty.ruch", "wartość musi być >= 0")
    if s <= 0:
        _blad("koszty.s", "wartość musi być > 0")
    if c_k < 0:
        _blad("koszty.c_k", "wartość musi być >= 0")
    if not 0 < p <= 1:
        _blad("koszty.p", "musi spełniać warunek 0 < p <= 1")
    if x <= 0:
        _blad("koszty.x", "wartość musi być > 0")
    if p * x - c_k <= 0:
        _blad("koszty.p, koszty.x, koszty.c_k", "muszą spełniać p * x - c_k > 0")

    tryb = _tekst(cfg, "siec", "tryb")
    if tryb not in {"argmax", "softmax"}:
        _blad("siec.tryb", "dozwolone wartości to 'argmax' albo 'softmax'")
    if "tau" not in cfg["siec"]:
        if tryb == "softmax":
            _blad("siec.tau", "brak parametru wymaganego w trybie softmax")
        cfg["siec"]["tau"] = 1.0  # w argmax wartość nie wpływa na wynik
    tau = _float(cfg, "siec", "tau")
    h = _int(cfg, "siec", "h")
    if tau <= 0:
        _blad("siec.tau", "wartość musi być > 0")
    if h < 0:
        _blad("siec.h", "wartość musi być >= 0")

    T = _float(cfg, "ewolucja", "T")
    E0 = _float(cfg, "ewolucja", "E0")
    r = cfg["ewolucja"]["r"]
    sigma = _float(cfg, "ewolucja", "sigma")
    w_max = _float(cfg, "ewolucja", "w_max")
    R = _int(cfg, "ewolucja", "R")
    if T < 0:
        _blad("ewolucja.T", "wartość musi być >= 0")
    if E0 <= 0:
        _blad("ewolucja.E0", "wartość musi być > 0")
    if r != "auto":
        if type(r) is not float or not math.isfinite(r):
            _blad("ewolucja.r", "wartość musi być liczbą float albo tekstem 'auto'")
        if not 0 <= r <= 1:
            _blad("ewolucja.r", "musi spełniać warunek 0 <= r <= 1")
    if sigma < 0:
        _blad("ewolucja.sigma", "wartość musi być >= 0")
    if w_max <= 0:
        _blad("ewolucja.w_max", "wartość musi być > 0")
    if R < 1:
        _blad("ewolucja.R", "wartość musi być >= 1")

    _tekst(cfg, "zapis", "katalog")
    return cfg


def wczytaj_konfiguracje(sciezka: str | PathLike[str]) -> dict:
    """Wczytuje plik TOML, waliduje go i zwraca zagnieżdżony słownik."""
    try:
        with open(sciezka, "rb") as plik:
            cfg = tomllib.load(plik)
    except tomllib.TOMLDecodeError as blad:
        raise BladKonfiguracji(f"plik TOML: niepoprawna składnia ({blad})") from blad

    return waliduj_konfiguracje(cfg)
