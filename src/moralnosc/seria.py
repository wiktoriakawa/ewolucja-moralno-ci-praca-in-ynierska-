"""Definicja, rozwijanie i wykonywanie serii przebiegów."""

import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

from moralnosc.konfiguracja import waliduj_konfiguracje, wczytaj_konfiguracje
from moralnosc.petla import uruchom_przebieg
from moralnosc.zapis import przygotuj_zapis


KOLUMNY_STATUSU = ("id", "wariant", "ziarno", "status", "katalog", "blad")
DOZWOLONA_NAZWA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
POLA_AUTOMATYCZNE = {"przebieg.ziarno", "zapis.katalog"}


class BladSerii(ValueError):
    """Błąd struktury lub wartości w definicji serii."""


@dataclass(frozen=True, slots=True)
class Wariant:
    """Nazwa wariantu i zmiany nakładane na konfigurację bazową."""

    nazwa: str
    zmiany: dict[str, object]


@dataclass(frozen=True, slots=True)
class DefinicjaSerii:
    """Wczytana definicja serii przed rozwinięciem wariantów i ziaren."""

    nazwa: str
    bazowa: Path
    katalog: Path
    ziarna: tuple[int, ...]
    stale: dict[str, object]
    warianty: tuple[Wariant, ...]


@dataclass(frozen=True, slots=True)
class PlanPrzebiegu:
    """Jedno połączenie wariantu i ziarna wraz z katalogiem wyników."""

    identyfikator: str
    wariant: Wariant
    ziarno: int
    katalog: Path


def _blad(pole: str, przyczyna: str) -> None:
    raise BladSerii(f"{pole}: {przyczyna}")


def _tekst(tabela: dict, pole: str, sciezka: str) -> str:
    if pole not in tabela:
        _blad(sciezka, "brak wymaganego parametru")
    wartosc = tabela[pole]
    if type(wartosc) is not str or not wartosc:
        _blad(sciezka, "wartość musi być niepustym tekstem")
    return wartosc


def _nazwa(tabela: dict, pole: str, sciezka: str) -> str:
    wartosc = _tekst(tabela, pole, sciezka)
    if DOZWOLONA_NAZWA.fullmatch(wartosc) is None:
        _blad(sciezka, "dozwolone są litery, cyfry oraz znaki '.', '_' i '-'")
    return wartosc


def _zmiany(tabela: object, sciezka: str) -> dict[str, object]:
    if type(tabela) is not dict:
        _blad(sciezka, "wartość musi być tabelą TOML")
    wynik: dict[str, object] = {}
    for klucz, wartosc in tabela.items():
        if type(klucz) is not str or klucz.count(".") != 1:
            _blad(sciezka, "każdy klucz zmiany musi mieć postać 'sekcja.pole'")
        if klucz in POLA_AUTOMATYCZNE:
            _blad(f"{sciezka}.{klucz}", "wartość jest ustawiana automatycznie")
        wynik[klucz] = wartosc
    return wynik


def wczytaj_serie(sciezka: str | Path) -> DefinicjaSerii:
    """Wczytuje i sprawdza definicję serii zgodną z ustalonym formatem."""
    try:
        with open(sciezka, "rb") as plik:
            dane = tomllib.load(plik)
    except tomllib.TOMLDecodeError as blad:
        raise BladSerii(f"plik TOML: niepoprawna składnia ({blad})") from blad

    dozwolone_sekcje = {"seria", "stale", "warianty"}
    nieznane = sorted(set(dane) - dozwolone_sekcje)
    if nieznane:
        _blad(nieznane[0], "nieznana sekcja definicji serii")

    if type(dane.get("seria")) is not dict:
        _blad("seria", "brak wymaganej tabeli")
    seria = dane["seria"]
    dozwolone_pola = {"nazwa", "bazowa", "katalog", "ziarna"}
    nieznane_pola = sorted(set(seria) - dozwolone_pola)
    if nieznane_pola:
        _blad(f"seria.{nieznane_pola[0]}", "nieznany parametr")

    nazwa = _nazwa(seria, "nazwa", "seria.nazwa")
    bazowa = Path(_tekst(seria, "bazowa", "seria.bazowa"))
    katalog = Path(_tekst(seria, "katalog", "seria.katalog"))

    ziarna_dane = seria.get("ziarna")
    if type(ziarna_dane) is not list or not ziarna_dane:
        _blad("seria.ziarna", "wartość musi być niepustą listą")
    if any(type(ziarno) is not int or ziarno < 0 for ziarno in ziarna_dane):
        _blad("seria.ziarna", "ziarna muszą być liczbami całkowitymi >= 0")
    if len(ziarna_dane) != len(set(ziarna_dane)):
        _blad("seria.ziarna", "ziarna nie mogą się powtarzać")

    stale = _zmiany(dane.get("stale", {}), "stale")
    warianty_dane = dane.get("warianty")
    if type(warianty_dane) is not list or not warianty_dane:
        _blad("warianty", "wymagana jest co najmniej jedna tabela [[warianty]]")

    warianty = []
    nazwy_wariantow = set()
    for numer, tabela in enumerate(warianty_dane):
        sciezka_wariantu = f"warianty[{numer}]"
        if type(tabela) is not dict:
            _blad(sciezka_wariantu, "wariant musi być tabelą TOML")
        nazwa_wariantu = _nazwa(tabela, "nazwa", f"{sciezka_wariantu}.nazwa")
        if nazwa_wariantu in nazwy_wariantow:
            _blad(f"{sciezka_wariantu}.nazwa", "nazwy wariantów nie mogą się powtarzać")
        nazwy_wariantow.add(nazwa_wariantu)
        warianty.append(
            Wariant(
                nazwa=nazwa_wariantu,
                zmiany=_zmiany(
                    {klucz: wartosc for klucz, wartosc in tabela.items() if klucz != "nazwa"},
                    sciezka_wariantu,
                ),
            )
        )

    return DefinicjaSerii(
        nazwa=nazwa,
        bazowa=bazowa,
        katalog=katalog,
        ziarna=tuple(ziarna_dane),
        stale=stale,
        warianty=tuple(warianty),
    )


def rozwin_serie(definicja: DefinicjaSerii) -> list[PlanPrzebiegu]:
    """Tworzy plan obejmujący każdą kombinację wariantu i ziarna."""
    plany = []
    for wariant in definicja.warianty:
        for ziarno in definicja.ziarna:
            identyfikator = f"{definicja.nazwa}_{wariant.nazwa}_z{ziarno}"
            plany.append(
                PlanPrzebiegu(
                    identyfikator=identyfikator,
                    wariant=wariant,
                    ziarno=ziarno,
                    katalog=definicja.katalog / identyfikator,
                )
            )
    return plany


def _naloz_zmiany(cfg: dict, zmiany: dict[str, object], zrodlo: str) -> None:
    for sciezka, wartosc in zmiany.items():
        sekcja, pole = sciezka.split(".")
        if sekcja not in cfg or pole not in cfg[sekcja]:
            _blad(f"{zrodlo}.{sciezka}", "parametr nie istnieje w konfiguracji bazowej")
        cfg[sekcja][pole] = wartosc


def zbuduj_konfiguracje(
    definicja: DefinicjaSerii,
    plan: PlanPrzebiegu,
    bazowa: dict,
) -> dict:
    """Składa i waliduje konfigurację jednego planowanego przebiegu."""
    cfg = deepcopy(bazowa)
    _naloz_zmiany(cfg, definicja.stale, "stale")
    _naloz_zmiany(cfg, plan.wariant.zmiany, f"warianty.{plan.wariant.nazwa}")
    cfg["przebieg"]["ziarno"] = plan.ziarno
    cfg["zapis"]["katalog"] = str(plan.katalog)
    return waliduj_konfiguracje(cfg)


def _zapisz_statusy(katalog: Path, statusy: list[dict[str, object]]) -> None:
    """Zapisuje aktualny stan całej serii atomowo do status.csv."""
    katalog.mkdir(parents=True, exist_ok=True)
    sciezka = katalog / "status.csv"
    tymczasowa = katalog / ".status.csv.tmp"
    with tymczasowa.open("w", encoding="utf-8", newline="") as plik:
        writer = csv.DictWriter(plik, fieldnames=KOLUMNY_STATUSU)
        writer.writeheader()
        writer.writerows(statusy)
    tymczasowa.replace(sciezka)


def wykonaj_serie(definicja: DefinicjaSerii) -> list[dict[str, object]]:
    """Wykonuje wszystkie przebiegi, zapisując status po każdej zmianie."""
    bazowa = wczytaj_konfiguracje(definicja.bazowa)
    plany = rozwin_serie(definicja)
    statusy = [
        {
            "id": plan.identyfikator,
            "wariant": plan.wariant.nazwa,
            "ziarno": plan.ziarno,
            "status": "oczekuje",
            "katalog": str(plan.katalog),
            "blad": "",
        }
        for plan in plany
    ]
    _zapisz_statusy(definicja.katalog, statusy)

    for plan, status in zip(plany, statusy):
        if plan.katalog.exists() and any(plan.katalog.iterdir()):
            status["status"] = "pominiety"
            status["blad"] = "katalog wynikowy nie jest pusty"
            _zapisz_statusy(definicja.katalog, statusy)
            continue

        status["status"] = "w_toku"
        _zapisz_statusy(definicja.katalog, statusy)
        try:
            cfg = zbuduj_konfiguracje(definicja, plan, bazowa)
            callback = przygotuj_zapis(cfg)
            uruchom_przebieg(cfg, po_pokoleniu=callback)
        except Exception as blad:
            status["status"] = "blad"
            status["blad"] = f"{type(blad).__name__}: {blad}"
        else:
            status["status"] = "zakonczony"
        _zapisz_statusy(definicja.katalog, statusy)

    return statusy
