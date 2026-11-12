"""Testy pełnej walidacji konfiguracji modelu."""

from copy import deepcopy
from pathlib import Path
import re

import pytest

from moralnosc.konfiguracja import (
    BladKonfiguracji,
    waliduj_konfiguracje,
    wczytaj_konfiguracje,
)


KATALOG_REPO = Path(__file__).resolve().parents[1]
PLIKI_KONFIGURACJI = [
    KATALOG_REPO / "konfiguracje" / "bazowa.toml",
    *sorted((KATALOG_REPO / "konfiguracje" / "pilotaz").glob("*.toml")),
]


def poprawna_konfiguracja() -> dict:
    """Zwraca małą, kompletną konfigurację z wartościami wewnątrz zakresów."""
    return {
        "przebieg": {"ziarno": 67, "pokolenia": 2, "kroki": 3},
        "plansza": {"H": 4, "W": 5},
        "populacja": {"N": 6},
        "zasoby": {"c_min": 1.0, "c_max": 5.0, "alfa": 1.0},
        "obserwacje": {"skala": 5.0},
        "koszty": {"m": 1.0, "ruch": 0.0, "s": 2.0, "c_k": 1.0, "p": 0.5, "x": 5.0},
        "siec": {"tryb": "softmax", "tau": 1.0, "h": 8},
        "ewolucja": {"T": 50.0, "E0": 50.0, "r": "auto", "sigma": 0.2, "w_max": 4.0, "R": 2},
        "zapis": {"katalog": "wyniki/test"},
    }


def ustaw(cfg: dict, sciezka: str, wartosc: object) -> None:
    """Ustawia wartość wskazaną nazwą ``sekcja.pole``."""
    sekcja, pole = sciezka.split(".")
    cfg[sekcja][pole] = wartosc


def test_poprawna_konfiguracja_przechodzi_bez_zmian():
    cfg = poprawna_konfiguracja()
    kopia = deepcopy(cfg)

    wynik = waliduj_konfiguracje(cfg)

    assert wynik is cfg
    assert wynik == kopia


@pytest.mark.parametrize("sciezka", PLIKI_KONFIGURACJI, ids=lambda p: p.name)
def test_wszystkie_konfiguracje_z_repozytorium_przechodza(sciezka):
    cfg = wczytaj_konfiguracje(sciezka)

    assert cfg["przebieg"]["pokolenia"] >= 1


def test_uzupelnia_wartosci_domyslne():
    cfg = poprawna_konfiguracja()
    cfg["plansza"] = {"H": 20, "W": 20}
    del cfg["populacja"]["N"]
    del cfg["obserwacje"]["skala"]
    del cfg["koszty"]["ruch"]
    cfg["siec"]["tryb"] = "argmax"
    del cfg["siec"]["tau"]
    del cfg["siec"]["h"]
    del cfg["ewolucja"]["r"]

    wynik = waliduj_konfiguracje(cfg)

    assert wynik["populacja"]["N"] == 200
    assert wynik["obserwacje"]["skala"] == 5.0
    assert wynik["koszty"]["ruch"] == 0.0
    assert wynik["siec"]["tau"] == 1.0
    assert wynik["siec"]["h"] == 0
    assert wynik["ewolucja"]["r"] == "auto"


def test_odrzuca_brakujaca_sekcje():
    cfg = poprawna_konfiguracja()
    del cfg["plansza"]

    with pytest.raises(BladKonfiguracji, match="plansza.*brak wymaganej sekcji"):
        waliduj_konfiguracje(cfg)


@pytest.mark.parametrize(
    ("sekcja", "pole"),
    [("przebieg", "ziarno"), ("koszty", "m"), ("zapis", "katalog")],
)
def test_odrzuca_brakujacy_wymagany_parametr(sekcja, pole):
    cfg = poprawna_konfiguracja()
    del cfg[sekcja][pole]

    with pytest.raises(BladKonfiguracji, match=re.escape(f"{sekcja}.{pole}")):
        waliduj_konfiguracje(cfg)


def test_odrzuca_nieznana_sekcje_i_parametr():
    cfg = poprawna_konfiguracja()
    cfg["literowka"] = {}
    with pytest.raises(BladKonfiguracji, match="literowka.*nieznana sekcja"):
        waliduj_konfiguracje(cfg)

    cfg = poprawna_konfiguracja()
    cfg["koszty"]["metabolizm"] = 1.0
    with pytest.raises(BladKonfiguracji, match=r"koszty\.metabolizm.*nieznany parametr"):
        waliduj_konfiguracje(cfg)


@pytest.mark.parametrize(
    ("sciezka", "wartosc"),
    [
        ("przebieg.ziarno", 1.0),
        ("plansza.H", True),
        ("zasoby.c_min", 1),
        ("siec.tryb", 1),
        ("siec.h", 2.0),
        ("ewolucja.r", 1),
        ("zapis.katalog", 1),
    ],
)
def test_odrzuca_niepoprawny_typ(sciezka, wartosc):
    cfg = poprawna_konfiguracja()
    ustaw(cfg, sciezka, wartosc)

    with pytest.raises(BladKonfiguracji, match=re.escape(sciezka)):
        waliduj_konfiguracje(cfg)


@pytest.mark.parametrize(
    ("sciezka", "wartosc"),
    [
        ("przebieg.ziarno", -1),
        ("przebieg.pokolenia", 0),
        ("przebieg.kroki", 0),
        ("plansza.H", 1),
        ("plansza.W", 1),
        ("populacja.N", 0),
        ("zasoby.c_min", -0.1),
        ("zasoby.alfa", -0.1),
        ("obserwacje.skala", 0.0),
        ("koszty.m", 0.0),
        ("koszty.ruch", -0.1),
        ("koszty.s", 0.0),
        ("koszty.c_k", -0.1),
        ("koszty.p", 0.0),
        ("koszty.p", 1.1),
        ("koszty.x", 0.0),
        ("siec.tau", 0.0),
        ("siec.h", -1),
        ("ewolucja.T", -0.1),
        ("ewolucja.E0", 0.0),
        ("ewolucja.r", -0.1),
        ("ewolucja.r", 1.1),
        ("ewolucja.sigma", -0.1),
        ("ewolucja.w_max", 0.0),
        ("ewolucja.R", 0),
    ],
)
def test_odrzuca_wartosc_spoza_zakresu(sciezka, wartosc):
    cfg = poprawna_konfiguracja()
    ustaw(cfg, sciezka, wartosc)

    with pytest.raises(BladKonfiguracji, match=re.escape(sciezka)):
        waliduj_konfiguracje(cfg)


def test_odrzuca_nieskonczona_wartosc_float():
    cfg = poprawna_konfiguracja()
    cfg["zasoby"]["alfa"] = float("inf")

    with pytest.raises(BladKonfiguracji, match="zasoby.alfa.*skończona"):
        waliduj_konfiguracje(cfg)


def test_odrzuca_populacje_wieksza_od_planszy():
    cfg = poprawna_konfiguracja()
    cfg["populacja"]["N"] = 21

    with pytest.raises(BladKonfiguracji, match=r"populacja\.N.*plansza\.H.*plansza\.W"):
        waliduj_konfiguracje(cfg)


def test_odrzuca_c_min_wieksze_od_c_max():
    cfg = poprawna_konfiguracja()
    cfg["zasoby"]["c_min"] = 6.0

    with pytest.raises(BladKonfiguracji, match=r"zasoby\.c_max.*zasoby\.c_min"):
        waliduj_konfiguracje(cfg)


def test_odrzuca_nieoplacalna_kradziez():
    cfg = poprawna_konfiguracja()
    cfg["koszty"].update({"p": 0.5, "x": 2.0, "c_k": 1.0})

    with pytest.raises(BladKonfiguracji, match=r"koszty\.p.*koszty\.x.*koszty\.c_k"):
        waliduj_konfiguracje(cfg)


def test_softmax_wymaga_tau():
    cfg = poprawna_konfiguracja()
    del cfg["siec"]["tau"]

    with pytest.raises(BladKonfiguracji, match=r"siec\.tau.*softmax"):
        waliduj_konfiguracje(cfg)


def test_odrzuca_nieznany_tryb_sieci():
    cfg = poprawna_konfiguracja()
    cfg["siec"]["tryb"] = "losowy"

    with pytest.raises(BladKonfiguracji, match=r"siec\.tryb.*argmax.*softmax"):
        waliduj_konfiguracje(cfg)


def test_odrzuca_niepoprawna_skladnie_toml(tmp_path):
    sciezka = tmp_path / "bledna.toml"
    sciezka.write_text("[przebieg\nziarno = 1", encoding="utf-8")

    with pytest.raises(BladKonfiguracji, match="plik TOML.*składnia"):
        wczytaj_konfiguracje(sciezka)
