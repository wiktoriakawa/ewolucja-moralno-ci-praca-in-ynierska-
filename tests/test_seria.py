"""Testy definiowania i wykonywania serii przebiegów."""

import csv
import json
from pathlib import Path

import pytest

from moralnosc import seria


KATALOG_REPO = Path(__file__).resolve().parents[1]


def _zapisz_bazowa(sciezka) -> None:
    """Zapisuje małą konfigurację bazową odpowiednią do szybkich testów."""
    sciezka.write_text(
        """
[przebieg]
ziarno = 1
pokolenia = 1
kroki = 1

[plansza]
H = 2
W = 2

[populacja]
N = 2

[zasoby]
c_min = 1.0
c_max = 2.0
alfa = 1.0

[obserwacje]
skala = 2.0

[koszty]
m = 0.1
ruch = 0.0
s = 1.0
c_k = 0.5
p = 1.0
x = 1.0

[siec]
tryb = "argmax"
tau = 1.0
h = 0

[ewolucja]
T = 0.0
E0 = 10.0
r = 0.0
sigma = 0.0
w_max = 4.0
R = 1

[zapis]
katalog = "wyniki/bazowa"
""".strip(),
        encoding="utf-8",
    )


def _zapisz_definicje(
    sciezka,
    bazowa,
    katalog,
    *,
    ziarna=(11, 12),
    warianty=("niska", "wysoka"),
) -> None:
    """Zapisuje niewielką definicję serii w formacie K1.toml."""
    tabele_wariantow = "\n\n".join(
        f"""[[warianty]]
nazwa = "{nazwa}"
"obserwacje.skala" = {wartosc}.0"""
        for nazwa, wartosc in zip(warianty, range(2, 2 + len(warianty)))
    )
    lista_ziaren = ", ".join(str(ziarno) for ziarno in ziarna)
    sciezka.write_text(
        f"""
[seria]
nazwa = "test"
bazowa = "{bazowa.as_posix()}"
katalog = "{katalog.as_posix()}"
ziarna = [{lista_ziaren}]

[stale]
"przebieg.pokolenia" = 2

{tabele_wariantow}
""".strip(),
        encoding="utf-8",
    )


def _wczytaj_statusy(sciezka):
    with sciezka.open(encoding="utf-8", newline="") as plik:
        return list(csv.DictReader(plik))


def test_wczytaj_i_rozwin_serie_tworzy_iloczyn_wariantow_i_ziaren(tmp_path):
    """Dwa warianty i dwa ziarna tworzą cztery jednoznaczne przebiegi."""
    bazowa = tmp_path / "bazowa.toml"
    definicja_toml = tmp_path / "seria.toml"
    katalog = tmp_path / "wyniki"
    _zapisz_bazowa(bazowa)
    _zapisz_definicje(definicja_toml, bazowa, katalog)

    definicja = seria.wczytaj_serie(definicja_toml)
    plany = seria.rozwin_serie(definicja)

    assert [plan.identyfikator for plan in plany] == [
        "test_niska_z11",
        "test_niska_z12",
        "test_wysoka_z11",
        "test_wysoka_z12",
    ]
    assert [plan.katalog for plan in plany] == [
        katalog / plan.identyfikator for plan in plany
    ]


def test_k1_tworzy_27_poprawnych_konfiguracji():
    """Repozytoryjna definicja K1 pozostaje zgodna ze schematem konfiguracji."""
    definicja = seria.wczytaj_serie(KATALOG_REPO / "konfiguracje/kalibracja/K1.toml")
    bazowa = seria.wczytaj_konfiguracje(KATALOG_REPO / definicja.bazowa)
    plany = seria.rozwin_serie(definicja)

    konfiguracje = [
        seria.zbuduj_konfiguracje(definicja, plan, bazowa) for plan in plany
    ]

    assert len(konfiguracje) == 27
    assert {cfg["siec"]["h"] for cfg in konfiguracje} == {8}
    assert {cfg["przebieg"]["ziarno"] for cfg in konfiguracje} == {401, 402, 403}


def test_zbuduj_konfiguracje_naklada_stale_wariant_i_pola_automatyczne(tmp_path):
    """Konfiguracja przebiegu otrzymuje zmiany wspólne, wariant, ziarno i katalog."""
    bazowa_toml = tmp_path / "bazowa.toml"
    definicja_toml = tmp_path / "seria.toml"
    katalog = tmp_path / "wyniki"
    _zapisz_bazowa(bazowa_toml)
    _zapisz_definicje(definicja_toml, bazowa_toml, katalog)
    definicja = seria.wczytaj_serie(definicja_toml)
    bazowa = seria.wczytaj_konfiguracje(bazowa_toml)
    plan = seria.rozwin_serie(definicja)[0]

    cfg = seria.zbuduj_konfiguracje(definicja, plan, bazowa)

    assert cfg["przebieg"]["pokolenia"] == 2
    assert cfg["obserwacje"]["skala"] == 2.0
    assert cfg["przebieg"]["ziarno"] == 11
    assert cfg["zapis"]["katalog"] == str(katalog / "test_niska_z11")
    assert bazowa["przebieg"]["pokolenia"] == 1


def test_wykonaj_serie_zapisuje_wyniki_i_status_kazdego_przebiegu(tmp_path):
    """Mała seria tworzy osobne wyniki i końcowy status dla każdego przebiegu."""
    bazowa = tmp_path / "bazowa.toml"
    definicja_toml = tmp_path / "seria.toml"
    katalog = tmp_path / "wyniki"
    _zapisz_bazowa(bazowa)
    _zapisz_definicje(
        definicja_toml,
        bazowa,
        katalog,
        ziarna=(11, 12),
        warianty=("jeden",),
    )

    statusy = seria.wykonaj_serie(seria.wczytaj_serie(definicja_toml))

    assert [status["status"] for status in statusy] == ["zakonczony", "zakonczony"]
    assert [wiersz["status"] for wiersz in _wczytaj_statusy(katalog / "status.csv")] == [
        "zakonczony",
        "zakonczony",
    ]
    for ziarno in (11, 12):
        katalog_przebiegu = katalog / f"test_jeden_z{ziarno}"
        assert {sciezka.name for sciezka in katalog_przebiegu.iterdir()} == {
            "konfiguracja.json",
            "pokolenia.csv",
            "agenci.csv",
            "dylematy.csv",
        }
        with (katalog_przebiegu / "konfiguracja.json").open(encoding="utf-8") as plik:
            cfg = json.load(plik)["konfiguracja"]
        assert cfg["przebieg"]["ziarno"] == ziarno


def test_wykonaj_serie_nie_nadpisuje_istniejacych_wynikow(tmp_path):
    """Ponowne uruchomienie pomija niepusty katalog zamiast zmieniać jego pliki."""
    bazowa = tmp_path / "bazowa.toml"
    definicja_toml = tmp_path / "seria.toml"
    katalog = tmp_path / "wyniki"
    _zapisz_bazowa(bazowa)
    _zapisz_definicje(
        definicja_toml,
        bazowa,
        katalog,
        ziarna=(11,),
        warianty=("jeden",),
    )
    definicja = seria.wczytaj_serie(definicja_toml)
    seria.wykonaj_serie(definicja)
    konfiguracja = katalog / "test_jeden_z11" / "konfiguracja.json"
    zawartosc_przed = konfiguracja.read_bytes()

    statusy = seria.wykonaj_serie(definicja)

    assert statusy[0]["status"] == "pominiety"
    assert konfiguracja.read_bytes() == zawartosc_przed


def test_wykonaj_serie_kontynuuje_po_bledzie_jednego_przebiegu(tmp_path, monkeypatch):
    """Błąd jednego ziarna jest zapisany, ale nie zatrzymuje następnego przebiegu."""
    bazowa = tmp_path / "bazowa.toml"
    definicja_toml = tmp_path / "seria.toml"
    katalog = tmp_path / "wyniki"
    _zapisz_bazowa(bazowa)
    _zapisz_definicje(
        definicja_toml,
        bazowa,
        katalog,
        ziarna=(11, 12),
        warianty=("jeden",),
    )

    def uruchom_lub_zglos_blad(cfg, *, po_pokoleniu):
        if cfg["przebieg"]["ziarno"] == 11:
            raise RuntimeError("celowy błąd testu")

    monkeypatch.setattr(seria, "uruchom_przebieg", uruchom_lub_zglos_blad)

    statusy = seria.wykonaj_serie(seria.wczytaj_serie(definicja_toml))

    assert [status["status"] for status in statusy] == ["blad", "zakonczony"]
    assert "celowy błąd testu" in statusy[0]["blad"]
    assert [wiersz["status"] for wiersz in _wczytaj_statusy(katalog / "status.csv")] == [
        "blad",
        "zakonczony",
    ]


def test_wczytaj_serie_odrzuca_nadpisanie_pola_ustawianego_automatycznie(tmp_path):
    """Ziarno i katalog wyników nie mogą zostać ukryte w zmianach wariantu."""
    definicja_toml = tmp_path / "seria.toml"
    definicja_toml.write_text(
        """
[seria]
nazwa = "test"
bazowa = "bazowa.toml"
katalog = "wyniki"
ziarna = [1]

[[warianty]]
nazwa = "bledny"
"przebieg.ziarno" = 999
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(seria.BladSerii, match=r"przebieg\.ziarno.*automatycznie"):
        seria.wczytaj_serie(definicja_toml)
