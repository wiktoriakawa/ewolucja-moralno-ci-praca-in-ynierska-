"""Test interfejsu uruchamiania przebiegu z pliku konfiguracyjnego."""

import json

from moralnosc import __main__ as cli
from moralnosc.konfiguracja import BladKonfiguracji


main = cli.main


def test_main_uruchamia_przebieg_i_zapisuje_wyniki(tmp_path):
    """Polecenie przebieg tworzy JSON i trzy pliki CSV."""
    katalog_wynikow = tmp_path / "wyniki"
    konfiguracja = tmp_path / "test.toml"
    konfiguracja.write_text(
        f"""
[przebieg]
ziarno = 67
pokolenia = 2
kroki = 2

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
E0 = 100.0
r = 0.0
sigma = 0.0
w_max = 4.0
R = 1

[zapis]
katalog = "{katalog_wynikow.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    kod_wyjscia = main(["przebieg", str(konfiguracja)])

    assert kod_wyjscia == 0
    assert {sciezka.name for sciezka in katalog_wynikow.iterdir()} == {
        "konfiguracja.json",
        "pokolenia.csv",
        "agenci.csv",
        "dylematy.csv",
    }
    with (katalog_wynikow / "konfiguracja.json").open(encoding="utf-8") as plik:
        zapisana_konfiguracja = json.load(plik)["konfiguracja"]
    assert zapisana_konfiguracja["przebieg"]["ziarno"] == 67
    assert zapisana_konfiguracja["przebieg"]["pokolenia"] == 2


def test_main_czytelnie_konczy_program_dla_blednej_konfiguracji(
    tmp_path, monkeypatch, capsys
):
    """Błąd konfiguracji daje kod 2 i jedną wiadomość zamiast tracebacka."""
    def zglos_blad(_sciezka):
        raise BladKonfiguracji("siec.h: wartość musi być >= 0")

    monkeypatch.setattr(cli, "wczytaj_konfiguracje", zglos_blad)

    kod_wyjscia = main(["przebieg", str(tmp_path / "bledna.toml")])

    wynik = capsys.readouterr()
    assert kod_wyjscia == 2
    assert wynik.out == ""
    assert "siec.h" in wynik.err
    assert "Traceback" not in wynik.err


def test_main_uruchamia_serie_i_zwraca_blad_gdy_jeden_przebieg_sie_nie_udaje(
    tmp_path, monkeypatch, capsys
):
    """Polecenie seria raportuje wszystkie statusy i zwraca kod 1 po błędzie przebiegu."""
    definicja_toml = tmp_path / "seria.toml"
    definicja_toml.write_text(
        """
[seria]
nazwa = "test"
bazowa = "bazowa.toml"
katalog = "wyniki"
ziarna = [1]

[[warianty]]
nazwa = "wariant"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "wykonaj_serie",
        lambda _definicja: [
            {
                "id": "test_wariant_z1",
                "status": "blad",
                "blad": "RuntimeError: awaria",
            }
        ],
    )

    kod_wyjscia = main(["seria", str(definicja_toml)])

    wynik = capsys.readouterr()
    assert kod_wyjscia == 1
    assert "test_wariant_z1: blad" in wynik.out
    assert "RuntimeError: awaria" in wynik.out
