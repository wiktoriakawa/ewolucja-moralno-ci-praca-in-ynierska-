"""Test interfejsu uruchamiania przebiegu z pliku konfiguracyjnego."""

import json

from moralnosc.__main__ import main


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
