"""Testy kontroli jakości i podstawowej analizy jednego przebiegu."""

from datetime import datetime
import json

import pandas as pd

from analiza.jakosc import analizuj, main, przygotuj_pokolenia, sprawdz_przebieg


def _konfiguracja() -> dict:
    return {
        "konfiguracja": {
            "przebieg": {"ziarno": 1, "pokolenia": 1, "kroki": 2},
            "populacja": {"N": 2},
            "zasoby": {"alfa": 1.0},
        },
        "metadane": {"commit": "abc123"},
    }


def _tabele():
    pokolenia = pd.DataFrame(
        [
            {
                "pokolenie": 0,
                "zywi_koniec": 1,
                "przezywalnosc": 0.5,
                "zakwalifikowani": 1,
                "awaryjny": False,
                "S": 1,
                "K": 0,
                "N_o": 0,
                "D": 1,
                **{f"akcja_{akcja}": 0 for akcja in range(8)},
            }
        ]
    )
    agenci = pd.DataFrame(
        [
            {
                "pokolenie": 0,
                "id": 0,
                "id_rodzica": -1,
                "S": 1,
                "K": 0,
                "N_o": 0,
                "D": 1,
                "zasoby_koniec": 2.0,
                "kwalifikacja": True,
                "krok_smierci": 2,
                "przyczyna_smierci": 2,
            },
            {
                "pokolenie": 0,
                "id": 1,
                "id_rodzica": -1,
                "S": 0,
                "K": 0,
                "N_o": 0,
                "D": 0,
                "zasoby_koniec": -0.1,
                "kwalifikacja": False,
                "krok_smierci": 1,
                "przyczyna_smierci": 1,
            },
        ]
    )
    dylematy = pd.DataFrame(
        [
            {
                "pokolenie": 0,
                "krok": 0,
                "id": 0,
                "id_celu": 1,
                "zasoby_agenta": 2.0,
                "zasoby_celu": 1.0,
                "akcja": 5,
                "klasa": "dzielenie",
                "wynik": "sukces",
                "kwota": 1.0,
            }
        ]
    )
    return pokolenia, agenci, dylematy


def _zapisz_przebieg(katalog):
    katalog.mkdir()
    konfiguracja = _konfiguracja()
    pokolenia, agenci, dylematy = _tabele()
    (katalog / "konfiguracja.json").write_text(
        json.dumps(konfiguracja), encoding="utf-8"
    )
    pokolenia.to_csv(katalog / "pokolenia.csv", index=False)
    agenci.to_csv(katalog / "agenci.csv", index=False)
    dylematy.to_csv(katalog / "dylematy.csv", index=False)


def test_sprawdz_przebieg_wykrywa_niezgodne_D():
    """Kontrola wykrywa naruszenie równania S + K + N_o = D."""
    pokolenia, agenci, dylematy = _tabele()
    agenci.loc[0, "D"] = 2

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = next(
        wynik for wynik in raport if wynik["kontrola"] == "S_plus_K_plus_N_o_rowne_D"
    )
    assert kontrola["poprawna"] is False


def test_przygotuj_pokolenia_ustawia_nan_gdy_brak_dylematow():
    """M_pop ma wartość NaN, gdy w pokoleniu nie wystąpił żaden dylemat."""
    pokolenia, _, _ = _tabele()
    pokolenia.loc[0, ["S", "K", "N_o", "D"]] = 0

    wynik = przygotuj_pokolenia(_konfiguracja(), pokolenia)

    assert pd.isna(wynik.loc[0, "M_pop"])


def test_przygotuj_pokolenia_oblicza_moralnosc_populacji():
    """M_pop jest różnicą S i K podzieloną przez liczbę dylematów."""
    pokolenia, _, _ = _tabele()
    pokolenia.loc[0, ["S", "K", "N_o", "D"]] = [3, 1, 0, 4]

    wynik = przygotuj_pokolenia(_konfiguracja(), pokolenia)

    assert wynik.loc[0, "M_pop"] == 0.5


def test_analizuj_zapisuje_raport_agregaty_i_wykresy(tmp_path):
    """Poprawny przebieg tworzy raport, dane analizy i trzy wykresy."""
    katalog_wynikow = tmp_path / "przebieg"
    _zapisz_przebieg(katalog_wynikow)
    katalog_analizy = tmp_path / "analiza"

    poprawne = analizuj(katalog_wynikow, katalog_analizy)

    assert poprawne
    assert {plik.name for plik in katalog_analizy.iterdir()} == {
        "kontrola_jakosci.csv",
        "pokolenia_analiza.csv",
        "przezywalnosc.png",
        "moralnosc_populacji.png",
        "liczba_rodzicow.png",
    }
    dane = pd.read_csv(katalog_analizy / "pokolenia_analiza.csv")
    assert dane["M_pop"].tolist() == [1.0]
    assert dane["rodzice"].tolist() == [1]


def test_analizuj_zglasza_brak_wymaganego_pliku(tmp_path):
    """Niekompletny katalog daje błąd kontroli bez tworzenia wykresów."""
    katalog_wynikow = tmp_path / "przebieg"
    _zapisz_przebieg(katalog_wynikow)
    (katalog_wynikow / "dylematy.csv").unlink()
    katalog_analizy = tmp_path / "analiza"

    poprawne = analizuj(katalog_wynikow, katalog_analizy)

    assert not poprawne
    raport = pd.read_csv(katalog_analizy / "kontrola_jakosci.csv")
    assert not bool(raport.loc[0, "poprawna"])
    assert raport.loc[0, "szczegoly"] == "dylematy.csv"
    assert not (katalog_analizy / "przezywalnosc.png").exists()


def test_main_tworzy_katalog_z_nazwa_przebiegu_i_data(tmp_path):
    """Interfejs tworzy osobny katalog opisany nazwą przebiegu i czasem."""
    katalog_wynikow = tmp_path / "przebieg_testowy"
    _zapisz_przebieg(katalog_wynikow)
    katalog_bazowy = tmp_path / "analizy"

    kod_wyjscia = main(
        [str(katalog_wynikow), "--wyjscie", str(katalog_bazowy)]
    )

    assert kod_wyjscia == 0
    katalogi = list(katalog_bazowy.iterdir())
    assert len(katalogi) == 1
    prefiks = "przebieg_testowy_"
    assert katalogi[0].name.startswith(prefiks)
    datetime.strptime(katalogi[0].name.removeprefix(prefiks), "%Y-%m-%d_%H-%M-%S")
    assert (katalogi[0] / "kontrola_jakosci.csv").is_file()
