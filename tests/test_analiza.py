"""Testy kontroli jakości i podstawowej analizy jednego przebiegu."""

from datetime import datetime
import json

import pandas as pd

from analiza.jakosc import (
    analizuj,
    main,
    porownaj_przebiegi,
    przygotuj_pokolenia,
    sprawdz_przebieg,
)


def _konfiguracja() -> dict:
    return {
        "konfiguracja": {
            "przebieg": {"ziarno": 1, "pokolenia": 1, "kroki": 2},
            "populacja": {"N": 2},
            "zasoby": {"alfa": 1.0},
            "koszty": {"s": 1.0, "x": 1.0},
            "ewolucja": {"T": 1.5},
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


def _kontrola(raport, nazwa):
    """Wyszukuje wskazaną kontrolę w raporcie."""
    return next(wynik for wynik in raport if wynik["kontrola"] == nazwa)


def test_sprawdz_przebieg_wykrywa_niezgodne_D():
    """Kontrola wykrywa naruszenie równania S + K + N_o = D."""
    pokolenia, agenci, dylematy = _tabele()
    agenci.loc[0, "D"] = 2

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = _kontrola(raport, "S_plus_K_plus_N_o_rowne_D")
    assert kontrola["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_krok_konca_u_zmarlego():
    """Zmarły agent nie może mieć kroku oznaczającego przeżycie."""
    pokolenia, agenci, dylematy = _tabele()
    agenci.loc[1, "krok_smierci"] = 2

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "zakres_kroku_smierci")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_niezgodne_sumy_pokolenia():
    """Agregaty pokolenia muszą być sumami liczników agentów."""
    pokolenia, agenci, dylematy = _tabele()
    pokolenia.loc[0, "S"] = 2

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = _kontrola(raport, "sumy_agentow_rowne_pokoleniom")
    assert kontrola["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_kwalifikacje_zmarlego():
    """Zmarły agent nie może zostać zakwalifikowany."""
    pokolenia, agenci, dylematy = _tabele()
    agenci.loc[1, "kwalifikacja"] = True
    pokolenia.loc[0, "zakwalifikowani"] = 2

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = _kontrola(raport, "kwalifikacja_zgodna_z_regula")
    assert kontrola["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_zerowa_kwote_dzielenia():
    """Udane dzielenie musi przekazywać kwotę s."""
    pokolenia, agenci, dylematy = _tabele()
    dylematy.loc[0, "kwota"] = 0.0

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = _kontrola(raport, "kwota_udanego_dzielenia")
    assert kontrola["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_dylemat_przypisany_innemu_agentowi():
    """Liczba zdarzeń każdego agenta musi być równa jego D."""
    pokolenia, agenci, dylematy = _tabele()
    dylematy.loc[0, "id"] = 1

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = _kontrola(raport, "liczba_dylematow_rowna_D_agenta")
    assert kontrola["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_dodatnie_zasoby_zmarlego():
    """Agent zmarły z wyczerpania musi mieć najwyżej zero zasobów."""
    pokolenia, agenci, dylematy = _tabele()
    agenci.loc[1, "zasoby_koniec"] = 0.1

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "zasoby_zgodne_ze_smiercia")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_nieprawidlowe_zasoby_dylematu():
    """Dylemat wymaga zasobów dawcy na s i dodatnich zasobów celu."""
    pokolenia, agenci, dylematy = _tabele()
    dylematy.loc[0, "zasoby_agenta"] = 0.5

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "zasoby_w_dylemacie")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_klase_niezgodna_z_akcja():
    """Kod akcji dzielenia musi mieć klasę dzielenie."""
    pokolenia, agenci, dylematy = _tabele()
    dylematy.loc[0, "klasa"] = "kradziez"

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "klasa_zgodna_z_akcja")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_zbyt_duza_kwote_kradziezy():
    """Kwota kradzieży nie może przekraczać parametru x."""
    pokolenia, agenci, dylematy = _tabele()
    dylematy.loc[0, ["akcja", "klasa", "wynik", "kwota"]] = [
        6,
        "kradziez",
        "sukces",
        2.0,
    ]

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "kwota_kradziezy")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_kwote_niewykonanej_akcji():
    """Niewykonana akcja nie może przenosić zasobów."""
    pokolenia, agenci, dylematy = _tabele()
    dylematy.loc[0, ["wynik", "kwota"]] = ["niewykonana", 1.0]

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    kontrola = _kontrola(raport, "zerowa_kwota_niewykonanej")
    assert kontrola["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_numeracje_bez_pokolenia_zerowego():
    """Numeracja pokoleń musi rozpoczynać się od zera i nie mieć luk."""
    pokolenia, agenci, dylematy = _tabele()
    pokolenia["pokolenie"] = 1
    agenci["pokolenie"] = 1
    dylematy["pokolenie"] = 1

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "ciaglosc_numerow_pokolen")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_nadmiarowe_pokolenie():
    """Nie można zapisać więcej pokoleń niż przewiduje konfiguracja."""
    pokolenia, agenci, dylematy = _tabele()
    dodatkowe = pokolenia.copy()
    dodatkowe["pokolenie"] = 1
    pokolenia = pd.concat([pokolenia, dodatkowe], ignore_index=True)

    raport = sprawdz_przebieg(_konfiguracja(), pokolenia, agenci, dylematy)

    assert _kontrola(raport, "zakonczenie_przebiegu")["poprawna"] is False


def test_sprawdz_przebieg_wykrywa_wymarcie_przed_ostatnim_pokoleniem():
    """Po pokoleniu zakończonym wymarciem nie może być kolejnego wpisu."""
    konfiguracja = _konfiguracja()
    konfiguracja["konfiguracja"]["przebieg"]["pokolenia"] = 2
    pokolenia, agenci, dylematy = _tabele()
    pokolenia.loc[0, "zywi_koniec"] = 0
    dodatkowe = pokolenia.copy()
    dodatkowe.loc[0, ["pokolenie", "zywi_koniec"]] = [1, 1]
    pokolenia = pd.concat([pokolenia, dodatkowe], ignore_index=True)

    raport = sprawdz_przebieg(konfiguracja, pokolenia, agenci, dylematy)

    assert _kontrola(raport, "zakonczenie_przebiegu")["poprawna"] is False


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


def test_porownaj_przebiegi_sprawdza_powtarzalnosc_csv(tmp_path):
    """NF-03 przechodzi dla identycznych CSV i wykrywa zmieniony plik."""
    katalog_a = tmp_path / "P3_a"
    katalog_b = tmp_path / "P3_b"
    _zapisz_przebieg(katalog_a)
    _zapisz_przebieg(katalog_b)

    zgodne = porownaj_przebiegi(katalog_a, katalog_b)

    assert all(kontrola["poprawna"] for kontrola in zgodne)

    agenci = pd.read_csv(katalog_b / "agenci.csv")
    agenci.loc[0, "zasoby_koniec"] = 999.0
    agenci.to_csv(katalog_b / "agenci.csv", index=False)

    rozne = porownaj_przebiegi(katalog_a, katalog_b)

    assert _kontrola(rozne, "powtarzalnosc_agenci")["poprawna"] is False
