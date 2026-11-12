"""Testy eksploracja.py; wartości liczone ręcznie z definicji 3.4.3.

Ostatni test uruchamia cały skrypt na małym, świeżo zasymulowanym pilotażu.
    PYTHONPATH=src python -m pytest analiza -q
"""
import re

import numpy as np
import pandas as pd
import pytest

from analiza import eksploracja
from moralnosc.__main__ import main as uruchom_symulacje


@pytest.fixture
def agenci():
    """Trzech agentów jednego pokolenia: częściej dzieli, bez dylematów, częściej kradnie."""
    return pd.DataFrame(
        {
            "przebieg": ["X", "X", "X"],
            "pokolenie": [0, 0, 0],
            "S": [3, 0, 0],
            "K": [1, 0, 1],
            "N_o": [0, 0, 3],
            "D": [4, 0, 4],
        }
    )


# --- współczynniki ----------------------------------------------------------------------


def test_profil_i_M(agenci):
    wynik = eksploracja.dodaj_wspolczynniki(agenci, a=2.0)

    np.testing.assert_allclose(wynik.loc[0, ["s", "k", "n", "M"]], [0.75, 0.25, 0.0, 0.5])
    np.testing.assert_allclose(wynik.loc[2, ["s", "k", "n", "M"]], [0.0, 0.25, 0.75, -0.25])
    # D = 0: brak obserwacji, a nie zachowanie neutralne
    assert wynik.loc[1, ["s", "k", "n", "M"]].isna().all()


def test_wygladzanie(agenci):
    wynik = eksploracja.dodaj_wspolczynniki(agenci, a=2.0)

    # M_wyg = (S − K) / (D + a): 2/6, 0/2, −1/6
    np.testing.assert_allclose(wynik["M_wyg"], [2 / 6, 0.0, -1 / 6])
    # (S+1, K+1, N_o+1) / (D+3)
    np.testing.assert_allclose(wynik.loc[0, ["s_wyg", "k_wyg", "n_wyg"]], [4 / 7, 2 / 7, 1 / 7])
    np.testing.assert_allclose(wynik.loc[1, ["s_wyg", "k_wyg", "n_wyg"]], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(wynik[["s_wyg", "k_wyg", "n_wyg"]].sum(axis=1), 1.0)


def test_dodaj_wspolczynniki_nie_zmienia_wejscia(agenci):
    kolumny_przed = list(agenci.columns)
    eksploracja.dodaj_wspolczynniki(agenci, a=2.0)
    assert list(agenci.columns) == kolumny_przed


def test_m_pop(agenci):
    bez_dylematow = agenci.assign(pokolenie=1, S=0, K=0, N_o=0, D=0)
    dane = pd.concat([agenci, bez_dylematow], ignore_index=True)

    wynik = eksploracja.m_pop(dane).set_index("pokolenie")["M_pop"]

    assert wynik[0] == pytest.approx((3 - 2) / 8)  # ΣS = 3, ΣK = 2, ΣD = 8
    assert np.isnan(wynik[1])  # ΣD = 0 → współczynnik nieokreślony


# --- 4.3.1 ------------------------------------------------------------------------------


def test_podsumuj_D():
    agenci = pd.DataFrame({"D": [0, 1, 4, 10], "przyczyna_smierci": [1, 1, 2, 2]})

    wynik = eksploracja.podsumuj_D(agenci, D_min=5)

    wszyscy = wynik.loc["wszyscy"]
    assert wszyscy["liczba"] == 4
    assert wszyscy["odsetek_D0"] == pytest.approx(0.25)
    assert wszyscy["odsetek_ponizej_Dmin"] == pytest.approx(0.75)
    # interpolacja liniowa (domyślna w pandas)
    assert wszyscy[["Q1", "mediana", "Q3"]].tolist() == pytest.approx([0.75, 2.5, 5.5])
    assert wynik.loc["zmarli", "mediana"] == pytest.approx(0.5)
    assert wynik.loc["dożyli końca", "odsetek_ponizej_Dmin"] == pytest.approx(0.5)


def test_rozrzut_M_wg_D_szum():
    # jedno dzielenie i jedna kradzież: p_s = p_k = 0,5 → Var = 1, szum przy D = 1 to 1
    agenci = eksploracja.dodaj_wspolczynniki(
        pd.DataFrame({"S": [1, 0], "K": [0, 1], "N_o": [0, 0], "D": [1, 1]}), a=2.0
    )

    wynik = eksploracja.rozrzut_M_wg_D(agenci, D_min=5)

    wiersz = wynik.loc["1"]
    assert wiersz["liczba"] == 2
    assert wiersz["odch_std_szum"] == pytest.approx(1.0)
    assert wiersz["odch_std_M"] == pytest.approx(np.sqrt(2))  # M = 1 i −1, std z próby
    assert wiersz["odsetek_skrajnych"] == pytest.approx(1.0)


def test_etykiety_przedzialow():
    assert eksploracja._etykiety_przedzialow([0, 1, 3, 5, 10]) == ["0", "1–2", "3–4", "5–9"]


# --- 4.4.2 ------------------------------------------------------------------------------


def test_opisz_pomija_NaN():
    wynik = eksploracja.opisz(pd.Series([1.0, 2.0, np.nan, 3.0]))
    assert wynik["liczba"] == 3
    assert [wynik[k] for k in ("srednia", "mediana", "odch_std", "min", "maks")] == pytest.approx(
        [2.0, 2.0, 1.0, 1.0, 3.0]
    )


def test_statystyki_opisowe_krok_smierci_tylko_zmarlych(agenci):
    dane = eksploracja.dodaj_wspolczynniki(
        agenci.assign(zasoby_koniec=[5.0, -1.0, 2.0], krok_smierci=[300, 40, 300],
                      przyczyna_smierci=[2, 1, 2]),
        a=2.0,
    )
    pokolenia = pd.DataFrame({"przezywalnosc": [2 / 3], "zakwalifikowani": [1], "M_pop": [0.125]})

    wynik = eksploracja.statystyki_opisowe(dane, pokolenia)

    assert list(wynik.index)[:4] == ["D (agenci)", "S (agenci)", "K (agenci)", "N_o (agenci)"]
    assert wynik.loc["krok śmierci (zmarli agenci)", "liczba"] == 1
    assert wynik.loc["krok śmierci (zmarli agenci)", "srednia"] == 40
    assert wynik.loc["M (agenci z D > 0)", "liczba"] == 2  # agent z D = 0 pominięty


def test_statystyki_pierwsze_ostatnie(agenci):
    # pokolenie 1 = pokolenie 0 z D większym o 2
    pierwsze = agenci.assign(alfa=0.85, zasoby_koniec=1.0, krok_smierci=300, przyczyna_smierci=2)
    ostatnie = pierwsze.assign(pokolenie=1, N_o=pierwsze["N_o"] + 2, D=pierwsze["D"] + 2)
    dane = eksploracja.dodaj_wspolczynniki(pd.concat([pierwsze, ostatnie], ignore_index=True), a=2.0)
    pokolenia = pd.DataFrame(
        {"alfa": [0.85, 0.85], "pokolenie": [0, 1], "przezywalnosc": [0.5, 1.0],
         "zakwalifikowani": [1, 3], "M_pop": [0.125, 0.1]}
    )

    wynik = eksploracja.statystyki_pierwsze_ostatnie(dane, pokolenia)

    assert list(wynik.columns) == [(0.85, 0), (0.85, 1)]
    assert wynik.loc["D (agenci)", (0.85, 0)] == pytest.approx(8 / 3)
    assert wynik.loc["D (agenci)", (0.85, 1)] == pytest.approx(14 / 3)
    assert wynik.loc["przeżywalność (pokolenia)", (0.85, 1)] == pytest.approx(1.0)


# --- 4.4.3 ------------------------------------------------------------------------------


def test_udzialy_akcji():
    pokolenia = pd.DataFrame({f"akcja_{kod}": [kod, kod] for kod in range(8)})
    wynik = eksploracja.udzialy_akcji(pokolenia)
    assert wynik.sum() == pytest.approx(1.0)
    assert wynik["nic"] == pytest.approx(7 / 28)  # akcja 7: 14 z 56 decyzji
    assert list(wynik.index)[4:6] == ["zbierz", "dziel"]


def test_najczestsza_akcja_w_dylematach():
    dylematy = pd.DataFrame(
        {"przebieg": ["X"] * 5, "id": [1, 1, 1, 1, 2], "akcja": [4, 4, 4, 5, 7]}
    )
    wynik = eksploracja.najczestsza_akcja_w_dylematach(dylematy, min_decyzji=2)
    assert wynik.to_dict() == {("X", 1): pytest.approx(0.75)}  # agent 2 ma 1 decyzję


# --- 4.4.4 ------------------------------------------------------------------------------


def test_przebieg_w_pokoleniach_liczy_ziarna():
    pokolenia = pd.DataFrame(
        {
            "alfa": [0.85, 0.85, 0.85],
            "przebieg": ["a", "a", "b"],
            "pokolenie": [0, 1, 0],
            "przezywalnosc": [0.2, 0.4, 0.0],
            "zakwalifikowani": [10, 20, 0],
            "M_pop": [0.1, 0.3, np.nan],
            "awaryjny": [False, False, True],
        }
    )
    wynik = eksploracja.przebieg_w_pokoleniach(pokolenia).set_index("pokolenie")
    assert wynik["liczba_ziaren"].tolist() == [2, 1]
    assert wynik.loc[0, "przezywalnosc_sr"] == pytest.approx(0.1)
    assert wynik.loc[0, "M_pop_sr"] == pytest.approx(0.1)  # NaN pominięty
    assert wynik.loc[0, "awaryjny_liczba"] == 1


def test_udzialy_akcji_w_pokoleniach():
    # dwa ziarna: najpierw suma zliczeń, potem dzielenie
    zliczenia = {f"akcja_{kod}": [0, 0] for kod in range(8)}
    zliczenia["akcja_0"] = [2, 0]  # ruch
    zliczenia["akcja_4"] = [2, 4]  # zbierz
    zliczenia["akcja_5"] = [0, 2]  # dziel
    pokolenia = pd.DataFrame({"alfa": [1.0, 1.0], "pokolenie": [0, 0], **zliczenia})

    wynik = eksploracja.udzialy_akcji_w_pokoleniach(pokolenia).iloc[0]

    assert wynik["ruch"] == pytest.approx(0.2)
    assert wynik["zbierz"] == pytest.approx(0.6)
    assert wynik["dziel"] == pytest.approx(0.2)
    assert wynik[list(eksploracja.GRUPY_AKCJI)].sum() == pytest.approx(1.0)


def test_etykieta_alfa():
    assert eksploracja._etykieta_alfa(0.85) == "α = 0,85"
    assert eksploracja._etykieta_alfa(1.0) == "α = 1,0"
    assert eksploracja._etykieta_alfa_tex(0.85) == "$\\alpha = 0{,}85$"


# --- tabele LaTeX-a -----------------------------------------------------------------------


def test_liczba_tex():
    assert eksploracja.liczba_tex(3.0) == "3"
    assert eksploracja.liczba_tex(0.25) == "0{,}25"
    assert eksploracja.liczba_tex(-0.25) == "$-$0{,}25"
    assert eksploracja.liczba_tex(-0.001) == "0"  # bez „−0”
    assert eksploracja.liczba_tex(np.nan) == "--"
    assert eksploracja.procent_tex(0.314) == "31{,}4\\%"


def test_zapisz_tabele_tex(tmp_path):
    plik = tmp_path / "tab.tex"
    eksploracja.zapisz_tabele_tex(
        [["a", "1"], ["b", "2"]], plik, naglowek=["Zmienna", "Wartość"], podpis="Podpis", etykieta="tab:x"
    )
    tekst = plik.read_text(encoding="utf-8")
    assert "\\begin{longtable}" in tekst and "\\end{longtable}" in tekst
    assert "\\caption{Podpis}\\label{tab:x}" in tekst
    assert "  a & 1 \\\\" in tekst and "  b & 2 \\\\" in tekst
    assert tekst.count("{") == tekst.count("}")


# --- całość -----------------------------------------------------------------------------

KONFIGURACJA = """
[przebieg]
ziarno = {ziarno}
pokolenia = {pokolenia}
kroki = 30
[plansza]
H = 4
W = 4
[populacja]
N = 8
[zasoby]
c_min = 1.0
c_max = 2.0
alfa = {alfa}
[obserwacje]
skala = 2.0
[koszty]
m = 0.5
ruch = 0.0
s = 1.0
c_k = 0.5
p = 0.5
x = 2.0
[siec]
tryb = "softmax"
tau = 1.0
h = 0
[ewolucja]
T = 5.0
E0 = 10.0
r = 0.1
sigma = 0.1
w_max = 4.0
R = 1
[zapis]
katalog = "{katalog}"
"""


def _maly_pilotaz(tmp_path):
    """Symuluje P1 (1 pokolenie) i dwie wartości α w P2 (po 3 pokolenia)."""
    wyniki = tmp_path / "wyniki"
    for nazwa, pokolenia, alfa in (
        ("P1_z1", 1, 1.0),
        ("P2_niedobor_z1", 3, 0.85),
        ("P2_obfitosc_z1", 3, 3.0),
    ):
        plik = tmp_path / f"{nazwa}.toml"
        plik.write_text(
            KONFIGURACJA.format(
                ziarno=1, pokolenia=pokolenia, alfa=alfa, katalog=(wyniki / nazwa).as_posix()
            ),
            encoding="utf-8",
        )
        assert uruchom_symulacje(["przebieg", str(plik)]) == 0
    return wyniki


def test_udzialy_akcji_krok0_sumuja_sie_do_1(tmp_path):
    wyniki = _maly_pilotaz(tmp_path)
    wynik = eksploracja.udzialy_akcji_krok0(wyniki, ["P1_z1"])
    assert len(wynik) == 8
    assert wynik.sum() == pytest.approx(1.0)
    assert (wynik > 0).all()  # softmax – każda akcja > 0


def test_main_tworzy_tabele_i_wykresy(tmp_path):
    wyniki = _maly_pilotaz(tmp_path)

    wyjscie = tmp_path / "analiza"
    assert eksploracja.main([str(wyniki), "--wyjscie", str(wyjscie), "--D-min", "2"]) == 0

    oczekiwane = {
        "tabele",
        "D_podsumowanie.csv",
        "M_rozrzut_wg_D.csv",
        "histogram_D.pdf",
        "M_od_D.pdf",
        "statystyki_P1.csv",
        "statystyki_P2_alfa_0.85.csv",
        "statystyki_P2_alfa_3.0.csv",
        "statystyki_P2_pierwsze_ostatnie.csv",
        "udzialy_akcji_P1.csv",
        "pokolenie_zerowe.pdf",
        "przebieg_w_pokoleniach.csv",
        "przebieg_wybrane_pokolenia.csv",
        "akcje_w_pokoleniach.csv",
        "przezywalnosc.pdf",
        "M_pop.pdf",
        "akcje_w_pokoleniach.pdf",
    }
    assert {plik.name for plik in wyjscie.iterdir()} == oczekiwane
    tabele = {plik.name for plik in (wyjscie / "tabele").iterdir()}
    assert tabele == {
        "tab_dylematy.tex",
        "tab_rozrzut.tex",
        "tab_statystyki_P1.tex",
        "tab_statystyki_P2.tex",
    }
    for plik in (wyjscie / "tabele").iterdir():
        tekst = plik.read_text(encoding="utf-8")
        assert tekst.count("{") == tekst.count("}")
        assert re.search(r"\bnan\b", tekst, re.IGNORECASE) is None  # braki danych jako „--”
