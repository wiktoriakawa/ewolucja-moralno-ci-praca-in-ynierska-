"""Testy skryptu eksploracji (analiza/eksploracja.py).

Wartości oczekiwane są policzone ręcznie z definicji w Lab02 (punkt 3.4.3).
Ostatni test uruchamia cały skrypt na małym, właśnie zasymulowanym pilotażu.
Uruchomienie z katalogu kod/:
    PYTHONPATH=src python -m pytest analiza -q
"""
import numpy as np
import pandas as pd
import pytest

import eksploracja
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
    # Drugie pokolenie bez żadnego dylematu.
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
    # kwartyle z interpolacją liniową (domyślna w pandas) dla 0, 1, 4, 10
    assert wszyscy[["Q1", "mediana", "Q3"]].tolist() == pytest.approx([0.75, 2.5, 5.5])
    assert wynik.loc["zmarli", "mediana"] == pytest.approx(0.5)
    assert wynik.loc["dożyli końca", "odsetek_ponizej_Dmin"] == pytest.approx(0.5)


def test_rozrzut_M_wg_D_szum():
    # Jeden agent z jednym dzieleniem, drugi z jedną kradzieżą: p_s = p_k = 0,5,
    # więc wariancja jednej decyzji = (0,5 + 0,5) − 0² = 1, a szum przy D = 1 to 1.
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


def test_etykieta_alfa():
    assert eksploracja._etykieta_alfa(0.85) == "α = 0,85"
    assert eksploracja._etykieta_alfa(1.0) == "α = 1,0"


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


def test_main_tworzy_tabele_i_wykresy(tmp_path):
    """Mały pilotaż (P1 i dwie wartości α w P2) → cały skrypt → wszystkie pliki."""
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

    wyjscie = tmp_path / "analiza"
    assert eksploracja.main([str(wyniki), "--wyjscie", str(wyjscie), "--D-min", "2"]) == 0

    oczekiwane = {
        "D_podsumowanie.csv",
        "M_rozrzut_wg_D.csv",
        "histogram_D.pdf",
        "M_od_D.pdf",
        "statystyki_P1.csv",
        "statystyki_P2_alfa_0.85.csv",
        "statystyki_P2_alfa_3.0.csv",
        "udzialy_akcji_P1.csv",
        "pokolenie_zerowe.pdf",
        "przebieg_w_pokoleniach.csv",
        "przebieg_wybrane_pokolenia.csv",
        "przezywalnosc.pdf",
        "M_pop.pdf",
    }
    assert {plik.name for plik in wyjscie.iterdir()} == oczekiwane
