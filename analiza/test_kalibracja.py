"""Testy analiza/kalibracja.py – wartości policzone ręcznie z definicji (seria K1)."""
import numpy as np
import pandas as pd
import pytest

from analiza import kalibracja
from moralnosc.__main__ import main as moralnosc_main


def test_entropia_akcji_rownomierna_i_jedna_akcja():
    pokolenia = pd.DataFrame({f"akcja_{k}": [10, 0, 0] for k in range(8)})
    pokolenia.loc[1, "akcja_4"] = 300               # jedna akcja → 0 bitów
    pokolenia.loc[2, ["akcja_4", "akcja_7"]] = 50   # dwie po równo → 1 bit

    H = kalibracja.entropia_akcji(pokolenia)

    np.testing.assert_allclose(H.to_numpy(), [3.0, 0.0, 1.0])


def test_entropia_nie_zmienia_wejscia():
    pokolenia = pd.DataFrame({**{f"akcja_{k}": [k, 1] for k in range(8)}, "pokolenie": [0, 1]})
    kopia = pokolenia.copy()

    kalibracja.entropia_akcji(pokolenia)

    pd.testing.assert_frame_equal(pokolenia, kopia)


def _akcje(**liczniki) -> dict[str, int]:
    return {f"akcja_{k}": liczniki.get(f"a{k}", 0) for k in range(8)}


def test_podsumuj_K1_policzone_recznie():
    # dwa ziarna jednego wariantu: A przeżywa 2 pokolenia, B kończy się po pierwszym
    pokolenia = pd.DataFrame([
        {"przebieg": "A", "pokolenie": 0, "zywi_koniec": 3, **_akcje(**{f"a{k}": 10 for k in range(8)})},
        {"przebieg": "A", "pokolenie": 1, "zywi_koniec": 5, **_akcje(a4=80)},
        {"przebieg": "B", "pokolenie": 0, "zywi_koniec": 0, **_akcje(a4=40, a7=40)},
    ]).assign(skala=5.0, tau=1.0, pokolenia_plan=2)
    agenci = pd.DataFrame({
        "przebieg": ["A", "A", "A", "B", "B"], "pokolenie": 0, "id": [1, 2, 3, 1, 2], "D": [5, 7, 2, 5, 9],
    })
    # dylematy pokolenia 0 (tyle wierszy, ile D): A1 – 3 z 5 to akcja 4 (0,6); A2 – 6 z 7 (6/7);
    # A3 – 2 decyzje (< D_min, pomijany); B1 – 5 z 5 (1,0); B2 – 9 z 9 (1,0)
    wiersze = (
        [("A", 1, 4)] * 3 + [("A", 1, 6)] * 2 + [("A", 2, 4)] * 6 + [("A", 2, 7)]
        + [("A", 3, 5)] * 2 + [("B", 1, 5)] * 5 + [("B", 2, 7)] * 9
    )
    dylematy = pd.DataFrame(wiersze, columns=["przebieg", "id", "akcja"])

    wynik = kalibracja.podsumuj_K1(pokolenia, agenci, dylematy, D_min=5, ostatnie=1)

    assert len(wynik) == 1
    w = wynik.iloc[0]
    kon_A, kon_B = (0.6 + 6 / 7) / 2, 1.0
    assert w["ziarna"] == 2
    assert w["przezycie"] == pytest.approx(0.5)
    assert w["koncentracja_p0"] == pytest.approx((kon_A + kon_B) / 2)
    assert w["koncentracja_p0_sd"] == pytest.approx(np.std([kon_A, kon_B], ddof=1))
    assert w["entropia_p0"] == pytest.approx(2.0)        # A: 3 bity, B: 1 bit
    assert w["entropia_koniec"] == pytest.approx(0.5)    # ostatnie pokolenie: A 0, B 1
    assert w["zbieranie_koniec"] == pytest.approx(0.75)  # A 80/80, B 40/80
    assert w["udzial_D_min"] == pytest.approx((2 / 3 + 1) / 2)
    assert w["mediana_D_p0"] == pytest.approx(6.0)       # A: mediana(5, 7, 2) = 5, B: 7
    assert not w["dopuszczalny"]                         # przeżycie 1/2 < 2/3


BAZOWA = """
[przebieg]
ziarno = 1
pokolenia = 3
kroki = 20
[plansza]
H = 8
W = 8
[populacja]
N = 10
[zasoby]
c_min = 1.0
c_max = 5.0
alfa = 1.0
[obserwacje]
skala = 5.0
[koszty]
m = 0.1
ruch = 0.0
s = 2.0
c_k = 1.0
p = 0.5
x = 5.0
[siec]
tryb = "softmax"
tau = 1.0
h = 8
[ewolucja]
T = 0.0
E0 = 50.0
r = "auto"
sigma = 0.2
w_max = 4.0
R = 2
[zapis]
katalog = "wyniki/nieuzywany"
"""


def test_caly_skrypt_na_malej_serii(tmp_path):
    bazowa = tmp_path / "bazowa.toml"
    bazowa.write_text(BAZOWA, encoding="utf-8")
    katalog_serii = tmp_path / "wyniki" / "T1"
    definicja = tmp_path / "T1.toml"
    definicja.write_text(
        f"""
[seria]
nazwa = "T1"
bazowa = "{bazowa.as_posix()}"
katalog = "{katalog_serii.as_posix()}"
ziarna = [1, 2]

[[warianty]]
nazwa = "t0.5"
"siec.tau" = 0.5

[[warianty]]
nazwa = "t2"
"siec.tau" = 2.0
""",
        encoding="utf-8",
    )
    assert moralnosc_main(["seria", str(definicja)]) == 0

    wyjscie = tmp_path / "analiza"
    assert kalibracja.main(["K1", str(katalog_serii), "--wyjscie", str(wyjscie)]) == 0

    for plik in ("K1_podsumowanie.csv", "K1_podsumowanie.tex", "K1_koncentracja.pdf", "K1_entropia.pdf"):
        assert (wyjscie / plik).exists(), plik
    tabela = pd.read_csv(wyjscie / "K1_podsumowanie.csv")
    assert len(tabela) == 2
    assert tabela["ziarna"].tolist() == [2, 2]
    assert "tab:K1" in (wyjscie / "K1_podsumowanie.tex").read_text(encoding="utf-8")
