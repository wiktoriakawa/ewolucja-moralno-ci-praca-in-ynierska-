"""Eksploracja danych pilotażu Lab02 (punkty 4.3.1 i 4.4).

Liczy współczynniki z 3.4.3 (profil, M, M_wyg, M_pop) z zapisanych CSV, wypisuje
liczby, zapisuje wykresy i tabele LaTeX-a (\\input{tabele/<nazwa>}). Symulacji
nie uruchamia – poza odtworzeniem pokolenia zerowego w 4.4.3.

Z katalogu kod/:
    PYTHONPATH=src python analiza/eksploracja.py ../asystent/Lab02/wyniki

Wyniki w --wyjscie (domyślnie wyniki/analiza):
    4.3.1  D_podsumowanie.csv, M_rozrzut_wg_D.csv, histogram_D.pdf, M_od_D.pdf,
           tabele/tab_dylematy.tex, tabele/tab_rozrzut.tex
    4.4.2  statystyki_P1.csv, statystyki_P2_*.csv, tabele/tab_statystyki_P1.tex, P2.tex
    4.4.3  udzialy_akcji_P1.csv, pokolenie_zerowe.pdf
    4.4.4  przebieg_w_pokoleniach.csv, przebieg_wybrane_pokolenia.csv,
           akcje_w_pokoleniach.csv, przezywalnosc.pdf, M_pop.pdf, akcje_w_pokoleniach.pdf

P3 pomijamy (to ten sam przebieg dwa razy – tylko powtarzalność).
P1 też ma α = 1,0, więc przed groupby("alfa") zawsze wybieramy seria == "P2".
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moralnosc.ewolucja import pokolenie_zerowe
from moralnosc.populacja import obserwacje
from moralnosc.siec import wyjscia
from moralnosc.stan import DZIEL, KRADNIJ, LICZBA_AKCJI, NIC, ZBIERZ

# kody z pomiar.py (agenci.csv, przyczyna_smierci)
PRZYCZYNA_WYCZERPANIE = 1
PRZYCZYNA_KONIEC_POKOLENIA = 2

NAZWY_AKCJI = {
    0: "ruch N",
    1: "ruch E",
    2: "ruch S",
    3: "ruch W",
    ZBIERZ: "zbierz",
    DZIEL: "dziel",
    KRADNIJ: "kradnij",
    NIC: "nic",
}
KOLUMNY_AKCJI = [f"akcja_{kod}" for kod in range(LICZBA_AKCJI)]

# cztery kierunki ruchu jako jedna grupa
GRUPY_AKCJI = {
    "ruch": [0, 1, 2, 3],
    "zbierz": [ZBIERZ],
    "dziel": [DZIEL],
    "kradnij": [KRADNIJ],
    "nic": [NIC],
}

# Dolne granice przedziałów D – rosną mniej więcej geometrycznie, bo rozkład
# jest bardzo skośny (mediana w P1 = 2, maks. 300).
GRANICE_D = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200)

ETYKIETY_TEX = {
    "D (agenci)": "$D$ (agenci)",
    "S (agenci)": "$S$ (agenci)",
    "K (agenci)": "$K$ (agenci)",
    "N_o (agenci)": "$N_o$ (agenci)",
    "M (agenci z D > 0)": "$M$ (agenci z $D > 0$)",
    "M_wyg (agenci)": "$M_{\\mathrm{wyg}}$ (agenci)",
    "zasoby na koniec (agenci)": "Zasoby na koniec (agenci)",
    "krok śmierci (zmarli agenci)": "Krok śmierci (zmarli agenci)",
    "przeżywalność (pokolenia)": "Przeżywalność (pokolenia)",
    "zakwalifikowani (pokolenia)": "Zakwalifikowani (pokolenia)",
    "M_pop (pokolenia z ΣD > 0)": "$M_{\\mathrm{pop}}$ (pokolenia z $\\sum D > 0$)",
}


# --- 1. Wczytywanie -------------------------------------------------------------------


def _wczytaj_konfiguracje(katalog: Path) -> dict:
    """Konfiguracja z konfiguracja.json (bez metadanych)."""
    with (katalog / "konfiguracja.json").open(encoding="utf-8") as plik:
        return json.load(plik)["konfiguracja"]


def _dopisz_opis_przebiegu(tabela: pd.DataFrame, katalog: Path, cfg: dict) -> None:
    """Wstawia na początek kolumny przebieg, seria, alfa, ziarno (w miejscu)."""
    opis = {
        "przebieg": katalog.name,
        "seria": katalog.name.split("_")[0],
        "alfa": cfg["zasoby"]["alfa"],
        "ziarno": cfg["przebieg"]["ziarno"],
    }
    for pozycja, (nazwa, wartosc) in enumerate(opis.items()):
        tabela.insert(pozycja, nazwa, wartosc)


def wczytaj_przebieg(katalog: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(pokolenia, agenci, cfg) jednego przebiegu.

    dylematy.csv pomijamy – duży, potrzebny tylko w 4.4.3 (wczytaj_dylematy).
    """
    pokolenia = pd.read_csv(katalog / "pokolenia.csv")
    agenci = pd.read_csv(katalog / "agenci.csv")
    cfg = _wczytaj_konfiguracje(katalog)
    return pokolenia, agenci, cfg


def wczytaj_pilotaz(
    katalog_wynikow: Path, serie: tuple[str, ...] = ("P1", "P2")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pokolenia, agenci) wszystkich przebiegów z wybranych serii, z kolumnami opisu.

    Kontrola: P1 + P2 = 14 przebiegów, 83 200 wierszy agentów, 416 pokoleń.
    """
    wszystkie_pokolenia = []
    wszyscy_agenci = []
    # sorted – stała kolejność niezależnie od systemu plików
    for katalog in sorted(katalog_wynikow.iterdir()):
        # is_dir pomija np. .DS_Store
        if not katalog.is_dir() or katalog.name.split("_")[0] not in serie:
            continue
        pokolenia, agenci, cfg = wczytaj_przebieg(katalog)
        _dopisz_opis_przebiegu(pokolenia, katalog, cfg)
        _dopisz_opis_przebiegu(agenci, katalog, cfg)
        wszystkie_pokolenia.append(pokolenia)
        wszyscy_agenci.append(agenci)

    if not wszyscy_agenci:
        raise FileNotFoundError(f"Brak przebiegów serii {serie} w {katalog_wynikow}")

    return (
        pd.concat(wszystkie_pokolenia, ignore_index=True),
        pd.concat(wszyscy_agenci, ignore_index=True),
    )


def wczytaj_dylematy(katalog_wynikow: Path, przebiegi: list[str]) -> pd.DataFrame:
    """dylematy.csv wskazanych przebiegów z kolumnami opisu (używane tylko dla P1)."""
    if not przebiegi:
        raise ValueError("Lista przebiegów do wczytania dylematów jest pusta")
    tabele = []
    for nazwa in przebiegi:
        katalog = katalog_wynikow / nazwa
        dylematy = pd.read_csv(katalog / "dylematy.csv")
        _dopisz_opis_przebiegu(dylematy, katalog, _wczytaj_konfiguracje(katalog))
        tabele.append(dylematy)
    return pd.concat(tabele, ignore_index=True)


# --- 2. Współczynniki (punkt 3.4.3) ----------------------------------------------------


def dodaj_wspolczynniki(agenci: pd.DataFrame, a: float) -> pd.DataFrame:
    """Kopia tabeli z kolumnami: s, k, n (profil), M, M_wyg, s_wyg, k_wyg, n_wyg.

    Przy D = 0 profil i M to NaN (brak danych, nie zachowanie neutralne);
    wersje wygładzone są określone zawsze.
    """
    assert a > 0, f"Parametr wygładzania a musi być dodatni, a jest {a}"
    wynik = agenci.copy()

    # D = 0 → NaN w mianowniku → NaN w wyniku
    D_dodatnie = wynik["D"].where(wynik["D"] > 0)
    wynik["s"] = wynik["S"] / D_dodatnie
    wynik["k"] = wynik["K"] / D_dodatnie
    wynik["n"] = wynik["N_o"] / D_dodatnie
    wynik["M"] = (wynik["S"] - wynik["K"]) / D_dodatnie

    wynik["M_wyg"] = (wynik["S"] - wynik["K"]) / (wynik["D"] + a)
    wynik["s_wyg"] = (wynik["S"] + 1) / (wynik["D"] + 3)
    wynik["k_wyg"] = (wynik["K"] + 1) / (wynik["D"] + 3)
    wynik["n_wyg"] = (wynik["N_o"] + 1) / (wynik["D"] + 3)
    return wynik


def m_pop(agenci: pd.DataFrame) -> pd.DataFrame:
    """M_pop = (ΣS − ΣK) / ΣD na pokolenie przebiegu; ΣD = 0 → NaN."""
    sumy = agenci.groupby(["przebieg", "pokolenie"], as_index=False)[["S", "K", "D"]].sum()
    sumy["M_pop"] = (sumy["S"] - sumy["K"]) / sumy["D"].where(sumy["D"] > 0)
    return sumy[["przebieg", "pokolenie", "M_pop"]]


# --- 3. Liczba dylematów (punkt 4.3.1) ------------------------------------------------


def _przedzialy_D(D: pd.Series, D_min: int | None = None, od: int = 0) -> list[int]:
    """Granice przedziałów D: GRANICE_D od `od`, plus D_min i górna granica z danych."""
    granice = {g for g in GRANICE_D if g >= od}
    if D_min is not None:
        granice.add(D_min)  # żaden przedział nie przecina progu
    gorna = max(int(D.max()) + 1 if len(D) else 0, max(granice) + 1)
    return sorted(granice) + [gorna]


def _etykiety_przedzialow(granice: list[int]) -> list[str]:
    """Etykiety [dolna, górna): „0”, „3–4”, „200–300”."""
    etykiety = []
    for dolna, gorna in zip(granice[:-1], granice[1:]):
        etykiety.append(f"{dolna}" if gorna - dolna == 1 else f"{dolna}–{gorna - 1}")
    return etykiety


def podsumuj_D(agenci: pd.DataFrame, D_min: int) -> pd.DataFrame:
    """Rozkład D w grupie: wiersze wszyscy / zmarli / dożyli końca;
    kolumny liczba, odsetek_D0, odsetek_ponizej_Dmin, mediana, Q1, Q3.
    """
    grupy = {
        "wszyscy": agenci,
        "zmarli": agenci[agenci["przyczyna_smierci"] == PRZYCZYNA_WYCZERPANIE],
        "dożyli końca": agenci[agenci["przyczyna_smierci"] == PRZYCZYNA_KONIEC_POKOLENIA],
    }
    wiersze = {}
    for nazwa, grupa in grupy.items():
        D = grupa["D"]
        wiersze[nazwa] = {
            "liczba": len(D),
            "odsetek_D0": (D == 0).mean(),   # średnia z bool = odsetek True
            "odsetek_ponizej_Dmin": (D < D_min).mean(),
            "mediana": D.median(),
            "Q1": D.quantile(0.25),
            "Q3": D.quantile(0.75),
        }
    return pd.DataFrame.from_dict(wiersze, orient="index")


def rozrzut_M_wg_D(
    agenci: pd.DataFrame, D_min: int, granice: list[int] | None = None
) -> pd.DataFrame:
    """Rozrzut M w przedziałach D wobec rozrzutu z samego przypadku (podstawa D_min i a).

    Decyzja x ∈ {+1, −1, 0}; przy wspólnych p_s, p_k w grupie:
        Var(x) = (p_s + p_k) − (p_s − p_k)²,   odch. std. M = √(Var(x) / D)  – „szum”.
    D_min tam, gdzie odch_std_M wyraźnie przewyższa odch_std_szum.
    Wymaga dodaj_wspolczynniki; agenci z D = 0 pomijani.
    """
    z_dylematami = agenci[agenci["D"] > 0]
    p_s = z_dylematami["S"].sum() / z_dylematami["D"].sum()
    p_k = z_dylematami["K"].sum() / z_dylematami["D"].sum()
    wariancja_decyzji = (p_s + p_k) - (p_s - p_k) ** 2

    if granice is None:
        granice = _przedzialy_D(z_dylematami["D"], D_min, od=1)
    przedzial = pd.cut(
        z_dylematami["D"], granice, right=False, labels=_etykiety_przedzialow(granice)
    )
    tabela = (
        z_dylematami.assign(przedzial=przedzial, wariancja_szumu=wariancja_decyzji / z_dylematami["D"])
        .groupby("przedzial", observed=True)
        .agg(
            liczba=("M", "size"),
            odch_std_M=("M", "std"),
            wariancja_szumu=("wariancja_szumu", "mean"),
            odsetek_M0=("M", lambda M: (M == 0).mean()),
            odsetek_skrajnych=("M", lambda M: (M.abs() == 1).mean()),
        )
    )
    # najpierw średnia wariancji, potem pierwiastek
    tabela.insert(2, "odch_std_szum", np.sqrt(tabela.pop("wariancja_szumu")))
    return tabela


def _z_przecinkiem(wartosc: float, _pozycja: int) -> str:
    """Etykieta osi z przecinkiem dziesiętnym i znakiem −."""
    return f"{wartosc:g}".replace(".", ",").replace("-", "−")


def _przecinki(os_: plt.Axes, *, x: bool = False) -> None:
    """Przecinek dziesiętny na osi y (i x, gdy x=True)."""
    os_.yaxis.set_major_formatter(_z_przecinkiem)
    if x:
        os_.xaxis.set_major_formatter(_z_przecinkiem)


def _zapisz_wykres(fig: plt.Figure, plik: Path) -> None:
    """Zapisuje i zamyka wykres; PDF bez daty w metadanych – ten sam plik za każdym razem."""
    metadane = {"CreationDate": None} if plik.suffix == ".pdf" else None
    fig.savefig(plik, bbox_inches="tight", dpi=300, metadata=metadane)
    plt.close(fig)


def wykres_histogram_D(grupy: dict[str, pd.Series], D_min: int, plik: Path) -> None:
    """Udział agentów w przedziałach D (GRANICE_D), panel na grupę; poniżej D_min – szare."""
    wszystkie_D = pd.concat(list(grupy.values()))
    granice = _przedzialy_D(wszystkie_D, D_min)
    etykiety = _etykiety_przedzialow(granice)
    kolory = ["0.65" if gorna <= D_min else "C0" for gorna in granice[1:]]

    kolumny = min(len(grupy), 2)
    wiersze = int(np.ceil(len(grupy) / kolumny))
    fig, osie = plt.subplots(
        wiersze, kolumny, figsize=(6.3, 2.5 * wiersze), sharey=True, squeeze=False
    )
    for os_, (tytul, D) in zip(osie.flat, grupy.items()):
        udzialy = (
            pd.cut(D, granice, right=False, labels=etykiety)
            .value_counts(normalize=True, sort=False)
            .reindex(etykiety)
        )
        os_.bar(range(len(etykiety)), udzialy.to_numpy(), color=kolory)
        os_.set_xticks(range(len(etykiety)), etykiety, rotation=90, fontsize=8)
        os_.set_title(f"{tytul} (n = {len(D)})", fontsize=9)
        _przecinki(os_)
    for os_ in osie[-1]:
        os_.set_xlabel("Liczba dylematów D")
    for os_ in osie[:, 0]:
        os_.set_ylabel("Udział agentów")
    for os_ in list(osie.flat)[len(grupy):]:
        os_.set_visible(False)  # pusty panel przy nieparzystej liczbie grup
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


def wykres_M_od_D(agenci: pd.DataFrame, D_min: int, plik: Path) -> None:
    """M i M_wyg od D (skala log.), D < D_min wyróżnione; widać, jak wygładzanie ściąga „wachlarz”.

    rasterized=True – inaczej PDF z kilkudziesięciu tysięcy punktów ma kilkanaście MB.
    """
    z_dylematami = agenci[agenci["D"] > 0]
    ponizej = z_dylematami["D"] < D_min
    fig, osie = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for os_, kolumna, tytul in zip(osie, ("M", "M_wyg"), (r"$M$", r"$M_{\mathrm{wyg}}$")):
        for maska, kolor, etykieta in (
            (~ponizej, "C0", f"D ≥ {D_min}"),
            (ponizej, "C3", f"D < {D_min}"),
        ):
            os_.scatter(
                z_dylematami.loc[maska, "D"],
                z_dylematami.loc[maska, kolumna],
                s=3,
                alpha=0.15,
                color=kolor,
                label=etykieta,
                rasterized=True,
            )
        os_.axvline(D_min, color="0.3", linestyle="--", linewidth=0.8)
        os_.set_xscale("log")
        os_.set_xlabel("Liczba dylematów D (skala log.)")
        os_.set_title(tytul)
        _przecinki(os_, x=True)  # 1, 10, 100 zamiast 10⁰, 10¹, 10²
    osie[0].set_ylabel("Wartość współczynnika")
    legenda = osie[0].legend(loc="lower right", fontsize=8, markerscale=3)
    for uchwyt in legenda.legend_handles:
        uchwyt.set_alpha(1)
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


# --- 4. Statystyki opisowe (punkt 4.4.2) ----------------------------------------------


def opisz(seria: pd.Series) -> dict[str, float]:
    """liczba, średnia, mediana, odch. std. (n − 1), min, maks; NaN pomijane."""
    return {
        "liczba": int(seria.count()),
        "srednia": seria.mean(),
        "mediana": seria.median(),
        "odch_std": seria.std(),
        "min": seria.min(),
        "maks": seria.max(),
    }


def statystyki_opisowe(agenci: pd.DataFrame, pokolenia: pd.DataFrame) -> pd.DataFrame:
    """Tabela statystyk 4.4.2 dla jednej grupy (agenci po dodaj_wspolczynniki, pokolenia z M_pop).

    Zmienne agentów – wszyscy agenci; wyjątki: M (tylko D > 0), krok śmierci
    (tylko zmarli z wyczerpania).
    """
    zmarli = agenci["przyczyna_smierci"] == PRZYCZYNA_WYCZERPANIE
    zmienne = {
        "D (agenci)": agenci["D"],
        "S (agenci)": agenci["S"],
        "K (agenci)": agenci["K"],
        "N_o (agenci)": agenci["N_o"],
        "M (agenci z D > 0)": agenci["M"],
        "M_wyg (agenci)": agenci["M_wyg"],
        "zasoby na koniec (agenci)": agenci["zasoby_koniec"],
        "krok śmierci (zmarli agenci)": agenci.loc[zmarli, "krok_smierci"],
        "przeżywalność (pokolenia)": pokolenia["przezywalnosc"],
        "zakwalifikowani (pokolenia)": pokolenia["zakwalifikowani"],
        "M_pop (pokolenia z ΣD > 0)": pokolenia["M_pop"],
    }
    return pd.DataFrame.from_dict(
        {nazwa: opisz(seria) for nazwa, seria in zmienne.items()}, orient="index"
    )


def statystyki_pierwsze_ostatnie(agenci: pd.DataFrame, pokolenia: pd.DataFrame) -> pd.DataFrame:
    """Średnie ze statystyki_opisowe w pierwszym i ostatnim pokoleniu każdego α (P2).

    Kolumny (alfa, pokolenie). Wszystkie pokolenia łącznie mieszałyby wąskie
    gardło pokolenia zerowego z populacją przystosowaną.
    """
    kolumny = {}
    for alfa, grupa in agenci.groupby("alfa"):
        for numer in (grupa["pokolenie"].min(), grupa["pokolenie"].max()):
            w_pokoleniu = grupa[grupa["pokolenie"] == numer]
            pokolenia_alfa = pokolenia[(pokolenia["alfa"] == alfa) & (pokolenia["pokolenie"] == numer)]
            kolumny[(alfa, int(numer))] = statystyki_opisowe(w_pokoleniu, pokolenia_alfa)["srednia"]
    return pd.DataFrame(kolumny)


# --- 5. Pokolenie zerowe (punkt 4.4.3) ------------------------------------------------


def udzialy_akcji(pokolenia: pd.DataFrame) -> pd.Series:
    """Udział każdej akcji we wszystkich decyzjach (akcja_0 … akcja_7, suma = 1).

    Ważone długością życia – porównaj z udzialy_akcji_krok0.
    """
    sumy = pokolenia[KOLUMNY_AKCJI].sum()
    udzialy = sumy / sumy.sum()
    udzialy.index = [NAZWY_AKCJI[kod] for kod in range(LICZBA_AKCJI)]
    udzialy.name = "udzial"
    return udzialy


def udzialy_akcji_krok0(katalog_wynikow: Path, przebiegi: list[str]) -> pd.Series:
    """Oczekiwany udział akcji w kroku 0 pokolenia zerowego (bez losowania akcji).

    Odtwarza pokolenie zerowe z konfiguracji i ziarna, liczy prawdopodobieństwa
    jak siec.wybierz_akcje. Różnica wobec udzialy_akcji = wpływ przeżywalności.
    """
    sumy = np.zeros(LICZBA_AKCJI)
    for nazwa in przebiegi:
        cfg = _wczytaj_konfiguracje(katalog_wynikow / nazwa)
        # dane Lab02 nie mają siec.h – sieć liniowa
        cfg.setdefault("siec", {}).setdefault("h", 0)
        stan = pokolenie_zerowe(cfg, np.random.default_rng(cfg["przebieg"]["ziarno"]))
        obs, _ = obserwacje(stan, cfg["obserwacje"]["skala"])
        y = wyjscia(obs, stan.genomy, cfg["siec"]["h"])
        if cfg["siec"]["tryb"] == "softmax":
            # odjęcie maksimum chroni exp przed przepełnieniem
            z = (y - y.max(axis=1, keepdims=True)) / cfg["siec"]["tau"]
            prawdopodobienstwa = np.exp(z)
            prawdopodobienstwa /= prawdopodobienstwa.sum(axis=1, keepdims=True)
        else:
            prawdopodobienstwa = np.eye(LICZBA_AKCJI)[y.argmax(axis=1)]
        sumy += prawdopodobienstwa.sum(axis=0)
    udzialy = pd.Series(sumy / sumy.sum(), index=[NAZWY_AKCJI[k] for k in range(LICZBA_AKCJI)])
    udzialy.name = "udzial_krok0"
    return udzialy


def najczestsza_akcja_w_dylematach(
    dylematy: pd.DataFrame, min_decyzji: int = 1
) -> pd.Series:
    """Udział najczęstszej akcji agenta w jego dylematach (1 = zawsze to samo).

    Czy softmax daje zróżnicowane decyzje? Tylko decyzje z dylematy.csv –
    akcji poza dylematami CSV nie zawiera. min_decyzji odsiewa agentów z 1–2 decyzjami.
    """
    agent = ["przebieg", "id"]
    udzialy = dylematy.groupby(agent)["akcja"].value_counts(normalize=True)
    najczestsza = udzialy.groupby(level=agent).max()
    liczba_decyzji = dylematy.groupby(agent).size()
    return najczestsza[liczba_decyzji >= min_decyzji]


def wykres_pokolenie_zerowe(
    udzialy: pd.Series,
    agenci_P1: pd.DataFrame,
    plik: Path,
    udzialy_krok0: pd.Series | None = None,
) -> None:
    """Lewy panel: udziały akcji (całe pokolenie i krok 0); prawy: histogram M agentów P1 z D > 0.

    21 przedziałów na [−1, 1] – M = 0 ma własny przedział.
    """
    M = agenci_P1["M"].dropna()
    fig, (lewy, prawy) = plt.subplots(
        1, 2, figsize=(6.3, 2.8), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    x = np.arange(len(udzialy))
    if udzialy_krok0 is None:
        lewy.bar(x, udzialy.to_numpy(), color="C0")
    else:
        lewy.bar(x - 0.2, udzialy_krok0.to_numpy(), width=0.4, color="0.65", label="krok 0 (oczekiwane)")
        lewy.bar(x + 0.2, udzialy.to_numpy(), width=0.4, color="C0", label="całe pokolenie")
        lewy.legend(fontsize=7, loc="upper left")
    lewy.set_xticks(x, udzialy.index, rotation=90)
    lewy.set_ylabel("Udział decyzji")
    lewy.set_title("Wybrane akcje", fontsize=9)
    _przecinki(lewy)

    prawy.hist(M, bins=21, range=(-1, 1), color="C0", edgecolor="white")
    prawy.set_xlabel(r"$M$")
    prawy.set_ylabel("Liczba agentów")
    prawy.set_title(f"$M$ agentów z D > 0 (n = {len(M)})", fontsize=9)
    _przecinki(prawy, x=True)
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


# --- 6. Kolejne pokolenia dla trzech α (punkt 4.4.4) ----------------------------------


def _etykieta_alfa(alfa: float) -> str:
    """„α = 0,85”, „α = 1,0”."""
    tekst = f"{alfa:.2f}"
    if tekst.endswith("0"):
        tekst = tekst[:-1]  # 1.00 → 1.0
    return "α = " + tekst.replace(".", ",")


def _etykieta_alfa_tex(alfa: float) -> str:
    """$\\alpha = 0{,}85$."""
    liczba = _etykieta_alfa(alfa).removeprefix("α = ").replace(",", "{,}")
    return f"$\\alpha = {liczba}$"


def przebieg_w_pokoleniach(pokolenia: pd.DataFrame) -> pd.DataFrame:
    """Na (alfa, pokolenie): liczba_ziaren, średnie przeżywalności, zakwalifikowanych
    i M_pop po ziarnach, awaryjny_liczba.

    Po wymarciu ziarna średnie są z mniejszej liczby ziaren (liczba_ziaren).
    """
    return pokolenia.groupby(["alfa", "pokolenie"], as_index=False).agg(
        liczba_ziaren=("przebieg", "nunique"),
        przezywalnosc_sr=("przezywalnosc", "mean"),
        zakwalifikowani_sr=("zakwalifikowani", "mean"),
        M_pop_sr=("M_pop", "mean"),
        awaryjny_liczba=("awaryjny", "sum"),
    )


def udzialy_akcji_w_pokoleniach(pokolenia: pd.DataFrame) -> pd.DataFrame:
    """Udział grup akcji na (alfa, pokolenie); najpierw suma po ziarnach, potem dzielenie."""
    sumy = pokolenia.groupby(["alfa", "pokolenie"])[KOLUMNY_AKCJI].sum()
    grupy = pd.DataFrame(
        {
            nazwa: sumy[[KOLUMNY_AKCJI[kod] for kod in kody]].sum(axis=1)
            for nazwa, kody in GRUPY_AKCJI.items()
        }
    )
    return grupy.div(grupy.sum(axis=1), axis=0).reset_index()


def _linie_ziaren_i_srednia(
    os_: plt.Axes, pokolenia: pd.DataFrame, kolumna: str, kolumna_sredniej: str
) -> None:
    """Cienkie linie ziaren, gruba średnia na α, krzyżyk = wymarcie.

    Po wymarciu średnia przerywana – może skoczyć tylko przez odpadnięcie ziarna.
    """
    srednie = przebieg_w_pokoleniach(pokolenia)
    for numer, (alfa, grupa) in enumerate(pokolenia.groupby("alfa")):
        kolor = f"C{numer}"
        for _, ziarno in grupa.groupby("przebieg"):
            ziarno = ziarno.sort_values("pokolenie")
            os_.plot(ziarno["pokolenie"], ziarno[kolumna], color=kolor, linewidth=0.6, alpha=0.4)
            ostatnie = ziarno.iloc[-1]
            if ostatnie["zywi_koniec"] == 0:
                os_.plot(ostatnie["pokolenie"], ostatnie[kolumna], "x", color=kolor, markersize=7)
        srednia = srednie[srednie["alfa"] == alfa]
        # wymarły nie wraca – pełne pokolenia to ciągły początek tabeli
        pelne = srednia["liczba_ziaren"] == srednia["liczba_ziaren"].max()
        os_.plot(
            srednia.loc[pelne, "pokolenie"],
            srednia.loc[pelne, kolumna_sredniej],
            color=kolor,
            linewidth=2,
            label=_etykieta_alfa(alfa),
        )
        if not pelne.all():
            # od ostatniego pełnego pokolenia – żeby linie się łączyły
            poczatek = max(0, int(np.argmin(pelne.to_numpy())) - 1)
            niepelna = srednia.iloc[poczatek:]
            os_.plot(
                niepelna["pokolenie"],
                niepelna[kolumna_sredniej],
                color=kolor,
                linewidth=1.5,
                linestyle="--",
            )


def wykres_przezywalnosc(pokolenia: pd.DataFrame, plik: Path) -> None:
    """Przeżywalność (góra) i zakwalifikowani (dół) w kolejnych pokoleniach P2."""
    fig, (gora, dol) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    _linie_ziaren_i_srednia(gora, pokolenia, "przezywalnosc", "przezywalnosc_sr")
    _linie_ziaren_i_srednia(dol, pokolenia, "zakwalifikowani", "zakwalifikowani_sr")
    gora.set_ylabel("Przeżywalność")
    gora.set_ylim(0, 1.02)
    dol.set_ylabel("Zakwalifikowani")
    dol.set_xlabel("Pokolenie")
    gora.legend(fontsize=8, loc="lower right")
    _przecinki(gora)
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


def wykres_m_pop(
    pokolenia: pd.DataFrame,
    plik: Path,
    poziom_odniesienia: float | None = None,
    zakres_odniesienia: tuple[float, float] | None = None,
) -> None:
    """M_pop w kolejnych pokoleniach P2 na tle P1 (linia – M_pop serii, pas – zakres ziaren).

    Zmiana przypisywalna ewolucji dopiero, gdy wyraźnie wychodzi poza pas.
    """
    fig, os_ = plt.subplots(figsize=(6.3, 2.8))
    if zakres_odniesienia is not None:
        os_.axhspan(*zakres_odniesienia, color="0.88", zorder=0, label="P1: zakres między ziarnami")
    _linie_ziaren_i_srednia(os_, pokolenia, "M_pop", "M_pop_sr")
    if poziom_odniesienia is not None:
        os_.axhline(
            poziom_odniesienia,
            color="0.3",
            linestyle="--",
            linewidth=0.8,
            label="P1 (pokolenie zerowe)",
        )
    os_.set_xlabel("Pokolenie")
    os_.set_ylabel(r"$M_{\mathrm{pop}}$")
    os_.legend(fontsize=8)
    _przecinki(os_)
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


def wykres_akcje_w_pokoleniach(pokolenia: pd.DataFrame, plik: Path) -> None:
    """Udział grup akcji w pokoleniach P2, panel na α.

    Pomaga czytać M_pop: M_pop ≈ 0 przy samym zbieraniu to brak interakcji,
    nie równowaga dzielenia i kradzieży.
    """
    tabela = udzialy_akcji_w_pokoleniach(pokolenia)
    wartosci_alfa = sorted(tabela["alfa"].unique())
    fig, osie = plt.subplots(
        1, len(wartosci_alfa), figsize=(6.3, 2.5), sharey=True, squeeze=False
    )
    for os_, alfa in zip(osie[0], wartosci_alfa):
        dane = tabela[tabela["alfa"] == alfa]
        for nazwa in GRUPY_AKCJI:
            os_.plot(dane["pokolenie"], dane[nazwa], linewidth=1.2, label=nazwa)
        os_.set_title(_etykieta_alfa(alfa), fontsize=9)
        os_.set_xlabel("Pokolenie")
        os_.set_ylim(0, 1)
        _przecinki(os_)
    osie[0][0].set_ylabel("Udział decyzji")
    osie[0][-1].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


# --- 7. Tabele do LaTeX-a ---------------------------------------------------------------


def liczba_tex(wartosc: float, miejsca: int = 2) -> str:
    """Liczba do LaTeX-a: {,} jako przecinek, $-$, całkowite bez ułamka, NaN → „--”."""
    if pd.isna(wartosc):
        return "--"
    v = round(float(wartosc), miejsca)
    if v == 0:
        v = 0.0  # bez −0
    tekst = str(int(v)) if v.is_integer() else f"{v:.{miejsca}f}"
    return tekst.replace("-", "$-$").replace(".", "{,}")


def procent_tex(udzial: float) -> str:
    """0.314 → 31{,}4\\%."""
    if pd.isna(udzial):
        return "--"
    return f"{udzial * 100:.1f}".replace(".", "{,}") + "\\%"


def zapisz_tabele_tex(
    wiersze: list[list[str]],
    plik: Path,
    *,
    naglowek: list[str],
    podpis: str,
    etykieta: str,
    szerokosc_pierwszej: float = 0.30,
    uwaga: str | None = None,
    naglowek_nad: str | None = None,
) -> None:
    """Zapisuje gotową tabelę longtable (komórki już sformatowane; typ P z preambuły).

    Suma szerokości < \\linewidth o odstępy między kolumnami (ok. 2,6% na granicę).
    """
    n = len(naglowek)
    razem = 0.97 - 0.0265 * (n - 1)
    reszta = (razem - szerokosc_pierwszej) / (n - 1)
    kolumny = " ".join(
        [f"P{{{szerokosc_pierwszej:.2f}\\linewidth}}"] + [f"P{{{reszta:.3f}\\linewidth}}"] * (n - 1)
    )
    linia_naglowka = "  " + " & ".join(naglowek) + " \\\\"
    if naglowek_nad:
        linia_naglowka = f"  {naglowek_nad} \\\\\n" + linia_naglowka
    linie = [
        "% Wygenerowano skryptem kod/analiza/eksploracja.py – liczb nie poprawiać ręcznie,",
        "% tylko uruchomić skrypt ponownie. Podpis można przeredagować.",
        "\\begingroup\\small",
        f"\\begin{{longtable}}{{@{{}}{kolumny}@{{}}}}",
        f"  \\caption{{{podpis}}}\\label{{{etykieta}}}\\\\",
        "  \\toprule",
        linia_naglowka,
        "  \\midrule",
        "  \\endfirsthead",
        "  \\toprule",
        linia_naglowka,
        "  \\midrule",
        "  \\endhead",
        *("  " + " & ".join(wiersz) + " \\\\" for wiersz in wiersze),
        "  \\bottomrule",
    ]
    if uwaga:
        linie.append(f"  \\multicolumn{{{n}}}{{@{{}}p{{0.9\\linewidth}}}}{{\\footnotesize {uwaga}}}")
    linie += ["\\end{longtable}", "\\endgroup", ""]
    plik.write_text("\n".join(linie), encoding="utf-8")


# --- 8. Całość ------------------------------------------------------------------------


def utworz_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eksploracja danych z pilotażu Lab02.")
    parser.add_argument("katalog_wynikow", type=Path, help="katalog z przebiegami P1–P3")
    parser.add_argument(
        "--wyjscie",
        type=Path,
        default=Path("wyniki/analiza"),
        help="katalog na wykresy i tabele (domyślnie wyniki/analiza, poza Gitem)",
    )
    parser.add_argument("--a", type=float, default=2.0, help="parametr wygładzania a")
    parser.add_argument("--D-min", dest="D_min", type=int, default=5, help="próg D_min")
    return parser


def _naglowek(tekst: str) -> None:
    print(f"\n=== {tekst} " + "=" * max(0, 70 - len(tekst)))


def _pokaz_i_zapisz(tabela: pd.DataFrame | pd.Series, plik: Path) -> None:
    """Wypisuje (3 miejsca) i zapisuje do CSV (4 miejsca)."""
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(tabela.round(3))
    tabela.to_csv(plik, float_format="%.4f")
    print(f"→ {plik}")


def main(argv: list[str] | None = None) -> int:
    """Cała eksploracja. Dobór danych (do opisania w tekście):

        4.3.1 – D: P1 (pokolenie 0) i ostatnie pokolenie P2 każdego α;
                rozrzut M osobno dla P1 i P2, wspólne przedziały D;
        4.4.2 – P1: pokolenie 0; P2: pierwsze i ostatnie pokolenie każdego α;
        4.4.3 – P1: udziały akcji (całe pokolenie i krok 0), dylematy agentów z ≥ D_min decyzjami;
        4.4.4 – P2, wszystkie pokolenia, średnie po ziarnach.
    """
    argumenty = utworz_parser().parse_args(argv)
    a, D_min, wyjscie = argumenty.a, argumenty.D_min, argumenty.wyjscie
    katalog_tabel = wyjscie / "tabele"
    katalog_tabel.mkdir(parents=True, exist_ok=True)

    # 1. Dane
    pokolenia, agenci = wczytaj_pilotaz(argumenty.katalog_wynikow)
    agenci = dodaj_wspolczynniki(agenci, a)
    pokolenia = pokolenia.merge(
        m_pop(agenci), on=["przebieg", "pokolenie"], how="left", validate="one_to_one"
    )
    agenci_P1 = agenci[agenci["seria"] == "P1"]
    agenci_P2 = agenci[agenci["seria"] == "P2"]
    pokolenia_P1 = pokolenia[pokolenia["seria"] == "P1"]
    pokolenia_P2 = pokolenia[pokolenia["seria"] == "P2"]
    if agenci_P1.empty or agenci_P2.empty:
        raise FileNotFoundError("Eksploracja wymaga przebiegów obu serii: P1 i P2")
    print(
        f"Wczytano {agenci['przebieg'].nunique()} przebiegów: {len(agenci)} wierszy "
        f"agentów, {len(pokolenia)} wierszy pokoleń. Parametry: a = {a}, D_min = {D_min}."
    )

    # 2. 4.3.1 – liczba dylematów, wybór a i D_min
    _naglowek("4.3.1 Liczba dylematów D na agenta")
    grupy_D = {"P1, pokolenie 0": agenci_P1}
    etykiety_tex = {"P1, pokolenie 0": "P1, pokolenie 0"}
    for alfa, grupa in agenci_P2.groupby("alfa"):
        ostatnie = grupa["pokolenie"].max()
        nazwa = f"P2, {_etykieta_alfa(alfa)}, pokolenie {ostatnie}"
        grupy_D[nazwa] = grupa[grupa["pokolenie"] == ostatnie]
        etykiety_tex[nazwa] = f"P2, {_etykieta_alfa_tex(alfa)}, pokolenie {ostatnie}"
    # concat słownika – nazwa grupy jako pierwszy poziom indeksu
    podsumowanie_D = pd.concat({nazwa: podsumuj_D(grupa, D_min) for nazwa, grupa in grupy_D.items()})
    _pokaz_i_zapisz(podsumowanie_D, wyjscie / "D_podsumowanie.csv")
    wiersze = []
    for (grupa, podgrupa), w in podsumowanie_D.iterrows():
        etykieta = etykiety_tex[grupa] if podgrupa == "wszyscy" else f"\\quad {podgrupa}"
        wiersze.append(
            [
                etykieta,
                liczba_tex(w["liczba"]),
                procent_tex(w["odsetek_D0"]),
                procent_tex(w["odsetek_ponizej_Dmin"]),
                liczba_tex(w["mediana"], 1),
                f"{liczba_tex(w['Q1'], 1)}--{liczba_tex(w['Q3'], 1)}",
            ]
        )
    zapisz_tabele_tex(
        wiersze,
        katalog_tabel / "tab_dylematy.tex",
        naglowek=["Grupa agentów", "Agenci", "$D = 0$", "$D < D_{\\min}$", "Mediana $D$", "$Q_1$--$Q_3$"],
        podpis=f"Liczba dylematów $D$ na agenta: P1 (pokolenie zerowe, 5 ziaren) i ostatnie pokolenie P2 dla każdej wartości $\\alpha$ ($D_{{\\min}} = {D_min}$)",
        etykieta="tab:dylematy",
        szerokosc_pierwszej=0.34,
    )

    print("\nRozrzut M w przedziałach D (obserwowany a szum przypadku):")
    granice = _przedzialy_D(agenci.loc[agenci["D"] > 0, "D"], D_min, od=1)
    rozrzut = {"P1": rozrzut_M_wg_D(agenci_P1, D_min, granice), "P2": rozrzut_M_wg_D(agenci_P2, D_min, granice)}
    _pokaz_i_zapisz(pd.concat(rozrzut), wyjscie / "M_rozrzut_wg_D.csv")
    obok_siebie = rozrzut["P1"].join(rozrzut["P2"], suffixes=("_P1", "_P2"), how="outer")
    obok_siebie = obok_siebie.reindex([e for e in _etykiety_przedzialow(granice) if e in obok_siebie.index])
    wiersze = []
    for przedzial, w in obok_siebie.iterrows():
        wiersz = [przedzial.replace("–", "--")]
        for seria in ("P1", "P2"):
            wiersz += [
                liczba_tex(w[f"liczba_{seria}"], 0),
                liczba_tex(w[f"odch_std_M_{seria}"]),
                liczba_tex(w[f"odch_std_szum_{seria}"]),
            ]
        wiersze.append(wiersz)
    zapisz_tabele_tex(
        wiersze,
        katalog_tabel / "tab_rozrzut.tex",
        naglowek=["Przedział $D$", "P1: agenci", "P1: odch. std. $M$", "P1: szum", "P2: agenci", "P2: odch. std. $M$", "P2: szum"],
        podpis="Odchylenie standardowe współczynnika $M$ w przedziałach liczby dylematów $D$ (agenci z $D > 0$) i odchylenie, które dałby sam przypadek przy tych samych skłonnościach wszystkich agentów (szum)",
        etykieta="tab:rozrzut",
        szerokosc_pierwszej=0.16,
    )
    wykres_histogram_D(
        {nazwa: grupa["D"] for nazwa, grupa in grupy_D.items()}, D_min, wyjscie / "histogram_D.pdf"
    )
    wykres_M_od_D(agenci, D_min, wyjscie / "M_od_D.pdf")

    # 3. 4.4.2 – statystyki opisowe
    _naglowek("4.4.2 Statystyki opisowe")
    print("P1 (pokolenie 0, 5 ziaren; zmienne agentów – wszyscy agenci):")
    statystyki_P1 = statystyki_opisowe(agenci_P1, pokolenia_P1)
    _pokaz_i_zapisz(statystyki_P1, wyjscie / "statystyki_P1.csv")
    zapisz_tabele_tex(
        [
            [ETYKIETY_TEX[nazwa]] + [liczba_tex(w[k]) for k in ("srednia", "mediana", "odch_std", "min", "maks")]
            for nazwa, w in statystyki_P1.iterrows()
        ],
        katalog_tabel / "tab_statystyki_P1.tex",
        naglowek=["Zmienna", "Średnia", "Mediana", "Odch. std.", "Min.", "Maks."],
        podpis=f"Statystyki opisowe – P1 (pokolenie zerowe, {agenci_P1['przebieg'].nunique()} ziaren, {len(agenci_P1)} agentów); zmienne agentów – wszyscy agenci",
        etykieta="tab:statystyki",
        szerokosc_pierwszej=0.34,
    )
    for alfa, grupa in agenci_P2.groupby("alfa"):
        print(f"\nP2, {_etykieta_alfa(alfa)} (wszystkie pokolenia i ziarna łącznie – tylko CSV):")
        _pokaz_i_zapisz(
            statystyki_opisowe(grupa, pokolenia_P2[pokolenia_P2["alfa"] == alfa]),
            wyjscie / f"statystyki_P2_alfa_{alfa}.csv",
        )
    print("\nP2 – średnie w pierwszym i ostatnim pokoleniu:")
    pierwsze_ostatnie = statystyki_pierwsze_ostatnie(agenci_P2, pokolenia_P2)
    _pokaz_i_zapisz(pierwsze_ostatnie, wyjscie / "statystyki_P2_pierwsze_ostatnie.csv")
    ziarna = pokolenia_P2.groupby(["alfa", "pokolenie"])["przebieg"].nunique()
    mniej_ziaren = [
        f"{_etykieta_alfa_tex(alfa)} – {ziarna[(alfa, numer)]} z {ziarna[(alfa, 0)]} ziaren"
        for alfa, numer in pierwsze_ostatnie.columns
        if numer > 0 and ziarna[(alfa, numer)] < ziarna[(alfa, 0)]
    ]
    uwaga = "Zmienne agentów – średnia po agentach pokolenia (ziarna łącznie); zmienne pokoleń – średnia po ziarnach."
    if mniej_ziaren:
        uwaga += " W ostatnim pokoleniu przy " + "; ".join(mniej_ziaren) + " (reszta wymarła wcześniej)."
    # wiersz nagłówka nad parami kolumn: wartość α
    wartosci_alfa = list(dict.fromkeys(alfa for alfa, _ in pierwsze_ostatnie.columns))
    naglowek_nad = " & " + " & ".join(
        f"\\multicolumn{{2}}{{l}}{{{_etykieta_alfa_tex(alfa)}}}" for alfa in wartosci_alfa
    )
    zapisz_tabele_tex(
        [
            [ETYKIETY_TEX[nazwa]] + [liczba_tex(v) for v in w]
            for nazwa, w in pierwsze_ostatnie.iterrows()
        ],
        katalog_tabel / "tab_statystyki_P2.tex",
        naglowek=["Zmienna"] + [f"pok.~{numer}" for _, numer in pierwsze_ostatnie.columns],
        podpis="Statystyki opisowe – P2: średnie w pierwszym i ostatnim pokoleniu dla każdej wartości $\\alpha$",
        etykieta="tab:statystyki-p2",
        szerokosc_pierwszej=0.24,
        uwaga=uwaga,
        naglowek_nad=naglowek_nad,
    )

    # 4. 4.4.3 – pokolenie zerowe jako punkt odniesienia
    _naglowek("4.4.3 Pokolenie zerowe (P1)")
    przebiegi_P1 = sorted(agenci_P1["przebieg"].unique())
    udzialy = udzialy_akcji(pokolenia_P1)
    udzialy_k0 = udzialy_akcji_krok0(argumenty.katalog_wynikow, przebiegi_P1)
    print("Udziały akcji: oczekiwane w kroku 0 i we wszystkich decyzjach pokolenia:")
    _pokaz_i_zapisz(pd.concat([udzialy_k0, udzialy], axis=1), wyjscie / "udzialy_akcji_P1.csv")
    z_dylematami = agenci_P1[agenci_P1["D"] > 0]
    print(f"\nŚredni profil (P_S, P_K, P_O) agentów z D > 0 (n = {len(z_dylematami)}):")
    print(z_dylematami[["s", "k", "n"]].mean().round(3).to_string())
    print(
        f"M agentów z D > 0: średnia {z_dylematami['M'].mean():.3f}, "
        f"mediana {z_dylematami['M'].median():.3f}, "
        f"odsetek M = 0: {(z_dylematami['M'] == 0).mean():.3f}"
    )
    M_pop_P1 = (agenci_P1["S"].sum() - agenci_P1["K"].sum()) / agenci_P1["D"].sum()
    zakres_P1 = (pokolenia_P1["M_pop"].min(), pokolenia_P1["M_pop"].max())
    print(
        f"M_pop całej serii P1 (poziom odniesienia dla 4.4.4): {M_pop_P1:.4f}; "
        f"zakres między ziarnami: {zakres_P1[0]:.3f} … {zakres_P1[1]:.3f}"
    )

    dylematy_P1 = wczytaj_dylematy(argumenty.katalog_wynikow, przebiegi_P1)
    najczestsza = najczestsza_akcja_w_dylematach(dylematy_P1, min_decyzji=D_min)
    print(
        f"\nUdział najczęstszej akcji agenta w jego dylematach (agenci z co najmniej "
        f"{D_min} decyzjami, n = {len(najczestsza)}): mediana {najczestsza.median():.3f}, "
        f"odsetek agentów z udziałem ≥ 0,9: {(najczestsza >= 0.9).mean():.3f}"
    )
    wykres_pokolenie_zerowe(udzialy, agenci_P1, wyjscie / "pokolenie_zerowe.pdf", udzialy_krok0=udzialy_k0)

    # 5. 4.4.4 – kolejne pokolenia dla trzech α
    _naglowek("4.4.4 Przeżywalność i M_pop w kolejnych pokoleniach (P2)")
    w_pokoleniach = przebieg_w_pokoleniach(pokolenia_P2)
    w_pokoleniach.to_csv(wyjscie / "przebieg_w_pokoleniach.csv", index=False, float_format="%.4f")
    wybrane_numery = [0, 1, 2, 5, 10, 20, 30, 40, 49]
    wybrane = w_pokoleniach[w_pokoleniach["pokolenie"].isin(wybrane_numery)]
    _pokaz_i_zapisz(wybrane.set_index(["alfa", "pokolenie"]), wyjscie / "przebieg_wybrane_pokolenia.csv")
    wymarle = pokolenia_P2.loc[pokolenia_P2["zywi_koniec"] == 0, ["przebieg", "pokolenie"]]
    print("Wymarcia:", ", ".join(f"{w.przebieg} (pokolenie {w.pokolenie})" for w in wymarle.itertuples()) or "brak")
    awaryjne = pokolenia_P2.groupby("alfa")["awaryjny"].sum()
    print(
        "Pokolenia z wariantem awaryjnym (suma po ziarnach):",
        ", ".join(f"{_etykieta_alfa(alfa)}: {int(liczba)}" for alfa, liczba in awaryjne.items()),
    )
    akcje = udzialy_akcji_w_pokoleniach(pokolenia_P2)
    akcje.to_csv(wyjscie / "akcje_w_pokoleniach.csv", index=False, float_format="%.4f")
    print("\nUdział grup akcji w wybranych pokoleniach:")
    print(akcje[akcje["pokolenie"].isin(wybrane_numery)].set_index(["alfa", "pokolenie"]).round(3).to_string())
    wykres_przezywalnosc(pokolenia_P2, wyjscie / "przezywalnosc.pdf")
    wykres_m_pop(
        pokolenia_P2, wyjscie / "M_pop.pdf", poziom_odniesienia=M_pop_P1, zakres_odniesienia=zakres_P1
    )
    wykres_akcje_w_pokoleniach(pokolenia_P2, wyjscie / "akcje_w_pokoleniach.pdf")

    print(f"\nGotowe. Tabele i wykresy: {wyjscie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
