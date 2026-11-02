"""Eksploracja danych z pilotażu Lab02 (zadanie 18; Lab02, punkty 4.3.1 i 4.4).

Skrypt wczytuje przebiegi pilotażowe, liczy współczynniki z punktu 3.4.3
(profil, M, M_wyg, M_pop), wypisuje liczby do tabel i zapisuje wykresy.
Symulacji nie uruchamia – pracuje wyłącznie na zapisanych plikach CSV.

Uruchomienie z katalogu kod/:
    PYTHONPATH=src python analiza/eksploracja.py ../asystent/Lab02/wyniki

Co powstaje w katalogu --wyjscie (domyślnie wyniki/analiza, poza Gitem):
    4.3.1  D_podsumowanie.csv, M_rozrzut_wg_D.csv, histogram_D.pdf, M_od_D.pdf
    4.4.2  statystyki_P1.csv, statystyki_P2_alfa_*.csv
    4.4.3  udzialy_akcji_P1.csv, pokolenie_zerowe.pdf
    4.4.4  przebieg_w_pokoleniach.csv, przezywalnosc.pdf, M_pop.pdf
Te same liczby skrypt wypisuje na ekran.

Seria P3 służy tylko do sprawdzenia powtarzalności (NF-03): P3_a i P3_b to ten
sam przebieg, więc w eksploracji jej nie liczymy – inaczej te same dane
weszłyby do statystyk dwa razy.

Uwaga przy grupowaniu: P1 też ma α = 1,0. Analizy „po α” dotyczą tylko serii P2,
więc przed groupby("alfa") zawsze wybieramy seria == "P2".
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moralnosc.stan import DZIEL, KRADNIJ, LICZBA_AKCJI, NIC, ZBIERZ

# Kody przyczyn śmierci z pomiar.py (kolumna przyczyna_smierci w agenci.csv).
PRZYCZYNA_WYCZERPANIE = 1
PRZYCZYNA_KONIEC_POKOLENIA = 2

# Etykiety akcji na wykresach; kody 0–3 to ruch na N, E, S, W.
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

# Dolne granice przedziałów liczby dylematów D w tabelach i na histogramie.
# Rozkład D jest bardzo skośny (mediana w P1 to 2, maksimum 300), więc
# przedziały rosną mniej więcej geometrycznie; górną granicę dobiera się z danych.
GRANICE_D = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200)


# --- 1. Wczytywanie -------------------------------------------------------------------


def _wczytaj_konfiguracje(katalog: Path) -> dict:
    """Konfiguracja przebiegu z pliku konfiguracja.json, bez metadanych."""
    with (katalog / "konfiguracja.json").open(encoding="utf-8") as plik:
        return json.load(plik)["konfiguracja"]


def _dopisz_opis_przebiegu(tabela: pd.DataFrame, katalog: Path, cfg: dict) -> None:
    """Wstawia na początek tabeli kolumny przebieg, seria, alfa i ziarno.

    Zmienia tabelę w miejscu (DataFrame.insert) – wywołujemy ją tylko na
    tabelach świeżo wczytanych z pliku, więc nic innego z nich nie korzysta.
    Każda kolumna ma tę samą wartość we wszystkich wierszach tabeli.
    """
    opis = {
        "przebieg": katalog.name,  # np. "P2_niedobor_z201"
        "seria": katalog.name.split("_")[0],  # część nazwy przed pierwszym "_"
        "alfa": cfg["zasoby"]["alfa"],
        "ziarno": cfg["przebieg"]["ziarno"],
    }
    for pozycja, (nazwa, wartosc) in enumerate(opis.items()):
        tabela.insert(pozycja, nazwa, wartosc)


def wczytaj_przebieg(katalog: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Wczytuje jeden przebieg: pokolenia.csv, agenci.csv i konfigurację.

    Argumenty:
        katalog: katalog jednego przebiegu, np. wyniki/P1_z101.

    Zwraca:
        (pokolenia, agenci, cfg) – dwie tabele pandas i słownik konfiguracji
        (klucz "konfiguracja" z pliku konfiguracja.json, bez metadanych).

    Pliku dylematy.csv tu nie wczytujemy – jest duży (do 80 MB na przebieg),
    a potrzebny tylko w punkcie 4.4.3 (patrz wczytaj_dylematy). Kolumny
    True/False (awaryjny, kwalifikacja) pandas sam zamienia na typ bool.
    """
    pokolenia = pd.read_csv(katalog / "pokolenia.csv")
    agenci = pd.read_csv(katalog / "agenci.csv")
    cfg = _wczytaj_konfiguracje(katalog)
    return pokolenia, agenci, cfg


def wczytaj_pilotaz(
    katalog_wynikow: Path, serie: tuple[str, ...] = ("P1", "P2")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wczytuje wszystkie przebiegi z wybranych serii i skleja je w dwie tabele.

    Argumenty:
        katalog_wynikow: katalog z podkatalogami przebiegów (P1_z101, …).
        serie: serie, które wczytujemy (domyślnie bez P3); seria to część
               nazwy katalogu przed pierwszym „_”.

    Zwraca:
        (pokolenia, agenci) – tabele wszystkich przebiegów razem. Na początku
        każdej są cztery kolumny opisujące przebieg:
            przebieg – nazwa katalogu, np. "P2_niedobor_z201";
            seria    – "P1" albo "P2";
            alfa     – odrost z konfiguracji (cfg["zasoby"]["alfa"]);
            ziarno   – ziarno z konfiguracji (cfg["przebieg"]["ziarno"]).

    Sprawdzenie: P1 + P2 to 14 przebiegów, 83 200 wierszy agentów (5 · 200 +
    8 · 10 000 + 2 200) i 416 wierszy pokoleń (5 + 8 · 50 + 11).
    """
    wszystkie_pokolenia = []
    wszyscy_agenci = []
    # sorted – zawsze ta sama kolejność przebiegów, niezależnie od systemu plików
    for katalog in sorted(katalog_wynikow.iterdir()):
        # is_dir pomija pliki w katalogu wyników, np. .DS_Store na macOS
        if not katalog.is_dir() or katalog.name.split("_")[0] not in serie:
            continue
        pokolenia, agenci, cfg = wczytaj_przebieg(katalog)
        _dopisz_opis_przebiegu(pokolenia, katalog, cfg)
        _dopisz_opis_przebiegu(agenci, katalog, cfg)
        wszystkie_pokolenia.append(pokolenia)
        wszyscy_agenci.append(agenci)

    # Bez tego pd.concat na pustej liście zgłosiłby mniej czytelny błąd.
    if not wszyscy_agenci:
        raise FileNotFoundError(f"Brak przebiegów serii {serie} w {katalog_wynikow}")

    # ignore_index=True numeruje wiersze od nowa (0, 1, 2, …) w sklejonej tabeli;
    # bez tego każdy przebieg zaczynałby numerację od 0 i numery by się powtarzały.
    return (
        pd.concat(wszystkie_pokolenia, ignore_index=True),
        pd.concat(wszyscy_agenci, ignore_index=True),
    )


def wczytaj_dylematy(katalog_wynikow: Path, przebiegi: list[str]) -> pd.DataFrame:
    """Wczytuje dylematy.csv wskazanych przebiegów i dopisuje te same cztery
    kolumny opisu co wczytaj_pilotaz.

    Argumenty:
        katalog_wynikow: katalog z podkatalogami przebiegów.
        przebiegi: nazwy katalogów, np. ["P1_z101", "P1_z102"].

    Używane tylko w punkcie 4.4.3 dla serii P1 (5 małych plików), więc nie
    trzeba się martwić o pamięć.
    """
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
    """Zwraca kopię tabeli agentów z dopisanymi współczynnikami z punktu 3.4.3.

    Nowe kolumny:
        s, k, n             – profil (S/D, K/D, N_o/D);
        M                   – (S − K) / D;
        M_wyg               – (S − K) / (D + a);
        s_wyg, k_wyg, n_wyg – ((S+1)/(D+3), (K+1)/(D+3), (N_o+1)/(D+3)).

    Agent z D = 0 nie miał żadnej obserwacji: s, k, n i M są wtedy NaN
    (brak danych), a nie 0 (zachowanie neutralne). M_wyg i profil wygładzony
    są określone także dla D = 0, bo mianownik jest dodatni.
    """
    assert a > 0, f"Parametr wygładzania a musi być dodatni, a jest {a}"
    wynik = agenci.copy()  # kopia – tabela wejściowa zostaje bez zmian

    # where zostawia D tam, gdzie D > 0, a w pozostałych wierszach wstawia NaN.
    # Dzielenie przez NaN daje NaN, więc agenci bez dylematów dostają „brak danych”.
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
    """Współczynnik moralności populacji w każdym pokoleniu każdego przebiegu.

    M_pop = (ΣS − ΣK) / ΣD po agentach jednego pokolenia; gdy ΣD = 0, wynik
    jest nieokreślony (NaN), bo brak dylematów to brak obserwacji.

    Zwraca:
        tabelę z kolumnami przebieg, pokolenie, M_pop – jeden wiersz na
        pokolenie przebiegu.
    """
    # as_index=False zostawia przebieg i pokolenie jako zwykłe kolumny wyniku.
    sumy = agenci.groupby(["przebieg", "pokolenie"], as_index=False)[["S", "K", "D"]].sum()
    sumy["M_pop"] = (sumy["S"] - sumy["K"]) / sumy["D"].where(sumy["D"] > 0)
    return sumy[["przebieg", "pokolenie", "M_pop"]]


# --- 3. Liczba dylematów (punkt 4.3.1) ------------------------------------------------


def _przedzialy_D(D: pd.Series, D_min: int | None = None, od: int = 0) -> list[int]:
    """Granice przedziałów D: GRANICE_D od wartości `od`, próg D_min jako
    dodatkowa granica i górna granica większa od największego D w danych."""
    granice = {g for g in GRANICE_D if g >= od}
    if D_min is not None:
        granice.add(D_min)  # dzięki temu żaden przedział nie przecina progu
    gorna = max(int(D.max()) + 1 if len(D) else 0, max(granice) + 1)
    return sorted(granice) + [gorna]


def _etykiety_przedzialow(granice: list[int]) -> list[str]:
    """Etykiety przedziałów [dolna, górna): „0”, „3–4”, „200–300” itd."""
    etykiety = []
    for dolna, gorna in zip(granice[:-1], granice[1:]):
        etykiety.append(f"{dolna}" if gorna - dolna == 1 else f"{dolna}–{gorna - 1}")
    return etykiety


def podsumuj_D(agenci: pd.DataFrame, D_min: int) -> pd.DataFrame:
    """Opis rozkładu liczby dylematów D w jednej grupie agentów.

    Zwraca tabelę z trzema wierszami: „wszyscy”, „zmarli” (przyczyna 1)
    i „dożyli końca” (przyczyna 2), oraz kolumnami:
        liczba, odsetek_D0, odsetek_ponizej_Dmin, mediana, Q1, Q3.

    Grupę wybiera się przed wywołaniem – w main są to P1 (pokolenie 0 pięciu
    ziaren) i ostatnie pokolenie P2 dla każdego α.
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
            # średnia z wartości True/False to odsetek wartości True
            "odsetek_D0": (D == 0).mean(),
            "odsetek_ponizej_Dmin": (D < D_min).mean(),
            "mediana": D.median(),
            "Q1": D.quantile(0.25),
            "Q3": D.quantile(0.75),
        }
    return pd.DataFrame.from_dict(wiersze, orient="index")


def rozrzut_M_wg_D(agenci: pd.DataFrame, D_min: int) -> pd.DataFrame:
    """Rozrzut współczynnika M w przedziałach D w porównaniu z samym przypadkiem.

    Podstawa wyboru D_min i a w punkcie 4.3.1. Wymaga tabeli po
    dodaj_wspolczynniki; agenci z D = 0 nie mają M, więc są pomijani.

    Każdą decyzję w dylemacie traktujemy jak liczbę x: +1 (dzielenie),
    −1 (kradzież), 0 (akcja obojętna); M agenta to średnia z jego D decyzji.
    Gdyby wszyscy agenci w grupie mieli te same skłonności p_s i p_k (równe
    udziałom w całej grupie), to:
        wariancja jednej decyzji = E[x²] − (E[x])² = (p_s + p_k) − (p_s − p_k)²,
        odchylenie standardowe M = √(wariancja jednej decyzji / D).
    To jest „szum” – rozrzut, który dałby sam przypadek przy D decyzjach.

    Zwraca tabelę z jednym wierszem na przedział D i kolumnami:
        liczba            – liczba agentów w przedziale;
        odch_std_M        – obserwowane odchylenie standardowe M;
        odch_std_szum     – odchylenie, które dałby sam przypadek (wzór wyżej);
        odsetek_M0        – odsetek agentów z M = 0;
        odsetek_skrajnych – odsetek agentów z |M| = 1.

    Odczyt: gdy odch_std_M jest bliskie odch_std_szum, różnice M między
    agentami w tym przedziale są głównie przypadkowe. D_min warto ustawić
    tam, gdzie obserwowany rozrzut zaczyna wyraźnie przewyższać szum.
    """
    z_dylematami = agenci[agenci["D"] > 0]
    p_s = z_dylematami["S"].sum() / z_dylematami["D"].sum()
    p_k = z_dylematami["K"].sum() / z_dylematami["D"].sum()
    wariancja_decyzji = (p_s + p_k) - (p_s - p_k) ** 2

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
    # Średnia wariancji w przedziale, a dopiero potem pierwiastek – tak łączy się
    # wariancje agentów o różnym D.
    tabela.insert(2, "odch_std_szum", np.sqrt(tabela.pop("wariancja_szumu")))
    return tabela


def _zapisz_wykres(fig: plt.Figure, plik: Path) -> None:
    """Zapisuje wykres i zamyka go (zamknięcie zwalnia pamięć).

    CreationDate=None usuwa datę z metadanych PDF, więc ten sam wykres daje
    ten sam plik przy każdym uruchomieniu.
    """
    metadane = {"CreationDate": None} if plik.suffix == ".pdf" else None
    fig.savefig(plik, bbox_inches="tight", dpi=300, metadata=metadane)
    plt.close(fig)


def wykres_histogram_D(grupy: dict[str, pd.Series], D_min: int, plik: Path) -> None:
    """Rozkład liczby dylematów D na agenta dla kilku grup (atrapa w 4.3.1).

    Argumenty:
        grupy: etykieta grupy → kolumna D tej grupy, np.
               {"P1, pokolenie 0": …, "P2, α = 1,0, pokolenie 49": …}.
        D_min: przedziały poniżej progu są szare.
        plik:  ścieżka wyjściowa (PDF – wektorowy, dobry do LaTeX-a).

    Rozkład jest bardzo skośny, dlatego zamiast zwykłego histogramu słupki
    pokazują udział agentów w przedziałach rosnących geometrycznie (GRANICE_D);
    D = 0 ma własny słupek. Panele są ułożone po dwa w wierszu.
    """
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
    for os_ in osie[-1]:
        os_.set_xlabel("Liczba dylematów D")
    for os_ in osie[:, 0]:
        os_.set_ylabel("Udział agentów")
    for os_ in list(osie.flat)[len(grupy):]:
        os_.set_visible(False)  # pusty panel przy nieparzystej liczbie grup
    fig.suptitle(f"Szare słupki: D < D_min = {D_min}", fontsize=8, y=1.0)
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


def wykres_M_od_D(agenci: pd.DataFrame, D_min: int, plik: Path) -> None:
    """M i M_wyg w zależności od D, agenci z D < D_min wyróżnieni (atrapa w 4.3.1).

    Wymaga tabeli po dodaj_wspolczynniki. Dwa panele obok siebie: lewy M,
    prawy M_wyg, oś x – D w skali logarytmicznej (agenci z D = 0 nie mają M,
    a ich M_wyg wynosi 0, więc pominięto ich na obu panelach). Na lewym panelu
    widać „wachlarz” wartości przy małym D, na prawym – jak wygładzanie
    ściąga go ku 0.

    Punkty są małe i półprzezroczyste, bo agentów jest kilkadziesiąt tysięcy;
    rasterized=True zapisuje je w PDF jako obraz, inaczej plik miałby
    kilkanaście MB.
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
    osie[0].set_ylabel("Wartość współczynnika")
    legenda = osie[0].legend(loc="lower right", fontsize=8, markerscale=3)
    for uchwyt in legenda.legend_handles:
        uchwyt.set_alpha(1)  # w legendzie punkty bez przezroczystości
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


# --- 4. Statystyki opisowe (punkt 4.4.2) ----------------------------------------------


def opisz(seria: pd.Series) -> dict[str, float]:
    """Liczby do jednego wiersza tabeli: liczba obserwacji, średnia, mediana,
    odch. std., min i maks.

    Wartości NaN są pomijane (pandas robi to domyślnie w count, mean, median,
    std, min i max). Odchylenie standardowe jest z próby (dzielnik n − 1).
    Kolumny „liczba” nie ma w tabeli dokumentu – przydaje się do podpisu.
    """
    return {
        "liczba": int(seria.count()),
        "srednia": seria.mean(),
        "mediana": seria.median(),
        "odch_std": seria.std(),
        "min": seria.min(),
        "maks": seria.max(),
    }


def statystyki_opisowe(agenci: pd.DataFrame, pokolenia: pd.DataFrame) -> pd.DataFrame:
    """Tabela „Statystyki opisowe” z punktu 4.4.2 dla jednej grupy danych.

    Argumenty:
        agenci:    agenci po dodaj_wspolczynniki (wybrana grupa);
        pokolenia: pokolenia tej samej grupy z dołączoną kolumną M_pop.

    Zwraca tabelę z wierszami w kolejności z dokumentu (S, K i N_o w osobnych
    wierszach) i kolumnami z funkcji opisz. Zmienne agentów dotyczą
    wszystkich agentów grupy – żywych i zmarłych – z wyjątkiem M (tylko
    D > 0, bo przy D = 0 jest NaN) i kroku śmierci (tylko zmarli
    z wyczerpania; dożywający końca mają krok_smierci = liczba kroków).
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


# --- 5. Pokolenie zerowe (punkt 4.4.3) ------------------------------------------------


def udzialy_akcji(pokolenia: pd.DataFrame) -> pd.Series:
    """Udział każdej z ośmiu akcji we wszystkich wyborach akcji (intencjach).

    Liczone z kolumn akcja_0 … akcja_7 pliku pokolenia.csv (zliczenia wszystkich
    decyzji, także poza dylematami), zsumowanych po przekazanych wierszach.
    Zwraca Series z indeksem z NAZWY_AKCJI i wartościami sumującymi się do 1.
    """
    sumy = pokolenia[KOLUMNY_AKCJI].sum()
    udzialy = sumy / sumy.sum()
    udzialy.index = [NAZWY_AKCJI[kod] for kod in range(LICZBA_AKCJI)]
    udzialy.name = "udzial"  # nagłówek kolumny w CSV
    return udzialy


def najczestsza_akcja_w_dylematach(
    dylematy: pd.DataFrame, min_decyzji: int = 1
) -> pd.Series:
    """Udział najczęstszej akcji każdego agenta w jego decyzjach w dylematach.

    Odpowiada na pytanie z 4.4.3: czy softmax daje zróżnicowane decyzje, czy
    prawie deterministyczne. Ograniczenie danych: pliki CSV nie zawierają
    akcji agenta poza dylematami (agenci.csv ma tylko S, K, N_o), więc liczymy
    wyłącznie decyzje z dylematy.csv – trzeba to zapisać w tekście.

    Argumenty:
        dylematy: tabela z wczytaj_dylematy.
        min_decyzji: pomija agentów z mniejszą liczbą decyzji – przy 1–2
                     decyzjach udział 1 nie świadczy o determinizmie.

    Zwraca Series: jeden element na agenta (indeks: przebieg, id), wartość
    w przedziale (0, 1]; 1 = agent w dylematach zawsze wybierał to samo.
    """
    agent = ["przebieg", "id"]  # id jest unikalne w obrębie przebiegu
    # value_counts(normalize=True) w każdej grupie: udział każdej akcji agenta
    udzialy = dylematy.groupby(agent)["akcja"].value_counts(normalize=True)
    najczestsza = udzialy.groupby(level=agent).max()
    liczba_decyzji = dylematy.groupby(agent).size()
    return najczestsza[liczba_decyzji >= min_decyzji]


def wykres_pokolenie_zerowe(
    udzialy: pd.Series, agenci_P1: pd.DataFrame, plik: Path
) -> None:
    """Rozkład akcji i współczynnika M w pokoleniu zerowym (atrapa w 4.4.3).

    Dwa panele: lewy – udziały ośmiu akcji we wszystkich decyzjach, prawy –
    histogram M agentów P1 z D > 0 (21 przedziałów na [−1, 1], więc M = 0
    ma własny przedział pośrodku). Wymaga agenci_P1 po dodaj_wspolczynniki.
    """
    M = agenci_P1["M"].dropna()
    fig, (lewy, prawy) = plt.subplots(
        1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    lewy.bar(udzialy.index, udzialy.to_numpy(), color="C0")
    lewy.tick_params(axis="x", rotation=90)
    lewy.set_ylabel("Udział decyzji")
    lewy.set_title("Wybrane akcje (wszystkie decyzje)", fontsize=9)

    prawy.hist(M, bins=21, range=(-1, 1), color="C0", edgecolor="white")
    prawy.set_xlabel(r"$M$")
    prawy.set_ylabel("Liczba agentów")
    prawy.set_title(f"$M$ agentów z D > 0 (n = {len(M)})", fontsize=9)
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


# --- 6. Kolejne pokolenia dla trzech α (punkt 4.4.4) ----------------------------------


def _etykieta_alfa(alfa: float) -> str:
    """„α = 0,85”, „α = 1,0” – z przecinkiem dziesiętnym, jak w dokumencie."""
    tekst = f"{alfa:.2f}"
    if tekst.endswith("0"):
        tekst = tekst[:-1]  # 1.00 → 1.0, ale 0.85 zostaje
    return "α = " + tekst.replace(".", ",")


def przebieg_w_pokoleniach(pokolenia: pd.DataFrame) -> pd.DataFrame:
    """Przeżywalność, zakwalifikowani, wariant awaryjny i M_pop w kolejnych pokoleniach.

    Argument: pokolenia serii P2 z dołączoną kolumną M_pop.

    Zwraca tabelę z jednym wierszem na parę (alfa, pokolenie) i kolumnami:
        liczba_ziaren      – ile ziaren doszło do tego pokolenia (przebieg,
                             który wymarł, kończy się wcześniej);
        przezywalnosc_sr, zakwalifikowani_sr, M_pop_sr – średnie po tych ziarnach;
        awaryjny_liczba    – w ilu ziarnach zadziałał wariant awaryjny.

    P2_niedobor_z201 wymarł w pokoleniu 10, więc od pokolenia 11 średnie dla
    α = 0,85 są z dwóch ziaren – pokazuje to liczba_ziaren, a na wykresach
    widać też linie poszczególnych ziaren.
    """
    return pokolenia.groupby(["alfa", "pokolenie"], as_index=False).agg(
        liczba_ziaren=("przebieg", "nunique"),
        przezywalnosc_sr=("przezywalnosc", "mean"),
        zakwalifikowani_sr=("zakwalifikowani", "mean"),
        M_pop_sr=("M_pop", "mean"),
        awaryjny_liczba=("awaryjny", "sum"),
    )


def _linie_ziaren_i_srednia(
    os_: plt.Axes, pokolenia: pd.DataFrame, kolumna: str, kolumna_sredniej: str
) -> None:
    """Rysuje na jednej osi cienkie linie ziaren i grubą linię średniej dla każdego α.

    Wymarcie (zywi_koniec = 0 w ostatnim pokoleniu przebiegu) jest zaznaczone
    krzyżykiem. Po wymarciu średnia liczy się z mniejszej liczby ziaren i może
    skoczyć tylko dlatego, że odpadł najsłabszy przebieg – tę część średniej
    rysujemy linią przerywaną.
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
        # Wymarły przebieg nie wraca, więc „pełne” pokolenia (wszystkie ziarna)
        # tworzą ciągły początek tabeli, a dalej są już tylko niepełne.
        pelne = srednia["liczba_ziaren"] == srednia["liczba_ziaren"].max()
        os_.plot(
            srednia.loc[pelne, "pokolenie"],
            srednia.loc[pelne, kolumna_sredniej],
            color=kolor,
            linewidth=2,
            label=_etykieta_alfa(alfa),
        )
        if not pelne.all():
            # Przerywana część zaczyna się w ostatnim pełnym pokoleniu, żeby obie
            # części linii się łączyły.
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
    """Przeżywalność i liczba zakwalifikowanych w kolejnych pokoleniach (atrapa w 4.4.4).

    Dwa panele jeden pod drugim, wspólna oś x (pokolenie). Jeden kolor na α,
    cienkie linie – poszczególne ziarna, gruba linia – średnia po ziarnach
    (przerywana, gdy część ziaren już wymarła), krzyżyk – wymarcie.
    Argument: pokolenia serii P2.
    """
    fig, (gora, dol) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    _linie_ziaren_i_srednia(gora, pokolenia, "przezywalnosc", "przezywalnosc_sr")
    _linie_ziaren_i_srednia(dol, pokolenia, "zakwalifikowani", "zakwalifikowani_sr")
    gora.set_ylabel("Przeżywalność")
    gora.set_ylim(0, 1.02)
    dol.set_ylabel("Zakwalifikowani")
    dol.set_xlabel("Pokolenie")
    gora.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


def wykres_m_pop(
    pokolenia: pd.DataFrame, plik: Path, poziom_odniesienia: float | None = None
) -> None:
    """M_pop w kolejnych pokoleniach dla trzech α (atrapa w 4.4.4).

    Układ linii jak w wykres_przezywalnosc. Pozioma przerywana linia to
    poziom odniesienia z pokolenia zerowego (punkt 4.4.3): zmianę M_pop można
    przypisać ewolucji tylko wtedy, gdy wyraźnie od niego odbiega. W main jest
    to M_pop całej serii P1 – ta sama miara co na osi y, więc porównanie jest
    uczciwe (średnia z M agentów ważyłaby agentów z 1 dylematem tak samo jak
    tych z 300).
    """
    fig, os_ = plt.subplots(figsize=(6.3, 2.8))
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
    fig.tight_layout()
    _zapisz_wykres(fig, plik)


# --- 7. Całość ------------------------------------------------------------------------


def utworz_parser() -> argparse.ArgumentParser:
    """Argumenty wiersza poleceń skryptu."""
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
    """Wyraźny nagłówek sekcji na ekranie."""
    print(f"\n=== {tekst} " + "=" * max(0, 70 - len(tekst)))


def _pokaz_i_zapisz(tabela: pd.DataFrame | pd.Series, plik: Path) -> None:
    """Wypisuje tabelę (zaokrągloną do 3 miejsc) i zapisuje ją do CSV (4 miejsca)."""
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(tabela.round(3))
    tabela.to_csv(plik, float_format="%.4f")
    print(f"→ {plik}")


def main(argv: list[str] | None = None) -> int:
    """Wykonuje całą eksplorację i wypisuje liczby do dokumentu.

    Decyzje o doborze danych (do opisania w tekście):
        4.3.1 – rozkład D dla P1 (pokolenie 0, 5 ziaren) i dla ostatniego
                pokolenia P2 każdego α; rozrzut M osobno dla P1 i P2;
        4.4.2 – P1: pokolenie 0 pięciu ziaren; P2: każde α osobno, wszystkie
                pokolenia i ziarna łącznie; zmienne agentów – wszyscy agenci;
        4.4.3 – P1, decyzje w dylematach agentów z co najmniej D_min decyzjami;
        4.4.4 – P2, wszystkie pokolenia, średnie po ziarnach.
    Zwraca 0 po poprawnym zakończeniu.
    """
    argumenty = utworz_parser().parse_args(argv)
    a, D_min, wyjscie = argumenty.a, argumenty.D_min, argumenty.wyjscie
    wyjscie.mkdir(parents=True, exist_ok=True)

    # 1. Dane: współczynniki agentów i M_pop dołączone do tabeli pokoleń.
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

    # 2. Punkt 4.3.1 – liczba dylematów na agenta, wybór a i D_min.
    _naglowek("4.3.1 Liczba dylematów D na agenta")
    grupy_D = {"P1, pokolenie 0": agenci_P1}
    for alfa, grupa in agenci_P2.groupby("alfa"):
        ostatnie = grupa["pokolenie"].max()
        grupy_D[f"P2, {_etykieta_alfa(alfa)}, pokolenie {ostatnie}"] = grupa[
            grupa["pokolenie"] == ostatnie
        ]
    # pd.concat słownika tabel dokleja nazwę grupy jako pierwszy poziom indeksu
    _pokaz_i_zapisz(
        pd.concat({nazwa: podsumuj_D(grupa, D_min) for nazwa, grupa in grupy_D.items()}),
        wyjscie / "D_podsumowanie.csv",
    )
    print("\nRozrzut M w przedziałach D (obserwowany a szum przypadku):")
    _pokaz_i_zapisz(
        pd.concat({"P1": rozrzut_M_wg_D(agenci_P1, D_min), "P2": rozrzut_M_wg_D(agenci_P2, D_min)}),
        wyjscie / "M_rozrzut_wg_D.csv",
    )
    wykres_histogram_D(
        {nazwa: grupa["D"] for nazwa, grupa in grupy_D.items()}, D_min, wyjscie / "histogram_D.pdf"
    )
    wykres_M_od_D(agenci, D_min, wyjscie / "M_od_D.pdf")

    # 3. Punkt 4.4.2 – statystyki opisowe.
    _naglowek("4.4.2 Statystyki opisowe")
    print("P1 (pokolenie 0, 5 ziaren; zmienne agentów – wszyscy agenci):")
    _pokaz_i_zapisz(statystyki_opisowe(agenci_P1, pokolenia_P1), wyjscie / "statystyki_P1.csv")
    for alfa, grupa in agenci_P2.groupby("alfa"):
        print(f"\nP2, {_etykieta_alfa(alfa)} (wszystkie pokolenia i ziarna łącznie):")
        _pokaz_i_zapisz(
            statystyki_opisowe(grupa, pokolenia_P2[pokolenia_P2["alfa"] == alfa]),
            wyjscie / f"statystyki_P2_alfa_{alfa}.csv",
        )

    # 4. Punkt 4.4.3 – pokolenie zerowe jako punkt odniesienia.
    _naglowek("4.4.3 Pokolenie zerowe (P1)")
    udzialy = udzialy_akcji(pokolenia_P1)
    print("Udziały akcji we wszystkich decyzjach:")
    _pokaz_i_zapisz(udzialy, wyjscie / "udzialy_akcji_P1.csv")
    z_dylematami = agenci_P1[agenci_P1["D"] > 0]
    print(f"\nŚredni profil (s, k, n) agentów z D > 0 (n = {len(z_dylematami)}):")
    print(z_dylematami[["s", "k", "n"]].mean().round(3).to_string())
    print(
        f"M agentów z D > 0: średnia {z_dylematami['M'].mean():.3f}, "
        f"mediana {z_dylematami['M'].median():.3f}, "
        f"odsetek M = 0: {(z_dylematami['M'] == 0).mean():.3f}"
    )
    M_pop_P1 = (agenci_P1["S"].sum() - agenci_P1["K"].sum()) / agenci_P1["D"].sum()
    print(f"M_pop całej serii P1 (poziom odniesienia dla 4.4.4): {M_pop_P1:.4f}")

    dylematy_P1 = wczytaj_dylematy(
        argumenty.katalog_wynikow, sorted(agenci_P1["przebieg"].unique())
    )
    najczestsza = najczestsza_akcja_w_dylematach(dylematy_P1, min_decyzji=D_min)
    print(
        f"\nUdział najczęstszej akcji agenta w jego dylematach (agenci z co najmniej "
        f"{D_min} decyzjami, n = {len(najczestsza)}): mediana {najczestsza.median():.3f}, "
        f"odsetek agentów z udziałem ≥ 0,9: {(najczestsza >= 0.9).mean():.3f}"
    )
    wykres_pokolenie_zerowe(udzialy, agenci_P1, wyjscie / "pokolenie_zerowe.pdf")

    # 5. Punkt 4.4.4 – kolejne pokolenia dla trzech wartości α.
    _naglowek("4.4.4 Przeżywalność i M_pop w kolejnych pokoleniach (P2)")
    w_pokoleniach = przebieg_w_pokoleniach(pokolenia_P2)
    w_pokoleniach.to_csv(wyjscie / "przebieg_w_pokoleniach.csv", index=False, float_format="%.4f")
    wybrane = w_pokoleniach[w_pokoleniach["pokolenie"].isin([0, 1, 2, 5, 10, 20, 30, 40, 49])]
    _pokaz_i_zapisz(wybrane.set_index(["alfa", "pokolenie"]), wyjscie / "przebieg_wybrane_pokolenia.csv")
    wymarle = pokolenia_P2.loc[pokolenia_P2["zywi_koniec"] == 0, ["przebieg", "pokolenie"]]
    print("Wymarcia:", ", ".join(f"{w.przebieg} (pokolenie {w.pokolenie})" for w in wymarle.itertuples()) or "brak")
    awaryjne = pokolenia_P2.groupby("alfa")["awaryjny"].sum()
    print(
        "Pokolenia z wariantem awaryjnym (suma po ziarnach):",
        ", ".join(f"{_etykieta_alfa(alfa)}: {int(liczba)}" for alfa, liczba in awaryjne.items()),
    )
    wykres_przezywalnosc(pokolenia_P2, wyjscie / "przezywalnosc.pdf")
    wykres_m_pop(pokolenia_P2, wyjscie / "M_pop.pdf", poziom_odniesienia=M_pop_P1)

    print(f"\nGotowe. Tabele i wykresy: {wyjscie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
