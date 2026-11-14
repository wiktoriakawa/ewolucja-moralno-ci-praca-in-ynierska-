"""Kontrola jakości i podstawowa analiza jednego przebiegu symulacji."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from moralnosc.pomiar import (
    KOLUMNY_DYLEMATY,
    PRZYCZYNA_KONIEC_POKOLENIA,
    PRZYCZYNA_WYCZERPANIE,
)
from moralnosc.stan import (
    DZIEL,
    KRADNIJ,
    LICZBA_AKCJI,
    NIEWYKONANA,
    PORAZKA,
    SUKCES,
    WYKONANA,
)
from moralnosc.zapis import KOLUMNY_AGENCI, KOLUMNY_POKOLENIA

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PLIKI_PRZEBIEGU = (
    "konfiguracja.json",
    "pokolenia.csv",
    "agenci.csv",
    "dylematy.csv",
)

def _wynik_kontroli(
    kontrola: str,
    poprawna: bool,
    szczegoly: str,
) -> dict[str, object]:
    """Tworzy jeden wiersz raportu kontroli jakości."""
    return {
        "kontrola": kontrola,
        "poprawna": bool(poprawna),
        "szczegoly": szczegoly,
    }


def _sprawdz_kolumny(
    nazwa_pliku: str,
    dane: pd.DataFrame,
    wymagane: tuple[str, ...],
) -> dict[str, object]:
    """Sprawdza obecność wymaganych kolumn w jednej tabeli."""
    brakujace = sorted(set(wymagane) - set(dane.columns))
    return _wynik_kontroli(
        f"kolumny_{nazwa_pliku}",
        not brakujace,
        "brak" if not brakujace else ", ".join(brakujace),
    )


def wczytaj_przebieg(
    katalog: str | Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wczytuje konfigurację i trzy tabele jednego przebiegu."""
    katalog = Path(katalog)
    with (katalog / "konfiguracja.json").open(encoding="utf-8") as plik:
        konfiguracja = json.load(plik)
    pokolenia = pd.read_csv(katalog / "pokolenia.csv")
    agenci = pd.read_csv(katalog / "agenci.csv")
    dylematy = pd.read_csv(katalog / "dylematy.csv")
    return konfiguracja, pokolenia, agenci, dylematy


def porownaj_przebiegi(
    katalog_a: str | Path,
    katalog_b: str | Path,
) -> list[dict[str, object]]:
    """Porównuje deterministyczne pliki CSV dwóch przebiegów."""
    katalog_a = Path(katalog_a)
    katalog_b = Path(katalog_b)
    wyniki: list[dict[str, object]] = []
    for nazwa in ("pokolenia.csv", "agenci.csv", "dylematy.csv"):
        sciezka_a = katalog_a / nazwa
        sciezka_b = katalog_b / nazwa
        komplet = sciezka_a.is_file() and sciezka_b.is_file()
        identyczne = komplet and sciezka_a.read_bytes() == sciezka_b.read_bytes()
        if not komplet:
            szczegoly = "brak pliku w co najmniej jednym przebiegu"
        else:
            szczegoly = "identyczne" if identyczne else "rozne"
        wyniki.append(
            _wynik_kontroli(
                f"powtarzalnosc_{nazwa.removesuffix('.csv')}",
                identyczne,
                szczegoly,
            )
        )
    return wyniki


def _sprawdz_transfery_do_celu(
    agenci: pd.DataFrame,
    dylematy: pd.DataFrame,
    *,
    akcja: int,
    kolumna_liczby: str,
    kolumna_wartosci: str,
    nazwa_kontroli: str,
) -> dict[str, object]:
    """Porównuje pomiary odbiorców lub ofiar z udanymi zdarzeniami."""
    indeks_agentow = pd.MultiIndex.from_frame(agenci[["pokolenie", "id"]])
    udane = dylematy[dylematy["akcja"].eq(akcja) & dylematy["wynik"].eq(SUKCES)]
    agregaty = udane.groupby(["pokolenie", "id_celu"])["kwota"].agg(
        liczba="size",
        wartosc="sum",
    )
    agregaty.index = agregaty.index.set_names(["pokolenie", "id"])
    znane_cele = bool(agregaty.index.isin(indeks_agentow).all())
    agregaty = agregaty.reindex(indeks_agentow, fill_value=0)

    zgodna_liczba = (
        agenci[kolumna_liczby].to_numpy() == agregaty["liczba"].to_numpy()
    )
    zgodna_wartosc = np.isclose(
        agenci[kolumna_wartosci].to_numpy(),
        agregaty["wartosc"].to_numpy(),
    )
    zgodne = zgodna_liczba & zgodna_wartosc
    return _wynik_kontroli(
        nazwa_kontroli,
        znane_cele and bool(zgodne.all()),
        f"bledni_agenci={int((~zgodne).sum())}, znane_cele={znane_cele}",
    )


def sprawdz_przebieg(
    konfiguracja: dict,
    pokolenia: pd.DataFrame,
    agenci: pd.DataFrame,
    dylematy: pd.DataFrame,
) -> list[dict[str, object]]:
    """Wykonuje kontrole jakości jednego przebiegu."""
    wyniki = [
        _sprawdz_kolumny("pokolenia", pokolenia, KOLUMNY_POKOLENIA),
        _sprawdz_kolumny("agenci", agenci, KOLUMNY_AGENCI),
        _sprawdz_kolumny("dylematy", dylematy, KOLUMNY_DYLEMATY),
    ]
    if not all(wynik["poprawna"] for wynik in wyniki):
        return wyniki

    if pokolenia.empty or agenci.empty:
        wyniki.append(
            _wynik_kontroli(
                "niepuste_tabele",
                False,
                f"pokolenia={len(pokolenia)}, agenci={len(agenci)}",
            )
        )
        return wyniki

    cfg = konfiguracja["konfiguracja"]
    liczba_agentow = int(cfg["populacja"]["N"])
    liczba_krokow = int(cfg["przebieg"]["kroki"])
    planowane = int(cfg["przebieg"]["pokolenia"])
    prog_kwalifikacji = float(cfg["ewolucja"]["T"])
    kwota_dzielenia = float(cfg["koszty"]["s"])
    maksymalna_kradziez = float(cfg["koszty"]["x"])

    brakujace = (
        int(pokolenia[list(KOLUMNY_POKOLENIA)].isna().sum().sum())
        + int(agenci[list(KOLUMNY_AGENCI)].isna().sum().sum())
        + int(dylematy[list(KOLUMNY_DYLEMATY)].isna().sum().sum())
    )
    wyniki.append(
        _wynik_kontroli("brakujace_wartosci", brakujace == 0, str(brakujace))
    )

    poprawna_roznorodnosc = (
        np.isfinite(pokolenia["roznorodnosc"])
        & pokolenia["roznorodnosc"].ge(0)
    )
    wyniki.append(
        _wynik_kontroli(
            "roznorodnosc_skonczona_i_nieujemna",
            bool(poprawna_roznorodnosc.all()),
            f"bledne_pokolenia={int((~poprawna_roznorodnosc).sum())}",
        )
    )

    rownanie_d = agenci["S"] + agenci["K"] + agenci["N_o"] == agenci["D"]
    wyniki.append(
        _wynik_kontroli(
            "S_plus_K_plus_N_o_rowne_D",
            bool(rownanie_d.all()),
            f"bledne_wiersze={int((~rownanie_d).sum())}",
        )
    )

    oczekiwane_pokolenia = list(range(len(pokolenia)))
    numery_pokolen = pokolenia["pokolenie"].tolist()
    numery_agentow = sorted(agenci["pokolenie"].unique().tolist())
    poprawne_numery = (
        numery_pokolen == oczekiwane_pokolenia
        and numery_agentow == oczekiwane_pokolenia
    )
    wyniki.append(
        _wynik_kontroli(
            "ciaglosc_numerow_pokolen",
            poprawne_numery,
            f"pokolenia={numery_pokolen}, agenci={numery_agentow}",
        )
    )

    unikalni_agenci = not agenci.duplicated(["pokolenie", "id"]).any()
    wyniki.append(
        _wynik_kontroli(
            "unikalny_agent_w_pokoleniu",
            unikalni_agenci,
            f"duplikaty={int(agenci.duplicated(['pokolenie', 'id']).sum())}",
        )
    )

    suma_d = int(agenci["D"].sum())
    wyniki.append(
        _wynik_kontroli(
            "liczba_dylematow_rowna_sumie_D",
            len(dylematy) == suma_d,
            f"wiersze={len(dylematy)}, suma_D={suma_d}",
        )
    )

    liczebnosci = agenci.groupby("pokolenie").size()
    wyniki.append(
        _wynik_kontroli(
            "N_agentow_w_pokoleniu",
            bool((liczebnosci == liczba_agentow).all()),
            f"oczekiwano={liczba_agentow}, min={liczebnosci.min()}, max={liczebnosci.max()}",
        )
    )

    kolumny_moralnosci = ["S", "K", "N_o", "D"]
    sumy_agentow = (
        agenci.groupby("pokolenie")[kolumny_moralnosci]
        .sum()
        .reindex(pokolenia["pokolenie"])
        .reset_index(drop=True)
    )
    zgodne_sumy = sumy_agentow.eq(
        pokolenia[kolumny_moralnosci].reset_index(drop=True)
    )
    wyniki.append(
        _wynik_kontroli(
            "sumy_agentow_rowne_pokoleniom",
            bool(zgodne_sumy.all().all()),
            f"bledne_pola={int((~zgodne_sumy).sum().sum())}",
        )
    )

    kolumny_transferow = [
        "liczba_otrzymanych_darow",
        "wartosc_otrzymanych_darow",
        "liczba_kradziezy_jako_ofiara",
        "wartosc_kradziezy_jako_ofiara",
    ]
    nieujemne_transfery = agenci[kolumny_transferow].ge(0)
    wyniki.append(
        _wynik_kontroli(
            "nieujemne_transfery",
            bool(nieujemne_transfery.all().all()),
            f"bledne_pola={int((~nieujemne_transfery).sum().sum())}",
        )
    )
    wyniki.append(
        _sprawdz_transfery_do_celu(
            agenci,
            dylematy,
            akcja=DZIEL,
            kolumna_liczby="liczba_otrzymanych_darow",
            kolumna_wartosci="wartosc_otrzymanych_darow",
            nazwa_kontroli="otrzymane_dary_zgodne_ze_zdarzeniami",
        )
    )
    wyniki.append(
        _sprawdz_transfery_do_celu(
            agenci,
            dylematy,
            akcja=KRADNIJ,
            kolumna_liczby="liczba_kradziezy_jako_ofiara",
            kolumna_wartosci="wartosc_kradziezy_jako_ofiara",
            nazwa_kontroli="kradzieze_ofiar_zgodne_ze_zdarzeniami",
        )
    )

    sumy_kwalifikacji = (
        agenci.groupby("pokolenie")["kwalifikacja"]
        .sum()
        .reindex(pokolenia["pokolenie"])
        .reset_index(drop=True)
    )
    zgodna_kwalifikacja = sumy_kwalifikacji.eq(
        pokolenia["zakwalifikowani"].reset_index(drop=True)
    )
    wyniki.append(
        _wynik_kontroli(
            "kwalifikacja_rowna_sumie_agentow",
            bool(zgodna_kwalifikacja.all()),
            f"bledne_pokolenia={int((~zgodna_kwalifikacja).sum())}",
        )
    )

    if unikalni_agenci:
        d_agentow = agenci.set_index(["pokolenie", "id"])["D"]
        liczba_dylematow_agenta = (
            dylematy.groupby(["pokolenie", "id"])
            .size()
            .reindex(d_agentow.index, fill_value=0)
        )
        zgodne_d_agentow = liczba_dylematow_agenta.eq(d_agentow)
        bledne_d_agentow = int((~zgodne_d_agentow).sum())
    else:
        zgodne_d_agentow = pd.Series([False])
        bledne_d_agentow = len(agenci)
    wyniki.append(
        _wynik_kontroli(
            "liczba_dylematow_rowna_D_agenta",
            bool(zgodne_d_agentow.all()),
            f"bledni_agenci={bledne_d_agentow}",
        )
    )

    zywi = agenci["przyczyna_smierci"] == PRZYCZYNA_KONIEC_POKOLENIA
    zmarli = agenci["przyczyna_smierci"] == PRZYCZYNA_WYCZERPANIE
    prawidlowe_przyczyny = zywi | zmarli
    wyniki.append(
        _wynik_kontroli(
            "dozwolone_przyczyny_smierci",
            bool(prawidlowe_przyczyny.all()),
            f"bledne={int((~prawidlowe_przyczyny).sum())}",
        )
    )

    dozyli = (
        agenci.loc[zywi]
        .groupby("pokolenie")
        .size()
        .reindex(pokolenia["pokolenie"], fill_value=0)
        .to_numpy()
    )
    zgodne_przezycie = dozyli == pokolenia["zywi_koniec"].to_numpy()
    wyniki.append(
        _wynik_kontroli(
            "przyczyna_2_rowna_zywi_koniec",
            bool(zgodne_przezycie.all()),
            f"bledne_pokolenia={int((~zgodne_przezycie).sum())}",
        )
    )

    poprawne_kroki_smierci = (zywi & agenci["krok_smierci"].eq(liczba_krokow)) | (
        zmarli & agenci["krok_smierci"].between(0, liczba_krokow - 1)
    )
    wyniki.append(
        _wynik_kontroli(
            "zakres_kroku_smierci",
            bool(poprawne_kroki_smierci.all()),
            f"zywi={liczba_krokow}, zmarli=0..{liczba_krokow - 1}, "
            f"bledne={int((~poprawne_kroki_smierci).sum())}",
        )
    )

    poprawne_zasoby = (zywi & agenci["zasoby_koniec"].gt(0)) | (
        zmarli & agenci["zasoby_koniec"].le(0)
    )
    wyniki.append(
        _wynik_kontroli(
            "zasoby_zgodne_ze_smiercia",
            bool(poprawne_zasoby.all()),
            f"bledne={int((~poprawne_zasoby).sum())}",
        )
    )

    logiczne = pd.api.types.is_bool_dtype(
        pokolenia["awaryjny"]
    ) and pd.api.types.is_bool_dtype(agenci["kwalifikacja"])
    wyniki.append(
        _wynik_kontroli("wartosci_logiczne", bool(logiczne), "True/False")
    )

    oczekiwana_kwalifikacja = zywi & agenci["zasoby_koniec"].ge(prog_kwalifikacji)
    zgodna_regula_kwalifikacji = agenci["kwalifikacja"].eq(
        oczekiwana_kwalifikacja
    )
    wyniki.append(
        _wynik_kontroli(
            "kwalifikacja_zgodna_z_regula",
            bool(zgodna_regula_kwalifikacji.all()),
            f"bledni_agenci={int((~zgodna_regula_kwalifikacji).sum())}",
        )
    )

    awaryjne = pokolenia["awaryjny"]
    awaryjne_bez_zakwalifikowanych = pokolenia.loc[
        awaryjne, "zakwalifikowani"
    ].eq(0)
    wyniki.append(
        _wynik_kontroli(
            "wariant_awaryjny_bez_zakwalifikowanych",
            bool(awaryjne_bez_zakwalifikowanych.all()),
            f"bledne_pokolenia={int((~awaryjne_bez_zakwalifikowanych).sum())}",
        )
    )

    zasoby_dylematu = dylematy["zasoby_agenta"].ge(kwota_dzielenia) & dylematy[
        "zasoby_celu"
    ].gt(0)
    wyniki.append(
        _wynik_kontroli(
            "zasoby_w_dylemacie",
            bool(zasoby_dylematu.all()),
            f"bledne_wiersze={int((~zasoby_dylematu).sum())}",
        )
    )

    oczekiwana_klasa = np.select(
        [dylematy["akcja"].eq(DZIEL), dylematy["akcja"].eq(KRADNIJ)],
        ["dzielenie", "kradziez"],
        default="obojetna",
    )
    zgodna_klasa = dylematy["klasa"].eq(oczekiwana_klasa)
    wyniki.append(
        _wynik_kontroli(
            "klasa_zgodna_z_akcja",
            bool(zgodna_klasa.all()),
            f"bledne_wiersze={int((~zgodna_klasa).sum())}",
        )
    )

    dozwolone_akcje = dylematy["akcja"].between(0, LICZBA_AKCJI - 1)
    dozwolone_wyniki = dylematy["wynik"].isin(
        [SUKCES, PORAZKA, NIEWYKONANA, WYKONANA]
    )
    wyniki.append(
        _wynik_kontroli(
            "dozwolone_akcje_i_wyniki",
            bool((dozwolone_akcje & dozwolone_wyniki).all()),
            f"bledne_wiersze={int((~(dozwolone_akcje & dozwolone_wyniki)).sum())}",
        )
    )

    udane_dzielenie = dylematy["akcja"].eq(DZIEL) & dylematy["wynik"].eq(SUKCES)
    zgodna_kwota_dzielenia = np.isclose(
        dylematy.loc[udane_dzielenie, "kwota"], kwota_dzielenia
    )
    wyniki.append(
        _wynik_kontroli(
            "kwota_udanego_dzielenia",
            bool(zgodna_kwota_dzielenia.all()),
            f"bledne_wiersze={int((~zgodna_kwota_dzielenia).sum())}",
        )
    )

    kradzieze = dylematy["akcja"].eq(KRADNIJ)
    kwoty_kradziezy = dylematy.loc[kradzieze, "kwota"]
    sukces_kradziezy = dylematy.loc[kradzieze, "wynik"].eq(SUKCES)
    kwota_w_zakresie = kwoty_kradziezy.ge(0) & kwoty_kradziezy.le(
        maksymalna_kradziez
    )
    # Sukces oznacza pomyślne losowanie, ale ofiara może nie mieć już zasobów,
    # jeśli wcześniej w tym samym kroku opróżnił ją inny agent. Dlatego sukces
    # może przenieść 0, natomiast wynik inny niż sukces musi mieć kwotę 0.
    kwota_zgodna_z_wynikiem = sukces_kradziezy | np.isclose(
        kwoty_kradziezy, 0.0
    )
    poprawne_kradzieze = kwota_w_zakresie & kwota_zgodna_z_wynikiem
    wyniki.append(
        _wynik_kontroli(
            "kwota_kradziezy",
            bool(poprawne_kradzieze.all()),
            f"bledne_wiersze={int((~poprawne_kradzieze).sum())}",
        )
    )

    niewykonane = dylematy["wynik"].eq(NIEWYKONANA)
    zerowa_kwota_niewykonanych = np.isclose(
        dylematy.loc[niewykonane, "kwota"], 0.0
    )
    wyniki.append(
        _wynik_kontroli(
            "zerowa_kwota_niewykonanej",
            bool(zerowa_kwota_niewykonanych.all()),
            f"bledne_wiersze={int((~zerowa_kwota_niewykonanych).sum())}",
        )
    )

    wymarcie = int(pokolenia.iloc[-1]["zywi_koniec"]) == 0
    poprawna_liczba_pokolen = len(pokolenia) == planowane or (
        len(pokolenia) < planowane and wymarcie
    )
    zera_przed_koncem = pokolenia.iloc[:-1]["zywi_koniec"].eq(0).any()
    poprawny_koniec = poprawna_liczba_pokolen and not zera_przed_koncem
    wyniki.append(
        _wynik_kontroli(
            "zakonczenie_przebiegu",
            poprawny_koniec,
            f"zapisane={len(pokolenia)}, planowane={planowane}, "
            f"wymarcie={wymarcie}, zero_przed_koncem={zera_przed_koncem}",
        )
    )
    return wyniki


def przygotuj_pokolenia(
    konfiguracja: dict,
    pokolenia: pd.DataFrame,
) -> pd.DataFrame:
    """Dodaje parametry przebiegu oraz podstawowe miary pokolenia."""
    cfg = konfiguracja["konfiguracja"]
    wynik = pokolenia.copy()
    wynik.insert(0, "ziarno", int(cfg["przebieg"]["ziarno"]))
    wynik.insert(1, "alfa", float(cfg["zasoby"]["alfa"]))
    mianownik = wynik["D"].where(wynik["D"] > 0)
    wynik["M_pop"] = (wynik["S"] - wynik["K"]) / mianownik
    wynik["rodzice"] = np.where(
        wynik["awaryjny"],
        wynik["zywi_koniec"],
        wynik["zakwalifikowani"],
    )
    return wynik


def _zapisz_wykres(
    dane: pd.DataFrame,
    kolumna: str,
    etykieta_y: str,
    sciezka: Path,
) -> None:
    """Zapisuje zmianę jednej miary w kolejnych pokoleniach."""
    fig, os = plt.subplots(figsize=(10, 6))
    os.plot(dane["pokolenie"], dane[kolumna], linewidth=1.2)
    os.set_xlabel("Pokolenie")
    os.set_ylabel(etykieta_y)
    os.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(sciezka, dpi=150)
    plt.close(fig)


def analizuj(
    katalog_przebiegu: str | Path,
    katalog_analizy: str | Path,
    *,
    katalog_porownania: str | Path | None = None,
) -> bool:
    """Analizuje jeden przebieg oraz zapisuje raport, miary i wykresy."""
    katalog_przebiegu = Path(katalog_przebiegu)
    katalog_analizy = Path(katalog_analizy)
    katalog_analizy.mkdir(parents=True, exist_ok=True)

    brakujace = [
        plik for plik in PLIKI_PRZEBIEGU if not (katalog_przebiegu / plik).is_file()
    ]
    raport = [
        _wynik_kontroli(
            "kompletnosc_plikow",
            not brakujace,
            "komplet" if not brakujace else ", ".join(brakujace),
        )
    ]
    if brakujace:
        pd.DataFrame(raport).to_csv(
            katalog_analizy / "kontrola_jakosci.csv", index=False
        )
        return False

    konfiguracja, pokolenia, agenci, dylematy = wczytaj_przebieg(katalog_przebiegu)
    raport.extend(sprawdz_przebieg(konfiguracja, pokolenia, agenci, dylematy))
    if katalog_porownania is not None:
        raport.extend(porownaj_przebiegi(katalog_przebiegu, katalog_porownania))
    tabela_raportu = pd.DataFrame(raport)
    tabela_raportu.to_csv(katalog_analizy / "kontrola_jakosci.csv", index=False)

    dane_pokolen = przygotuj_pokolenia(konfiguracja, pokolenia)
    dane_pokolen.to_csv(katalog_analizy / "pokolenia_analiza.csv", index=False)
    _zapisz_wykres(
        dane_pokolen,
        "przezywalnosc",
        "Przeżywalność",
        katalog_analizy / "przezywalnosc.png",
    )
    _zapisz_wykres(
        dane_pokolen,
        "M_pop",
        "Moralność populacji",
        katalog_analizy / "moralnosc_populacji.png",
    )
    _zapisz_wykres(
        dane_pokolen,
        "rodzice",
        "Liczba rodziców",
        katalog_analizy / "liczba_rodzicow.png",
    )
    return bool(tabela_raportu["poprawna"].all())


def main(argv: list[str] | None = None) -> int:
    """Uruchamia kontrolę jakości z wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Kontrola jakosci i podstawowa analiza jednego przebiegu."
    )
    parser.add_argument(
        "przebieg", type=Path, help="katalog z czterema plikami przebiegu"
    )
    parser.add_argument(
        "--wyjscie",
        type=Path,
        default=Path("wyniki_analizy"),
        help="katalog bazowy raportow i wykresow",
    )
    parser.add_argument(
        "--porownaj",
        type=Path,
        help="drugi katalog przebiegu do kontroli powtarzalnosci NF-03",
    )
    argumenty = parser.parse_args(argv)
    znacznik_czasu = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nazwa_analizy = f"{argumenty.przebieg.name}_{znacznik_czasu}"
    katalog_analizy = argumenty.wyjscie / nazwa_analizy
    poprawne = analizuj(
        argumenty.przebieg,
        katalog_analizy,
        katalog_porownania=argumenty.porownaj,
    )
    print(f"Raport zapisano w: {katalog_analizy}")
    print("Kontrola jakosci: " + ("OK" if poprawne else "BLEDY"))
    return 0 if poprawne else 1


if __name__ == "__main__":
    raise SystemExit(main())
