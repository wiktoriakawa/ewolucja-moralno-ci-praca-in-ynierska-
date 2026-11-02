"""Kontrola jakości i podstawowa analiza jednego przebiegu symulacji."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PLIKI_PRZEBIEGU = (
    "konfiguracja.json",
    "pokolenia.csv",
    "agenci.csv",
    "dylematy.csv",
)

KOLUMNY_POKOLENIA = {
    "pokolenie",
    "zywi_koniec",
    "przezywalnosc",
    "zakwalifikowani",
    "awaryjny",
    "S",
    "K",
    "N_o",
    "D",
    *(f"akcja_{akcja}" for akcja in range(8)),
}

KOLUMNY_AGENCI = {
    "pokolenie",
    "id",
    "id_rodzica",
    "S",
    "K",
    "N_o",
    "D",
    "zasoby_koniec",
    "kwalifikacja",
    "krok_smierci",
    "przyczyna_smierci",
}

KOLUMNY_DYLEMATY = {
    "pokolenie",
    "krok",
    "id",
    "id_celu",
    "zasoby_agenta",
    "zasoby_celu",
    "akcja",
    "klasa",
    "wynik",
    "kwota",
}


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
    wymagane: set[str],
) -> dict[str, object]:
    """Sprawdza obecność wymaganych kolumn w jednej tabeli."""
    brakujace = sorted(wymagane - set(dane.columns))
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

    brakujace = (
        int(pokolenia[list(KOLUMNY_POKOLENIA)].isna().sum().sum())
        + int(agenci[list(KOLUMNY_AGENCI)].isna().sum().sum())
        + int(dylematy[list(KOLUMNY_DYLEMATY)].isna().sum().sum())
    )
    wyniki.append(
        _wynik_kontroli("brakujace_wartosci", brakujace == 0, str(brakujace))
    )

    rownanie_d = agenci["S"] + agenci["K"] + agenci["N_o"] == agenci["D"]
    wyniki.append(
        _wynik_kontroli(
            "S_plus_K_plus_N_o_rowne_D",
            bool(rownanie_d.all()),
            f"bledne_wiersze={int((~rownanie_d).sum())}",
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

    cfg = konfiguracja["konfiguracja"]
    liczba_agentow = int(cfg["populacja"]["N"])
    liczebnosci = agenci.groupby("pokolenie").size()
    wyniki.append(
        _wynik_kontroli(
            "N_agentow_w_pokoleniu",
            bool((liczebnosci == liczba_agentow).all()),
            f"oczekiwano={liczba_agentow}, min={liczebnosci.min()}, max={liczebnosci.max()}",
        )
    )

    dozyli = (
        agenci.loc[agenci["przyczyna_smierci"] == 2]
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

    liczba_krokow = int(cfg["przebieg"]["kroki"])
    zakres_smierci = agenci["krok_smierci"].between(0, liczba_krokow)
    wyniki.append(
        _wynik_kontroli(
            "zakres_kroku_smierci",
            bool(zakres_smierci.all()),
            f"zakres=0..{liczba_krokow}, bledne={int((~zakres_smierci).sum())}",
        )
    )

    logiczne = pd.api.types.is_bool_dtype(
        pokolenia["awaryjny"]
    ) and pd.api.types.is_bool_dtype(agenci["kwalifikacja"])
    wyniki.append(
        _wynik_kontroli("wartosci_logiczne", bool(logiczne), "True/False")
    )

    planowane = int(cfg["przebieg"]["pokolenia"])
    zakonczone_wczesniej = len(pokolenia) < planowane
    wymarcie = int(pokolenia.iloc[-1]["zywi_koniec"]) == 0
    poprawny_koniec = not zakonczone_wczesniej or wymarcie
    wyniki.append(
        _wynik_kontroli(
            "zakonczenie_przebiegu",
            poprawny_koniec,
            f"zapisane={len(pokolenia)}, planowane={planowane}, wymarcie={wymarcie}",
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


def analizuj(katalog_przebiegu: str | Path, katalog_analizy: str | Path) -> bool:
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
    argumenty = parser.parse_args(argv)
    znacznik_czasu = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nazwa_analizy = f"{argumenty.przebieg.name}_{znacznik_czasu}"
    katalog_analizy = argumenty.wyjscie / nazwa_analizy
    poprawne = analizuj(argumenty.przebieg, katalog_analizy)
    print(f"Raport zapisano w: {katalog_analizy}")
    print("Kontrola jakosci: " + ("OK" if poprawne else "BLEDY"))
    return 0 if poprawne else 1


if __name__ == "__main__":
    raise SystemExit(main())
