"""Strumieniowy zapis konfiguracji i wyników przebiegu."""

from collections.abc import Callable
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import NDArray

from moralnosc.pomiar import KOLUMNY_DYLEMATY, Pomiar
from moralnosc.stan import LICZBA_AKCJI, Stan


KOLUMNY_POKOLENIA = (
    "pokolenie",
    "zywi_koniec",
    "przezywalnosc",
    "zakwalifikowani",
    "awaryjny",
    "S",
    "K",
    "N_o",
    "D",
    *(f"akcja_{akcja}" for akcja in range(LICZBA_AKCJI)),
)

KOLUMNY_AGENCI = (
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
)

# Typ callbacku wywoływanego przez `uruchom_przebieg` po każdym pokoleniu.
# To tylko opis interfejsu; konkretną funkcję tworzy `przygotuj_zapis` poniżej.
ObslugaPokolenia = Callable[[Stan, Pomiar, NDArray[np.bool_], bool], None]


def _commit_repozytorium() -> str:
    """Zwraca identyfikator wersji kodu, która wygenerowała wyniki.

    Identyfikator commita pozwala odróżnić przebiegi wykonane z tą samą
    konfiguracją i ziarnem, ale przez różne wersje programu. Gdy Git nie
    jest dostępny, zapisuje bezpieczną wartość ``"nieznany"``.
    """
    katalog_repo = Path(__file__).resolve().parents[2]
    try:
        wynik = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=katalog_repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "nieznany"
    return wynik.stdout.strip() if wynik.returncode == 0 else "nieznany"


def zapisz_konfiguracje(cfg: dict, katalog: str | Path) -> Path:
    """Zapisuje konfigurację i metadane przebiegu do czytelnego pliku JSON."""
    katalog = Path(katalog)
    katalog.mkdir(parents=True, exist_ok=True)
    sciezka = katalog / "konfiguracja.json"
    dane = {
        "konfiguracja": cfg,
        "metadane": {
            "data_uruchomienia": datetime.now().astimezone().isoformat(),
            "commit": _commit_repozytorium(),
        },
    }
    with sciezka.open("w", encoding="utf-8") as plik:
        json.dump(dane, plik, ensure_ascii=False, indent=2)
        plik.write("\n")
    return sciezka


def _utworz_csv(sciezka: Path, kolumny: tuple[str, ...]) -> None:
    """Rozpoczyna nowy plik CSV i zapisuje jego nagłówek."""
    with sciezka.open("w", encoding="utf-8", newline="") as plik:
        csv.DictWriter(plik, fieldnames=kolumny).writeheader()


def _dopisz_wiersze(
    sciezka: Path,
    kolumny: tuple[str, ...],
    wiersze: list[dict[str, object]],
) -> None:
    """Dopisuje wiersze do istniejącego pliku CSV."""
    if not wiersze:
        return
    with sciezka.open("a", encoding="utf-8", newline="") as plik:
        csv.DictWriter(plik, fieldnames=kolumny).writerows(wiersze)


def zapisz_pokolenie(
    stan: Stan,
    pomiar: Pomiar,
    maska_rodzicow: NDArray[np.bool_],
    awaryjny: bool,
    *,
    katalog: str | Path,
    prog_kwalifikacji: float,
) -> None:
    """Dopisuje zakończone pokolenie do trzech plików wynikowych CSV."""
    katalog = Path(katalog)
    liczba_agentow = stan.zywy.size
    zywi_koniec = int(stan.zywy.sum())
    # Maska rodziców może awaryjnie obejmować wszystkich żywych. Kolumna
    # `kwalifikacja` ma natomiast zawsze oznaczać faktyczne spełnienie progu T.
    kwalifikacja = stan.zywy & (stan.zasoby_agentow >= prog_kwalifikacji)

    wiersz_pokolenia: dict[str, object] = {
        "pokolenie": stan.pokolenie,
        "zywi_koniec": zywi_koniec,
        "przezywalnosc": zywi_koniec / liczba_agentow,
        "zakwalifikowani": int(kwalifikacja.sum()),
        "awaryjny": awaryjny,
        "S": int(pomiar.S.sum()),
        "K": int(pomiar.K.sum()),
        "N_o": int(pomiar.N_o.sum()),
        "D": int(pomiar.D.sum()),
    }
    wiersz_pokolenia.update(
        {f"akcja_{akcja}": int(licznik) for akcja, licznik in enumerate(pomiar.akcje_licznik)}
    )
    _dopisz_wiersze(
        katalog / "pokolenia.csv",
        KOLUMNY_POKOLENIA,
        [wiersz_pokolenia],
    )

    wiersze_agentow = [
        {
            "pokolenie": stan.pokolenie,
            "id": int(stan.ids[i]),
            "id_rodzica": int(stan.rodzic[i]),
            "S": int(pomiar.S[i]),
            "K": int(pomiar.K[i]),
            "N_o": int(pomiar.N_o[i]),
            "D": int(pomiar.D[i]),
            "zasoby_koniec": float(stan.zasoby_agentow[i]),
            "kwalifikacja": bool(kwalifikacja[i]),
            "krok_smierci": int(pomiar.krok_smierci[i]),
            "przyczyna_smierci": int(pomiar.przyczyna_smierci[i]),
        }
        for i in range(liczba_agentow)
    ]
    _dopisz_wiersze(katalog / "agenci.csv", KOLUMNY_AGENCI, wiersze_agentow)
    _dopisz_wiersze(
        katalog / "dylematy.csv",
        KOLUMNY_DYLEMATY,
        pomiar.zdarzenia,
    )


def przygotuj_zapis(cfg: dict) -> ObslugaPokolenia:
    """Przygotowuje katalog wyników i zwraca callback zapisu pokoleń.

    Rozpoczyna od nowa trzy pliki CSV wskazanego przebiegu. Inne pliki
    znajdujące się w katalogu pozostawia bez zmian.
    """
    katalog = Path(cfg["zapis"]["katalog"])
    katalog.mkdir(parents=True, exist_ok=True)
    zapisz_konfiguracje(cfg, katalog)
    _utworz_csv(katalog / "pokolenia.csv", KOLUMNY_POKOLENIA)
    _utworz_csv(katalog / "agenci.csv", KOLUMNY_AGENCI)
    _utworz_csv(katalog / "dylematy.csv", KOLUMNY_DYLEMATY)

    # Ta funkcja jest konkretnym callbackiem `po_pokoleniu`. Jako domknięcie
    # pamięta `katalog` i próg T z `cfg`, więc pętla przekazuje jej tylko dane
    # właśnie zakończonego pokolenia.
    def po_pokoleniu(
        stan: Stan,
        pomiar: Pomiar,
        maska_rodzicow: NDArray[np.bool_],
        awaryjny: bool,
    ) -> None:
        zapisz_pokolenie(
            stan,
            pomiar,
            maska_rodzicow,
            awaryjny,
            katalog=katalog,
            prog_kwalifikacji=cfg["ewolucja"]["T"],
        )

    # Zwracamy funkcję bez jej wywoływania. Uruchomi ją `uruchom_przebieg`
    # osobno po każdym pokoleniu.
    return po_pokoleniu
