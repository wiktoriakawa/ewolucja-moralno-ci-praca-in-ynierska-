"""Krótkie przebiegi próbne do wyboru E0 i zakresu α (T = E0). Niczego nie zapisuje.

Z katalogu kod/ – siatka z 30.10.2026 (dała E0 = 50):
    PYTHONPATH=src python konfiguracje/proba_przezycia.py konfiguracje/bazowa.toml

Test odporności niedoboru (źródło komentarza przy POZIOMY_ALFA w generuj_pilotaz.py):
    PYTHONPATH=src python konfiguracje/proba_przezycia.py konfiguracje/bazowa.toml --e0 50 --alfa 0.7 0.85 --ziarna 1 2 3 4 5 --pokolenia 100
"""

import argparse
import copy
import time

import numpy as np

from moralnosc import ewolucja, petla
from moralnosc.konfiguracja import wczytaj_konfiguracje
from moralnosc.stan import Stan

# siatka z 30.10.2026
SIATKA_E0 = [20.0, 30.0, 50.0, 100.0]
SIATKA_ALFA = [0.3, 0.5, 0.7, 0.85, 1.0, 1.5, 3.0]
POKOLENIA = 5
ZIARNA = [1, 2]


def jedno_pokolenie(stan: Stan, cfg: dict, rng: np.random.Generator) -> int | None:
    """Kroki jednego pokolenia; None – ktoś przeżył, inaczej krok wymarcia."""
    stan.krok = 0
    for _ in range(cfg["przebieg"]["kroki"]):
        if not petla.krok(stan, cfg, rng):
            return stan.krok
    return None


def przebieg_probny(cfg: dict, pokolenia: int, ziarno: int) -> list[dict]:
    """Słownik na pokolenie: zywi, rodzice, awaryjny, wymarcie; kończy się po wymarciu."""
    rng = np.random.default_rng(ziarno)
    stan = ewolucja.pokolenie_zerowe(cfg, rng)
    wiersze = []
    for _ in range(pokolenia):
        wymarcie = jedno_pokolenie(stan, cfg, rng)
        zywi = int(stan.zywy.sum())
        if wymarcie is not None:
            wiersze.append({"zywi": zywi, "rodzice": 0, "awaryjny": False, "wymarcie": wymarcie})
            break
        maska, awaryjny = ewolucja.kwalifikacja(stan, cfg["ewolucja"]["T"])
        rodzice = int(maska.sum())
        wiersze.append({"zywi": zywi, "rodzice": rodzice, "awaryjny": awaryjny, "wymarcie": None})
        ewolucja.nowe_pokolenie(stan, maska, cfg, rng)
    return wiersze


def argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Próba przeżycia przed pilotażem.")
    parser.add_argument("konfiguracja", help="plik bazowy TOML")
    parser.add_argument("--e0", type=float, nargs="+", default=SIATKA_E0, help="wartości E0 (i T)")
    parser.add_argument("--alfa", type=float, nargs="+", default=SIATKA_ALFA, help="wartości α")
    parser.add_argument("--ziarna", type=int, nargs="+", default=ZIARNA, help="ziarna generatora")
    parser.add_argument("--pokolenia", type=int, default=POKOLENIA, help="liczba pokoleń")
    return parser.parse_args()


def opis_przebiegu(wiersze: list[dict]) -> str:
    """Lista pokoleń; powyżej 10 pokoleń – podsumowanie."""
    if len(wiersze) <= 10:
        return " | ".join(
            f"wymarli (krok {w['wymarcie']})" if w["wymarcie"] is not None
            else f"{w['zywi']} żyw., {w['rodzice']} rodz.{' (A)' if w['awaryjny'] else ''}"
            for w in wiersze
        )
    awaryjne = sum(w["awaryjny"] for w in wiersze)
    if wiersze[-1]["wymarcie"] is not None:
        return (f"wymarli w pokoleniu {len(wiersze) - 1} (krok {wiersze[-1]['wymarcie']}); "
                f"(A) w {awaryjne} pok.")
    ostatnie = wiersze[-10:]
    zywi = sum(w["zywi"] for w in ostatnie) / len(ostatnie)
    rodzice = sum(w["rodzice"] for w in ostatnie) / len(ostatnie)
    return (f"{len(wiersze)} pok.; ostatnie 10: śr. {zywi:.0f} żyw., {rodzice:.0f} rodz.; "
            f"(A) w {awaryjne} pok.")


def main() -> None:
    args = argumenty()
    baza = wczytaj_konfiguracje(args.konfiguracja)
    N = baza["populacja"]["N"]
    print(f"N = {N}, kroki = {baza['przebieg']['kroki']}, pokoleń: {args.pokolenia}, "
          f"ziarna: {args.ziarna}; T = E0; (A) = wariant awaryjny – nikt nie osiągnął T")
    for E0 in args.e0:
        for alfa in args.alfa:
            cfg = copy.deepcopy(baza)
            cfg["ewolucja"]["E0"] = E0
            cfg["zasoby"]["alfa"] = alfa
            cfg["ewolucja"]["T"] = E0
            for ziarno in args.ziarna:
                start = time.perf_counter()
                wiersze = przebieg_probny(cfg, args.pokolenia, ziarno)
                czas = (time.perf_counter() - start) / len(wiersze)
                print(f"E0={E0:<6} α={alfa:<6} z={ziarno}  {czas:4.1f} s/pok.  "
                      f"{opis_przebiegu(wiersze)}")


if __name__ == "__main__":
    main()
