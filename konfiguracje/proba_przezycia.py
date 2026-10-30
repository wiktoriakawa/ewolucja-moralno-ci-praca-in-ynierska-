"""Krótkie przebiegi próbne przed pilotażem (zadanie 15, instrukcja krok 6).

Pomagają wybrać E0 i zakres α: dla każdej pary (E0, α) z siatki skrypt
uruchamia kilka pokoleń na podstawie pliku bazowego i wypisuje, ilu agentów
dożyło końca pokolenia, ilu zostało rodzicami i czy zadziałał wariant
awaryjny – albo w którym kroku populacja wymarła. Niczego nie zapisuje;
to narzędzie tymczasowe, do czasu gotowego uruchom_przebieg.

Uruchomienie z katalogu kod/:
    PYTHONPATH=src python konfiguracje/proba_przezycia.py konfiguracje/bazowa.toml
"""

import copy
import sys
import time

import numpy as np

from moralnosc import ewolucja, petla
from moralnosc.konfiguracja import wczytaj_konfiguracje

SIATKA_E0 = [20.0, 30.0, 50.0, 100.0]
SIATKA_ALFA = [0.3, 0.5, 0.7, 0.85, 1.0, 1.5, 3.0]
POKOLENIA = 5      # tyle wystarczy, żeby zobaczyć, czy selekcja w ogóle rusza
ZIARNA = [1, 2]    # dwa ziarna, żeby nie wyciągać wniosków z jednego losowania


def jedno_pokolenie(stan: Stan, cfg: dict, rng: np.random.Generator) -> int | None:
    """Wykonuje wszystkie kroki jednego pokolenia.

    Zwraca None, gdy ktoś dożył końca pokolenia, albo wartość stan.krok
    w chwili, gdy zginął ostatni agent.
    """
    stan.krok = 0
    for _ in range(cfg["przebieg"]["kroki"]):
        if not petla.krok(stan, cfg, rng):
            return stan.krok
    return None


def przebieg_probny(cfg: dict, pokolenia: int, ziarno: int) -> list[dict]:
    """Uruchamia co najwyżej `pokolenia` pokoleń i zwraca słownik na pokolenie.

    Klucze słownika: "zywi" (żywi na koniec pokolenia), "rodzice"
    (zakwalifikowani), "awaryjny" (bool), "wymarcie" (wynik jedno_pokolenie).
    Po wymarciu przebieg się kończy – ostatni słownik ma "wymarcie" różne od None,
    a "rodzice" i "awaryjny" mogą wtedy wynosić 0 i False.
    """
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


def main() -> None:
    baza = wczytaj_konfiguracje(sys.argv[1])
    N = baza["populacja"]["N"]
    print(f"N = {N}, kroki = {baza['przebieg']['kroki']}, pokoleń: {POKOLENIA}, "
          f"ziarna: {ZIARNA}; (A) = wariant awaryjny – nikt nie osiągnął T")
    for E0 in SIATKA_E0:
        for alfa in SIATKA_ALFA:
            cfg = copy.deepcopy(baza)  # kopia, żeby nie zmieniać bazy między próbami
            cfg["ewolucja"]["E0"] = E0
            cfg["zasoby"]["alfa"] = alfa
            cfg["ewolucja"]["T"] = E0
            for ziarno in ZIARNA:
                start = time.perf_counter()
                wiersze = przebieg_probny(cfg, POKOLENIA, ziarno)
                czas = (time.perf_counter() - start) / len(wiersze)
                opis = " | ".join(
                    f"wymarli (krok {w['wymarcie']})" if w["wymarcie"] is not None
                    else f"{w['zywi']} żyw., {w['rodzice']} rodz.{' (A)' if w['awaryjny'] else ''}"
                    for w in wiersze
                )
                print(f"E0={E0:<6} α={alfa:<6} z={ziarno}  {czas:4.1f} s/pok.  {opis}")


if __name__ == "__main__":
    main()