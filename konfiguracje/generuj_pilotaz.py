"""Generuje pliki pilotażu P1–P3 z bazowa.toml (zmienia tylko ziarno, pokolenia, α, katalog).

Każdy plik jest sprawdzany: wczytany z dysku = bazowa + dokładnie te zmiany.
Plików w pilotaz/ nie edytujemy ręcznie.

Z katalogu kod/:
    PYTHONPATH=src python konfiguracje/generuj_pilotaz.py
"""

import copy
import json
import re
from pathlib import Path

from moralnosc.konfiguracja import wczytaj_konfiguracje

KATALOG = Path(__file__).parent
BAZA = KATALOG / "bazowa.toml"
WYJSCIE = KATALOG / "pilotaz"
NAGLOWEK = "# Wygenerowano z bazowa.toml przez generuj_pilotaz.py – nie edytuj ręcznie.\n"

# rozłączne z ziarnami próby przeżycia (1–5), na których dobrano E0 i α
ZIARNA_P1 = [101, 102, 103, 104, 105]
ZIARNA_P2 = [201, 202, 203]
ZIARNO_P3 = 301

POKOLENIA_P2 = 50
POKOLENIA_P3 = 10

# Z próby przeżycia (E0 = T = 50): α ≤ 0,5 – wymarcie; α = 0,7 – 4/5 przebiegów
# wymiera w 100 pokoleniach; α = 0,85 – 5/5 przetrwało (z selekcją).
POZIOMY_ALFA = [("niedobor", 0.85), ("posrednia", 1.0), ("obfitosc", 3.0)]


def _zmiany(nazwa: str, ziarno: int, pokolenia: int) -> dict[tuple[str, str], object]:
    """Wspólne zmiany: ziarno, pokolenia, własny katalog."""
    return {
        ("przebieg", "ziarno"): ziarno,
        ("przebieg", "pokolenia"): pokolenia,
        ("zapis", "katalog"): f"wyniki/{nazwa}",
    }


def plan_przebiegow() -> list[tuple[str, dict[tuple[str, str], object]]]:
    """(nazwa, zmiany): P1 – 5 ziaren × 1 pokolenie; P2 – 3 α × 3 ziarna;
    P3 – dwa pliki z tym samym ziarnem (powtarzalność, NF-03).
    """
    plan = []
    for ziarno in ZIARNA_P1:
        nazwa = f"P1_z{ziarno}"
        plan.append((nazwa, _zmiany(nazwa, ziarno, 1)))
    for poziom, alfa in POZIOMY_ALFA:
        for ziarno in ZIARNA_P2:
            nazwa = f"P2_{poziom}_z{ziarno}"
            zmiany = _zmiany(nazwa, ziarno, POKOLENIA_P2)
            zmiany[("zasoby", "alfa")] = alfa
            plan.append((nazwa, zmiany))
    for wersja in ["a", "b"]:
        nazwa = f"P3_{wersja}"
        plan.append((nazwa, _zmiany(nazwa, ZIARNO_P3, POKOLENIA_P3)))
    return plan


def jako_toml(wartosc: object) -> str:
    """Wartość w składni TOML."""
    # bool przed int – bool jest podklasą int
    if isinstance(wartosc, bool):
        return "true" if wartosc else "false"
    if isinstance(wartosc, int):
        return str(wartosc)
    if isinstance(wartosc, float):
        return repr(wartosc)  # zawsze z kropką albo wykładnikiem
    if isinstance(wartosc, str):
        return json.dumps(wartosc)
    raise TypeError(f"nieobsługiwany typ wartości: {type(wartosc).__name__}")


def podmien(tekst: str, klucz: str, wartosc: object) -> str:
    """Podmienia linię `klucz = …`; inna liczba trafień niż 1 → ValueError."""
    nowa_linia = f"{klucz} = {jako_toml(wartosc)}"
    # ^ + MULTILINE: "x" nie trafi w "c_max = …"
    wzorzec = rf"^{re.escape(klucz)}\s*=.*$"
    # zamiennik jako funkcja – bez interpretacji ukośników
    nowy_tekst, liczba = re.subn(wzorzec, lambda _: nowa_linia, tekst, flags=re.MULTILINE)
    if liczba != 1:
        raise ValueError(f"klucz {klucz!r}: {liczba} trafień zamiast 1")
    return nowy_tekst


def main() -> None:
    tekst_bazy = BAZA.read_text(encoding="utf-8")
    baza = wczytaj_konfiguracje(BAZA)
    plan = plan_przebiegow()

    nazwy = [nazwa for nazwa, _ in plan]
    katalogi = [zmiany.get(("zapis", "katalog")) for _, zmiany in plan]
    assert len(set(nazwy)) == len(nazwy), "powtórzona nazwa pliku w planie"
    assert None not in katalogi and len(set(katalogi)) == len(katalogi), \
        "każdy przebieg potrzebuje własnego zapis.katalog"

    WYJSCIE.mkdir(exist_ok=True)
    for nazwa, zmiany in plan:
        tekst = tekst_bazy
        oczekiwana = copy.deepcopy(baza)
        for (sekcja, klucz), wartosc in zmiany.items():
            tekst = podmien(tekst, klucz, wartosc)
            oczekiwana[sekcja][klucz] = wartosc
        sciezka = WYJSCIE / f"{nazwa}.toml"
        # najpierw .tmp – do pilotaz/ trafia tylko plik po kontroli
        roboczy = sciezka.with_suffix(".tmp")
        roboczy.write_text(NAGLOWEK + tekst, encoding="utf-8")
        assert wczytaj_konfiguracje(roboczy) == oczekiwana, f"{nazwa}: plik niezgodny z planem"
        roboczy.replace(sciezka)
        print(f"{sciezka.name}: " + ", ".join(f"{k} = {v}" for (_, k), v in zmiany.items()))
    print(f"Zapisano {len(plan)} plików w {WYJSCIE}")


if __name__ == "__main__":
    main()
