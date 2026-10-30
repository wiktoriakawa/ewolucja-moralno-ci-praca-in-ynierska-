"""Generuje pliki przebiegów pilotażowych P1–P3 z konfiguracji bazowej.

Każdy plik to kopia bazowa.toml, w której zmieniono tylko klucze z planu
(ziarno, liczba pokoleń, α, katalog wyników). Generator sprawdza, że plik
wczytany z dysku to dokładnie bazowa + te zmiany i że przechodzi walidację
z wczytaj_konfiguracje. Wartości zmieniamy w bazowa.toml albo w stałych
poniżej, a potem uruchamiamy generator ponownie – plików w pilotaz/ nie
edytujemy ręcznie.

Uruchomienie z katalogu kod/:
    PYTHONPATH=src python konfiguracje/generuj_pilotaz.py
"""

import copy
import json
import re
from pathlib import Path

from moralnosc.konfiguracja import wczytaj_konfiguracje

KATALOG = Path(__file__).parent  # kod/konfiguracje/, niezależnie od miejsca uruchomienia
BAZA = KATALOG / "bazowa.toml"
WYJSCIE = KATALOG / "pilotaz"
NAGLOWEK = "# Wygenerowano z bazowa.toml przez generuj_pilotaz.py – nie edytuj ręcznie.\n"

# Ziarna pilotażu są rozłączne z ziarnami próby przeżycia (1–5), na których
# dobrano E0 i α, więc wartości nie są dopasowane do wyników samego pilotażu.
ZIARNA_P1 = [101, 102, 103, 104, 105]
ZIARNA_P2 = [201, 202, 203]
ZIARNO_P3 = 301

POKOLENIA_P2 = 50
POKOLENIA_P3 = 10

# Poziomy α do P2 z próby przeżycia (E0 = T = 50; polecenia w proba_przezycia.py).
# Siatka (ziarna 1–2, 5 pokoleń): przy α ≤ 0,5 pokolenie zerowe wymiera, a przy
# α ≥ 1,0 po wąskim gardle w pokoleniu zerowym przeżywa prawie każdy.
# Test odporności (ziarna 1–5, 100 pokoleń): przy α = 0,7 wymarły 4 z 5 przebiegów,
# przy 0,85 przetrwało 5 z 5 (ziarno 1 ledwo: wariant awaryjny w 74 pokoleniach).
# 0,85 – niedobór z selekcją; 1,0 – wartość pośrednia; 3,0 – obfitość.
POZIOMY_ALFA = [("niedobor", 0.85), ("posrednia", 1.0), ("obfitosc", 3.0)]


def _zmiany(nazwa: str, ziarno: int, pokolenia: int) -> dict[tuple[str, str], object]:
    """Zmiany wspólne dla każdego przebiegu: ziarno, liczba pokoleń i własny katalog."""
    return {
        ("przebieg", "ziarno"): ziarno,
        ("przebieg", "pokolenia"): pokolenia,
        ("zapis", "katalog"): f"wyniki/{nazwa}",
    }


def plan_przebiegow() -> list[tuple[str, dict[tuple[str, str], object]]]:
    """Zwraca listę par (nazwa pliku bez .toml, zmiany względem bazowej).

    P1 – pokolenie zerowe: 1 pokolenie, 5 ziaren (5 plików).
    P2 – 3 poziomy α × 3 ziarna, POKOLENIA_P2 pokoleń (9 plików).
    P3 – powtarzalność (NF-03): dwa pliki z tym samym ziarnem, różne tylko
    katalogiem; ich pliki CSV mają być identyczne (2 pliki).
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
    """Zapisuje wartość w składni TOML: int → 7, float → 0.5, str → "tekst"."""
    # bool przed int: w Pythonie bool jest podklasą int (True == 1),
    # a w TOML piszemy true i false.
    if isinstance(wartosc, bool):
        return "true" if wartosc else "false"
    if isinstance(wartosc, int):
        return str(wartosc)
    if isinstance(wartosc, float):
        return repr(wartosc)  # zawsze z kropką albo wykładnikiem: 1.0, 0.85, 1e-05
    if isinstance(wartosc, str):
        return json.dumps(wartosc)  # cudzysłów i znaki specjalne zgodne z TOML
    raise TypeError(f"nieobsługiwany typ wartości: {type(wartosc).__name__}")


def podmien(tekst: str, klucz: str, wartosc: object) -> str:
    """Zamienia linię `klucz = …` w tekście TOML na `klucz = <wartość w TOML>`.

    Wymaga dokładnie jednego trafienia – inaczej ValueError z nazwą klucza.
    """
    nowa_linia = f"{klucz} = {jako_toml(wartosc)}"
    # ^ i MULTILINE: klucz musi zaczynać linię, więc wzorzec dla "x" nie trafi
    # w "c_max = …" ani "w_max = …"; wielkość liter ma znaczenie ("r" ≠ "R").
    wzorzec = rf"^{re.escape(klucz)}\s*=.*$"
    # Zamiennik jako funkcja: w zwykłym napisie re.subn interpretowałby ukośniki.
    nowy_tekst, liczba = re.subn(wzorzec, lambda _: nowa_linia, tekst, flags=re.MULTILINE)
    if liczba != 1:
        raise ValueError(f"klucz {klucz!r}: {liczba} trafień zamiast 1")
    return nowy_tekst


def main() -> None:
    tekst_bazy = BAZA.read_text(encoding="utf-8")
    baza = wczytaj_konfiguracje(BAZA)  # bazowa też musi przejść walidację
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
        # Najpierw plik roboczy .tmp: do pilotaz/*.toml trafia tylko plik po kontroli,
        # a przy błędzie zostaje .tmp do obejrzenia.
        roboczy = sciezka.with_suffix(".tmp")
        roboczy.write_text(NAGLOWEK + tekst, encoding="utf-8")
        # Kontrola: plik wczytany z dysku = bazowa + dokładnie te zmiany.
        assert wczytaj_konfiguracje(roboczy) == oczekiwana, f"{nazwa}: plik niezgodny z planem"
        roboczy.replace(sciezka)
        print(f"{sciezka.name}: " + ", ".join(f"{k} = {v}" for (_, k), v in zmiany.items()))
    print(f"Zapisano {len(plan)} plików w {WYJSCIE}")


if __name__ == "__main__":
    main()
