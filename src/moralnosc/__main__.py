"""Interfejs wiersza poleceń pakietu moralnosc."""

import argparse
from pathlib import Path
import sys

from moralnosc.konfiguracja import BladKonfiguracji, wczytaj_konfiguracje
from moralnosc.petla import uruchom_przebieg
from moralnosc.seria import BladSerii, wykonaj_serie, wczytaj_serie
from moralnosc.zapis import przygotuj_zapis


def utworz_parser() -> argparse.ArgumentParser:
    """Tworzy parser poleceń programu."""
    parser = argparse.ArgumentParser(prog="python -m moralnosc")
    podpolecenia = parser.add_subparsers(dest="polecenie", required=True)

    przebieg = podpolecenia.add_parser(
        "przebieg",
        help="uruchom jeden przebieg na podstawie pliku TOML",
    )
    przebieg.add_argument("konfiguracja", type=Path, help="sciezka do pliku TOML")

    seria = podpolecenia.add_parser(
        "seria",
        help="uruchom serię wariantów i ziaren na podstawie pliku TOML",
    )
    seria.add_argument("definicja", type=Path, help="sciezka do definicji serii TOML")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Obsługuje argumenty, uruchamia przebieg i zapisuje jego wyniki."""
    argumenty = utworz_parser().parse_args(argv)

    if argumenty.polecenie == "seria":
        try:
            definicja = wczytaj_serie(argumenty.definicja)
            statusy = wykonaj_serie(definicja)
        except (BladSerii, BladKonfiguracji, OSError) as blad:
            print(f"Blad serii: {blad}", file=sys.stderr)
            return 2

        for status in statusy:
            komunikat = f"{status['id']}: {status['status']}"
            if status["blad"]:
                komunikat += f" ({status['blad']})"
            print(komunikat)
        return 1 if any(status["status"] == "blad" for status in statusy) else 0

    try:
        cfg = wczytaj_konfiguracje(argumenty.konfiguracja)
    except (BladKonfiguracji, OSError) as blad:
        print(f"Blad konfiguracji: {blad}", file=sys.stderr)
        return 2

    callback = przygotuj_zapis(cfg)
    stan = uruchom_przebieg(cfg, po_pokoleniu=callback)

    print(f"Zakonczono przebieg na pokoleniu {stan.pokolenie}.")
    print(f"Wyniki zapisano w: {cfg['zapis']['katalog']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
