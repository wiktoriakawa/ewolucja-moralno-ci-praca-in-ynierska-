"""Interfejs wiersza poleceń pakietu moralnosc."""

import argparse
from pathlib import Path
import sys

from moralnosc.konfiguracja import BladKonfiguracji, wczytaj_konfiguracje
from moralnosc.petla import uruchom_przebieg
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Obsługuje argumenty, uruchamia przebieg i zapisuje jego wyniki."""
    argumenty = utworz_parser().parse_args(argv)
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
