"""Testy minimalnej walidacji konfiguracji wymaganej w Lab02."""

import pytest

from moralnosc.konfiguracja import wczytaj_konfiguracje


def utworz_plik_toml(tmp_path, *, n=6, h=2, w=3, p=0.5, x=4.0, c_k=1.0):
    """Tworzy minimalny plik TOML potrzebny do testów walidacji.

    Fixture pytest ``tmp_path`` wskazuje osobny katalog tymczasowy dla każdego
    wykonania testu. Dzięki temu testy nie tworzą plików w repozytorium i nie
    wpływają na siebie nawzajem. Parametry nazwane pozwalają łatwo podmienić
    jedną badaną wartość, zachowując pozostałe jako poprawne.
    """
    # Operator / obiektu Path dokleja nazwę pliku do ścieżki katalogu.
    sciezka = tmp_path / "konfiguracja.toml"

    # f-string wstawia wartości parametrów do tekstu w składni TOML.
    sciezka.write_text(
        f"""
[populacja]
N = {n}

[plansza]
H = {h}
W = {w}

[koszty]
p = {p}
x = {x}
c_k = {c_k}
""",
        encoding="utf-8",
    )
    # Test otrzymuje ścieżkę, którą przekaże funkcji wczytującej konfigurację.
    return sciezka


def test_wczytuje_toml_jako_zagniezdzony_slownik(tmp_path):
    # Przygotowanie: zapisujemy poprawną konfigurację z wartościami domyślnymi
    # funkcji pomocniczej.
    sciezka = utworz_plik_toml(tmp_path)

    # Wykonanie: wczytujemy zapisany plik tak samo jak plik użytkownika.
    cfg = wczytaj_konfiguracje(sciezka)

    # Sprawdzenie: nagłówki TOML stają się kluczami pierwszego poziomu, a pola
    # umieszczone pod nagłówkami tworzą zagnieżdżone słowniki.
    assert cfg["populacja"]["N"] == 6
    assert cfg["plansza"] == {"H": 2, "W": 3}
    assert cfg["koszty"] == {"p": 0.5, "x": 4.0, "c_k": 1.0}


def test_akceptuje_jednego_agenta(tmp_path):
    """Dolna dozwolona granica populacji N = 1 przechodzi walidacje."""
    sciezka = utworz_plik_toml(tmp_path, n=1)

    cfg = wczytaj_konfiguracje(sciezka)

    assert cfg["populacja"]["N"] == 1


# Ten sam test zostanie wykonany dwa razy: dla N równego 0 oraz dla N równego
# 7. Plansza 2 x 3 ma sześć pól, więc wartości sprawdzają obie granice warunku.
@pytest.mark.parametrize("n", [0, 7])
def test_odrzuca_liczbe_agentow_spoza_planszy(tmp_path, n):
    # Pozostałe parametry są poprawne, dlatego ewentualny błąd powinien wynikać
    # wyłącznie z niedozwolonej liczby agentów.
    sciezka = utworz_plik_toml(tmp_path, n=n)

    # pytest.raises wymaga wystąpienia AssertionError. Parametr match dodatkowo
    # sprawdza komunikat błędu; zapis \. oznacza dosłowną kropkę w wyrażeniu
    # regularnym. Dzięki temu komunikat musi wskazać parametr populacja.N.
    with pytest.raises(AssertionError, match=r"populacja\.N"):
        wczytaj_konfiguracje(sciezka)


def test_odrzuca_nieoplacalna_kradziez(tmp_path):
    # Dla tych danych p * x - c_k = 0.5 * 2.0 - 1.0 = 0. Kontrakt wymaga
    # wartości ściśle większej od zera, więc przypadek graniczny jest błędny.
    sciezka = utworz_plik_toml(tmp_path, p=0.5, x=2.0, c_k=1.0)

    # Oczekujemy błędu, którego komunikat wymienia wszystkie trzy parametry
    # tworzące sprawdzany warunek. Fragmenty .* oznaczają dowolny tekst między
    # kolejnymi nazwami parametrów.
    with pytest.raises(AssertionError, match=r"koszty\.p.*koszty\.x.*koszty\.c_k"):
        wczytaj_konfiguracje(sciezka)
