"""Dzielenie, kradziez i koszty metabolizmu agentow."""

import numpy as np

from moralnosc.stan import Stan


def wykonaj_dzielenie(stan: Stan, i: int, j: int, s: float) -> bool:
    """Przekazuje kwote s od agenta i do j, jesli dawca ma dosc zasobow."""
    if stan.zasoby_agentow[i] < s:
        return False

    stan.zasoby_agentow[i] -= s
    stan.zasoby_agentow[j] += s
    return True


def wykonaj_kradziez(
    stan: Stan,
    i: int,
    j: int,
    c_k: float,
    p: float,
    x: float,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """Wykonuje zweryfikowana przez petle probe kradziezy.

    Funkcja zaklada, ze agent j nadal znajduje sie przed agentem i.
    Po rozpoczeciu prawidlowej proby zawsze odejmuje sprawcy koszt c_k,
    a nastepnie wykonuje jedno losowanie. Kradziez jest udana, gdy
    rng.random() < p.

    Przy sukcesie przenosi od ofiary do sprawcy kwote nie wieksza niz x
    i nie wieksza niz nieujemne zasoby ofiary. Jesli ofiara ma zasoby
    rowne zero albo ujemne, kwota transferu wynosi 0.0.

    Zwraca:
        Pare (sukces, kwota), gdzie sukces informuje o wyniku losowania,
        a kwota oznacza rzeczywiscie przeniesione zasoby.
    """
    stan.zasoby_agentow[i] -= c_k

    if rng.random() >= p:
        return False, 0.0

    dostepne = max(0.0, stan.zasoby_agentow[j])
    kwota = min(x, dostepne)
    stan.zasoby_agentow[i] += kwota
    stan.zasoby_agentow[j] -= kwota
    return True, float(kwota)


def nalicz_koszty(stan: Stan, m: float) -> None:
    """Odejmuje koszt metabolizmu m od zasobow wszystkich zywych agentow."""
    stan.zasoby_agentow[stan.zywy] -= m
