"""Testy zapisu konfiguracji i wyników przebiegu."""

import csv
from datetime import datetime
import json

import numpy as np

from moralnosc import zapis
from moralnosc.pomiar import PRZYCZYNA_KONIEC_POKOLENIA, pusty_pomiar
from moralnosc.stan import PUSTE, pusty_stan


def _cfg(katalog):
    """Minimalna konfiguracja potrzebna modułowi zapisu."""
    return {
        "przebieg": {"ziarno": 67, "pokolenia": 2, "kroki": 3},
        "ewolucja": {"T": 5.0},
        "zapis": {"katalog": str(katalog)},
    }


def _stan_i_pomiar():
    """Końcowy stan pokolenia dwóch agentów i odpowiadający mu pomiar."""
    stan = pusty_stan(H=2, W=2, N=2, L=48)
    stan.pokolenie = 0
    stan.ids[:] = [10, 20]
    stan.rodzic[:] = [PUSTE, PUSTE]
    stan.zywy[:] = True
    stan.zasoby_agentow[:] = [6.0, 4.0]

    pomiar = pusty_pomiar(N=2)
    pomiar.S[:] = [1, 0]
    pomiar.K[:] = [0, 1]
    pomiar.N_o[:] = [2, 3]
    pomiar.D[:] = [3, 4]
    pomiar.krok_smierci[:] = 3
    pomiar.przyczyna_smierci[:] = PRZYCZYNA_KONIEC_POKOLENIA
    pomiar.akcje_licznik[:] = np.arange(8)
    pomiar.zdarzenia.append(
        {
            "pokolenie": 0,
            "krok": 1,
            "id": 10,
            "id_celu": 20,
            "zasoby_agenta": 6.0,
            "zasoby_celu": 4.0,
            "akcja": 5,
            "klasa": "dzielenie",
            "wynik": "sukces",
            "kwota": 1.0,
        }
    )
    return stan, pomiar


def _wczytaj_csv(sciezka):
    """Wczytuje CSV jako listę słowników, tak jak zrobi to analiza."""
    with sciezka.open(encoding="utf-8", newline="") as plik:
        return list(csv.DictReader(plik))


def test_zapisz_konfiguracje_zapisuje_cfg_i_metadane(tmp_path, monkeypatch):
    """JSON zawiera pełną konfigurację, datę i wersję kodu."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(zapis, "_commit_repozytorium", lambda: "abc123")

    sciezka = zapis.zapisz_konfiguracje(cfg, tmp_path)

    with sciezka.open(encoding="utf-8") as plik:
        dane = json.load(plik)
    assert dane["konfiguracja"] == cfg
    assert dane["metadane"]["commit"] == "abc123"
    datetime.fromisoformat(dane["metadane"]["data_uruchomienia"])


def test_callback_zapisuje_poprawne_dane_pokolenia(tmp_path):
    """Callback zapisuje agregaty, agentów i zdarzenia do właściwych CSV."""
    cfg = _cfg(tmp_path)
    stan, pomiar = _stan_i_pomiar()
    callback = zapis.przygotuj_zapis(cfg)

    # Maska obejmuje awaryjnie obu rodziców, ale próg T spełnia tylko agent 10.
    callback(stan, pomiar, np.array([True, True]), True)

    pokolenia = _wczytaj_csv(tmp_path / "pokolenia.csv")
    assert len(pokolenia) == 1
    assert pokolenia[0] == {
        "pokolenie": "0",
        "zywi_koniec": "2",
        "przezywalnosc": "1.0",
        "zakwalifikowani": "1",
        "awaryjny": "True",
        "S": "1",
        "K": "1",
        "N_o": "5",
        "D": "7",
        **{f"akcja_{akcja}": str(akcja) for akcja in range(8)},
    }

    agenci = _wczytaj_csv(tmp_path / "agenci.csv")
    assert len(agenci) == 2
    assert agenci[0]["id"] == "10"
    assert agenci[0]["id_rodzica"] == str(PUSTE)
    assert agenci[0]["kwalifikacja"] == "True"
    assert agenci[0]["S"] == "1"
    assert agenci[0]["krok_smierci"] == "3"
    assert agenci[1]["id"] == "20"
    assert agenci[1]["kwalifikacja"] == "False"
    assert agenci[1]["K"] == "1"

    dylematy = _wczytaj_csv(tmp_path / "dylematy.csv")
    assert len(dylematy) == 1
    assert dylematy[0]["id"] == "10"
    assert dylematy[0]["id_celu"] == "20"
    assert dylematy[0]["klasa"] == "dzielenie"
    assert dylematy[0]["wynik"] == "sukces"
    assert dylematy[0]["kwota"] == "1.0"


def test_callback_dopisuje_pokolenia_bez_powtarzania_naglowka(tmp_path):
    """Kolejne wywołania callbacku dopisują dane do tych samych plików."""
    cfg = _cfg(tmp_path)
    stan, pomiar = _stan_i_pomiar()
    callback = zapis.przygotuj_zapis(cfg)

    callback(stan, pomiar, np.array([True, False]), False)
    stan.pokolenie = 1
    stan.ids[:] = [30, 40]
    stan.rodzic[:] = [10, 20]
    pomiar.zdarzenia[0]["pokolenie"] = 1
    callback(stan, pomiar, np.array([True, False]), False)

    pokolenia = _wczytaj_csv(tmp_path / "pokolenia.csv")
    agenci = _wczytaj_csv(tmp_path / "agenci.csv")
    dylematy = _wczytaj_csv(tmp_path / "dylematy.csv")
    assert [wiersz["pokolenie"] for wiersz in pokolenia] == ["0", "1"]
    assert [wiersz["id"] for wiersz in agenci] == ["10", "20", "30", "40"]
    assert [wiersz["pokolenie"] for wiersz in dylematy] == ["0", "1"]
