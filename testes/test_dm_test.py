"""
Testa a implementacao do Diebold-Mariano + correcao HLN. O maior risco aqui
e um erro de sinal silencioso que inverteria "o modelo bate o Focus" para o
oposto -- por isso o teste de simetria (a) abaixo e o mais importante do
arquivo.
"""
from __future__ import annotations

import numpy as np
import pytest

from focus.dm_test import diebold_mariano


def _perdas_com_diferenca_clara(n=200, seed=0):
    """perda_b e sistematicamente menor que perda_a por uma margem clara
    (b e a previsao melhor)."""
    rng = np.random.default_rng(seed)
    perda_a = 1.0 + rng.normal(scale=0.1, size=n)
    perda_b = 0.5 + rng.normal(scale=0.1, size=n)
    return perda_a, perda_b


def test_d_medio_positivo_quando_a_e_pior():
    perda_a, perda_b = _perdas_com_diferenca_clara()
    resultado = diebold_mariano(perda_a, perda_b, h=3)
    assert resultado["d_medio"] > 0
    assert resultado["dm_ajustado_hln"] > 0
    assert resultado["p_valor"] < 0.01, "diferenca tao clara deveria ser altamente significativa"


def test_simetria_ao_trocar_a_e_b():
    """Trocar a ordem dos argumentos deve inverter o sinal de d_medio/dm_stat
    mas manter o mesmo p-valor -- e a checagem mais importante do arquivo:
    um bug de sinal aqui inverteria silenciosamente a conclusao do teste."""
    perda_a, perda_b = _perdas_com_diferenca_clara()
    r_ab = diebold_mariano(perda_a, perda_b, h=3)
    r_ba = diebold_mariano(perda_b, perda_a, h=3)

    assert r_ab["d_medio"] == pytest.approx(-r_ba["d_medio"])
    assert r_ab["dm_stat"] == pytest.approx(-r_ba["dm_stat"])
    assert r_ab["dm_ajustado_hln"] == pytest.approx(-r_ba["dm_ajustado_hln"])
    assert r_ab["p_valor"] == pytest.approx(r_ba["p_valor"])


def test_sem_diferenca_sistematica_nao_e_significativo():
    rng = np.random.default_rng(1)
    n = 200
    perda_a = 1.0 + rng.normal(scale=0.2, size=n)
    perda_b = 1.0 + rng.normal(scale=0.2, size=n)  # mesma media, sem vantagem real
    resultado = diebold_mariano(perda_a, perda_b, h=1)
    assert resultado["p_valor"] > 0.05


def test_h_maior_aumenta_maxlags_e_pode_reduzir_significancia():
    """Nao exige uma direcao especifica de mudanca no p-valor (depende dos
    dados), so que a funcao rode sem erro para h variados e produza graus de
    liberdade consistentes com n."""
    perda_a, perda_b = _perdas_com_diferenca_clara(n=100)
    for h in (1, 3, 6, 12):
        resultado = diebold_mariano(perda_a, perda_b, h=h)
        assert resultado["graus_liberdade"] == 99
        assert resultado["n"] == 100


def test_h_invalido_levanta_erro():
    perda_a, perda_b = _perdas_com_diferenca_clara(n=50)
    with pytest.raises(ValueError):
        diebold_mariano(perda_a, perda_b, h=0)


def test_amostra_pequena_demais_levanta_erro():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.0, 1.5], h=1)
