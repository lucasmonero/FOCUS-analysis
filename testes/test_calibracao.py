"""
Testa os instrumentos de calibracao.

Os testes mais importantes sao os de PODER: um intervalo calibrado nao pode
ser rejeitado (evita acusar o Focus injustamente) e um intervalo estreito
demais TEM que ser rejeitado (senao o instrumento nao detecta o que foi
construido para detectar).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from focus.calibracao import (
    avaliar_intervalo,
    pit_normal,
    teste_christoffersen,
    teste_kupiec,
    teste_uniformidade_pit,
)


def test_kupiec_nao_rejeita_intervalo_calibrado():
    rng = np.random.default_rng(0)
    violacoes = (rng.random(2000) < 0.10).astype(float)
    r = teste_kupiec(violacoes, p_nominal=0.10)
    assert not r["rejeita_5pct"]
    assert r["taxa_violacao_observada"] == pytest.approx(0.10, abs=0.02)


def test_kupiec_rejeita_intervalo_estreito_demais():
    """Cenario central do projeto: o intervalo declara 90% mas so cobre 70%."""
    rng = np.random.default_rng(1)
    violacoes = (rng.random(1000) < 0.30).astype(float)
    r = teste_kupiec(violacoes, p_nominal=0.10)
    assert r["rejeita_5pct"]
    assert r["taxa_violacao_observada"] > 0.10


def test_kupiec_lida_com_zero_violacoes_sem_nan():
    """0*log(0) apareceria como NaN numa implementacao ingenua."""
    r = teste_kupiec(np.zeros(200), p_nominal=0.10)
    assert np.isfinite(r["lr_uc"])
    assert r["violacoes"] == 0
    assert r["rejeita_5pct"], "nenhuma violacao em 200 obs com 10% nominal deveria ser rejeitado"


def test_kupiec_valida_entradas():
    with pytest.raises(ValueError):
        teste_kupiec(np.array([]), 0.1)
    with pytest.raises(ValueError):
        teste_kupiec(np.zeros(10), p_nominal=1.5)


def test_christoffersen_detecta_agrupamento_de_violacoes():
    """Mesma taxa global de violacao, mas agrupada -- e o padrao de um
    intervalo que ignora regime. LR_ind deve rejeitar a independencia."""
    n = 600
    violacoes = np.zeros(n)
    violacoes[100:160] = 1.0   # bloco contiguo
    violacoes[400:400 + 60] = 1.0
    r = teste_christoffersen(violacoes, p_nominal=0.20)
    assert r["rejeita_independencia_5pct"]


def test_christoffersen_nao_rejeita_violacoes_independentes():
    rng = np.random.default_rng(3)
    violacoes = (rng.random(3000) < 0.10).astype(float)
    r = teste_christoffersen(violacoes, p_nominal=0.10)
    assert not r["rejeita_independencia_5pct"]


def test_pit_de_densidade_correta_e_uniforme():
    rng = np.random.default_rng(4)
    n = 3000
    mu = rng.normal(size=n)
    sigma = np.full(n, 1.5)
    y = rng.normal(mu, sigma)
    r = teste_uniformidade_pit(pit_normal(y, mu, sigma))
    assert r["calibrada_5pct"]


def test_pit_detecta_densidade_estreita_demais():
    """Se sigma declarado e menor que o real, o PIT acumula nas pontas."""
    rng = np.random.default_rng(5)
    n = 2000
    mu = np.zeros(n)
    y = rng.normal(0.0, 2.0, size=n)          # sigma real = 2.0
    pit = pit_normal(y, mu, np.full(n, 0.8))  # sigma declarado = 0.8
    r = teste_uniformidade_pit(pit)
    assert not r["calibrada_5pct"]
    # acumulo nas caudas: muitos valores proximos de 0 ou 1
    assert np.mean((pit < 0.05) | (pit > 0.95)) > 0.30


def test_pit_rejeita_sigma_nao_positivo():
    with pytest.raises(ValueError):
        pit_normal(np.array([1.0]), np.array([0.0]), np.array([0.0]))


def test_avaliar_intervalo_em_caso_calibrado_e_em_caso_estreito():
    rng = np.random.default_rng(6)
    n = 2000
    y = rng.normal(size=n)

    z90 = stats.norm.ppf(0.95)  # intervalo central de 90%
    calibrado = avaliar_intervalo(y, np.full(n, -z90), np.full(n, z90), nivel_nominal=0.90)
    assert calibrado["cobertura_observada"] == pytest.approx(0.90, abs=0.03)
    assert not calibrado["rejeita_calibracao_5pct"]

    estreito = avaliar_intervalo(y, np.full(n, -0.5), np.full(n, 0.5), nivel_nominal=0.90)
    assert estreito["cobertura_observada"] < 0.60
    assert estreito["rejeita_calibracao_5pct"]
    assert estreito["largura_media"] < calibrado["largura_media"]
