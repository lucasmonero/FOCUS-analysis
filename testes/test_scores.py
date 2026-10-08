"""
Testa as regras de pontuacao probabilistica.

O teste mais importante e o do CRPS contra a forma fechada da normal: ele
valida de uma vez a decomposicao quantilica inteira, incluindo o fator 2 --
um erro nesse fator passaria despercebido em qualquer comparacao RELATIVA
entre modelos, mas tornaria o numero absoluto sem sentido.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from focus.scores import (
    cobertura,
    crps_por_quantis,
    interval_score,
    pinball_loss,
    pinball_por_quantil,
    sharpness,
)


def test_pinball_na_mediana_e_metade_do_erro_absoluto():
    y = np.array([1.0, -2.0, 3.5])
    q = np.array([0.0, 0.0, 0.0])
    assert pinball_loss(y, q, 0.5) == pytest.approx(0.5 * np.abs(y - q))


def test_pinball_e_assimetrico_no_sentido_certo():
    """Em tau=0.9, subestimar (quantil abaixo do realizado) deve custar
    bem mais caro que superestimar pela mesma margem."""
    y = np.array([10.0])
    subestimou = pinball_loss(y, np.array([9.0]), 0.9)[0]   # q < y
    superestimou = pinball_loss(y, np.array([11.0]), 0.9)[0]  # q > y
    assert subestimou == pytest.approx(0.9)
    assert superestimou == pytest.approx(0.1)
    assert subestimou > superestimou


def test_pinball_tau_invalido_levanta_erro():
    with pytest.raises(ValueError):
        pinball_loss(np.array([1.0]), np.array([1.0]), 0.0)


def test_pinball_e_minimizado_pelo_quantil_verdadeiro():
    """Propriedade que define a regra: a perda esperada e minima quando o
    quantil predito e o quantil verdadeiro da distribuicao."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000)
    tau = 0.8
    q_verdadeiro = stats.norm.ppf(tau)
    perda_certa = pinball_loss(y, np.full_like(y, q_verdadeiro), tau).mean()
    for desvio in (-0.3, -0.1, 0.1, 0.3):
        perda_errada = pinball_loss(y, np.full_like(y, q_verdadeiro + desvio), tau).mean()
        assert perda_certa < perda_errada


def test_crps_bate_com_a_forma_fechada_da_normal():
    """CRPS(N(mu, sigma), y) tem forma fechada; a aproximacao por grade de
    quantis deve convergir para ela."""
    rng = np.random.default_rng(1)
    n = 500
    mu, sigma = 0.3, 1.7
    y = rng.normal(mu, sigma, size=n)

    taus = np.linspace(0.001, 0.999, 999)
    quantis = np.tile(stats.norm.ppf(taus, loc=mu, scale=sigma), (n, 1))
    crps_aprox = crps_por_quantis(y, quantis, taus)

    z = (y - mu) / sigma
    crps_exato = np.mean(
        sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    )
    assert crps_aprox == pytest.approx(crps_exato, rel=0.01)


def test_crps_premia_a_densidade_correta():
    """Uma densidade com a variancia certa deve pontuar melhor (menor CRPS)
    que uma larga demais ou estreita demais."""
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, size=4000)
    taus = np.linspace(0.01, 0.99, 99)

    def crps_com_sigma(sigma):
        q = np.tile(stats.norm.ppf(taus, loc=0.0, scale=sigma), (len(y), 1))
        return crps_por_quantis(y, q, taus)

    assert crps_com_sigma(1.0) < crps_com_sigma(0.4)
    assert crps_com_sigma(1.0) < crps_com_sigma(2.5)


def test_pinball_por_quantil_valida_forma_da_matriz():
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="forma"):
        pinball_por_quantil(y, np.zeros((2, 3)), [0.1, 0.9])


def test_cobertura_e_sharpness():
    y = np.array([0.0, 5.0, 10.0])
    inf = np.array([-1.0, -1.0, -1.0])
    sup = np.array([1.0, 1.0, 20.0])
    assert cobertura(y, inf, sup) == pytest.approx(2 / 3)
    assert sharpness(inf, sup) == pytest.approx((2 + 2 + 21) / 3)


def test_interval_score_penaliza_muito_mais_errar_do_que_alargar():
    """Alargar um pouco o intervalo deve custar menos do que deixar o
    realizado de fora -- e o que impede o score de premiar intervalos
    artificialmente estreitos."""
    y = np.array([0.0])
    alpha = 0.1

    estreito_mas_cobre = interval_score(y, np.array([-0.5]), np.array([0.5]), alpha)
    largo_e_cobre = interval_score(y, np.array([-2.0]), np.array([2.0]), alpha)
    estreito_e_erra = interval_score(y, np.array([1.0]), np.array([1.5]), alpha)

    assert estreito_mas_cobre < largo_e_cobre, "cobrindo, mais estreito deve pontuar melhor"
    assert estreito_e_erra > largo_e_cobre, "deixar o realizado de fora deve ser o pior caso"


def test_interval_score_alpha_invalido():
    with pytest.raises(ValueError):
        interval_score(np.array([0.0]), np.array([-1.0]), np.array([1.0]), alpha=1.5)


def test_crps_por_linha_media_bate_com_crps_agregado():
    from focus.scores import crps_por_linha
    rng = np.random.default_rng(7)
    y = rng.normal(size=300)
    taus = np.linspace(0.05, 0.95, 19)
    q = np.tile(stats.norm.ppf(taus), (300, 1))
    assert crps_por_linha(y, q, taus).mean() == pytest.approx(crps_por_quantis(y, q, taus))


def test_crps_por_linha_tem_uma_entrada_por_observacao_e_e_nao_negativa():
    from focus.scores import crps_por_linha
    y = np.array([0.0, 1.0, -2.0])
    taus = [0.25, 0.5, 0.75]
    r = crps_por_linha(y, np.zeros((3, 3)), taus)
    assert r.shape == (3,) and (r >= 0).all()
