"""
Testa os baselines B0/B0'/B1 da escada probabilistica.

O teste mais importante e o de RECUPERACAO do fator k: se o ajuste nao
recupera uma escala conhecida, o concorrente central da Etapa 5 estaria
mal-especificado, e toda a comparacao com o modelo complexo seria injusta.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from focus.probabilistico import (
    TAUS_PADRAO,
    ajustar_fator_sigma,
    quantis_historicos,
    quantis_normais,
)
from focus.scores import crps_por_quantis


def test_taus_padrao_e_simetrico_e_contem_a_mediana():
    taus = np.array(TAUS_PADRAO)
    assert np.isclose(taus, 0.5).any()
    for t in taus:
        assert np.isclose(taus, 1 - t).any(), f"falta o espelho de {t}"


def test_quantis_normais_tem_a_forma_e_a_escala_corretas():
    q = quantis_normais(np.array([1.0, 2.0]), [0.05, 0.5, 0.95], k=1.0)
    assert q.shape == (2, 3)
    assert q[0, 1] == pytest.approx(0.0)
    assert q[0, 2] == pytest.approx(stats.norm.ppf(0.95))
    assert q[1, 2] == pytest.approx(2 * stats.norm.ppf(0.95))     # sigma dobrado
    # k escala linearmente
    assert quantis_normais(np.array([1.0]), [0.95], k=3.0)[0, 0] == pytest.approx(3 * stats.norm.ppf(0.95))


def test_quantis_normais_com_sigma_zero_e_intervalo_de_largura_zero():
    q = quantis_normais(np.array([0.0]), [0.05, 0.95])
    assert (q == 0).all()


@pytest.mark.parametrize("k_verdadeiro", [0.5, 1.0, 2.5, 4.7])
def test_ajustar_fator_sigma_recupera_a_escala_verdadeira(k_verdadeiro):
    """Erros gerados com desvio k*sigma: o ajuste deve devolver ~k."""
    rng = np.random.default_rng(0)
    n = 6000
    sigma = rng.uniform(0.05, 0.4, size=n)
    erro = rng.normal(0.0, k_verdadeiro * sigma)
    assert ajustar_fator_sigma(erro, sigma) == pytest.approx(k_verdadeiro, rel=0.05)


def test_ajuste_do_fator_reduz_o_crps_frente_ao_sigma_publicado():
    """Se o sigma publicado e pequeno demais por um fator 3, o sigma reescalado
    precisa pontuar MELHOR (menor CRPS) que o publicado, fora da amostra."""
    rng = np.random.default_rng(1)
    n = 4000
    sigma = rng.uniform(0.05, 0.3, size=n)
    erro = rng.normal(0.0, 3.0 * sigma)
    tr, te = slice(0, 2000), slice(2000, None)
    k = ajustar_fator_sigma(erro[tr], sigma[tr])
    taus = list(TAUS_PADRAO)
    crps_publicado = crps_por_quantis(erro[te], quantis_normais(sigma[te], taus, 1.0), taus)
    crps_reescalado = crps_por_quantis(erro[te], quantis_normais(sigma[te], taus, k), taus)
    assert crps_reescalado < crps_publicado


def test_ajustar_fator_ignora_linhas_com_sigma_zero_ou_nan():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.1, 0.3, size=500)
    erro = rng.normal(0.0, 2.0 * sigma)
    sigma_sujo = np.concatenate([sigma, [0.0, 0.0, np.nan]])
    erro_sujo = np.concatenate([erro, [5.0, -5.0, 1.0]])
    assert ajustar_fator_sigma(erro_sujo, sigma_sujo) == pytest.approx(ajustar_fator_sigma(erro, sigma), rel=1e-6)


def test_ajustar_fator_amostra_pequena_levanta_erro():
    with pytest.raises(ValueError, match="pequena demais"):
        ajustar_fator_sigma(np.array([0.1, 0.2]), np.array([0.1, 0.1]))


def test_quantis_historicos_repete_os_quantis_empiricos():
    erro = np.arange(1.0, 101.0)
    q = quantis_historicos(erro, n_linhas=4, taus=[0.25, 0.5, 0.75])
    assert q.shape == (4, 3)
    assert np.allclose(q[0], np.quantile(erro, [0.25, 0.5, 0.75]))
    assert (q == q[0]).all()


def test_quantis_historicos_sem_dados_levanta_erro():
    with pytest.raises(ValueError):
        quantis_historicos(np.array([np.nan]), 3)


# ---------------------------------------------------------------------------
# escala_erros_resolvidos: onde um erro de logica vaza o futuro
# ---------------------------------------------------------------------------
import pandas as pd

from focus.probabilistico import escala_erros_resolvidos


def _serie_mensal(n=30):
    alvos = pd.date_range("2015-01-01", periods=n, freq="MS")
    decisao = alvos - pd.DateOffset(months=3)          # previsao feita 3 meses antes do alvo
    return alvos, decisao


def test_escala_nao_usa_erro_de_previsao_ainda_nao_resolvida():
    """Um erro ENORME no alvo de ago/2016 nao pode aparecer em nenhuma escala
    calculada antes de ago/2016 + defasagem."""
    alvos, decisao = _serie_mensal()
    erro = np.ones(len(alvos))
    i_bomba = int(np.flatnonzero(alvos == pd.Timestamp("2016-08-01"))[0])
    erro[i_bomba] = 1000.0

    escala = escala_erros_resolvidos(alvos, decisao, erro, n_ultimos=12, defasagem_dias=60, minimo=3)
    liberado_em = pd.Timestamp("2016-08-01") + pd.Timedelta(days=60)
    antes = (decisao < liberado_em) & np.isfinite(escala)
    assert antes.sum() > 0
    assert (escala[antes] < 10).all(), "a escala enxergou um erro de previsao ainda nao resolvida"
    depois = decisao >= liberado_em
    assert (escala[depois & np.isfinite(escala)] > 50).any(), "o erro grande deveria aparecer apos ser resolvido"


def test_escala_e_nan_sem_erros_suficientes_resolvidos():
    alvos, decisao = _serie_mensal()
    escala = escala_erros_resolvidos(alvos, decisao, np.ones(len(alvos)), minimo=6)
    assert np.isnan(escala[0])


def test_escala_alinhada_a_ordem_de_entrada():
    alvos, decisao = _serie_mensal(24)
    erro = np.arange(1.0, 25.0)
    a = escala_erros_resolvidos(alvos, decisao, erro, minimo=3)
    perm = np.random.default_rng(0).permutation(24)
    b = escala_erros_resolvidos(alvos[perm], decisao[perm], erro[perm], minimo=3)
    assert np.allclose(b, a[perm], equal_nan=True)


def test_escala_com_erros_constantes_e_essa_constante():
    alvos, decisao = _serie_mensal()
    escala = escala_erros_resolvidos(alvos, decisao, np.full(len(alvos), 2.5), minimo=3)
    assert np.allclose(escala[np.isfinite(escala)], 2.5)


def test_defasagem_maior_reduz_o_que_ja_esta_resolvido():
    alvos, decisao = _serie_mensal()
    curta = np.isfinite(escala_erros_resolvidos(alvos, decisao, np.ones(len(alvos)), defasagem_dias=0, minimo=3)).sum()
    longa = np.isfinite(escala_erros_resolvidos(alvos, decisao, np.ones(len(alvos)), defasagem_dias=200, minimo=3)).sum()
    assert longa <= curta
