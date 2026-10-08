"""
Testa a predicao conformal.

Os testes centrais sao os de COBERTURA: por construcao, sob permutabilidade,
o CQR deve cobrir >= 1-alpha MESMO quando o modelo base e pessimo (estreito
demais ou largo demais). Se esses testes falham, a "garantia" e falsa.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from focus.conformal import (
    aci_intervalos,
    cqr_ajuste,
    cqr_multiplos_niveis,
    escores_cqr,
    ordenar_quantis,
    quantil_conformal,
)


# ---------------------------------------------------------------------------
# quantil conformal: a correcao de amostra finita
# ---------------------------------------------------------------------------
def test_quantil_conformal_usa_o_indice_ceil_1_menos_alpha_vezes_n_mais_1():
    escores = np.arange(1.0, 20.0)            # n = 19  ->  k = ceil(0.9*20) = 18
    assert quantil_conformal(escores, alpha=0.10) == 18.0


def test_quantil_conformal_n_pequeno_devolve_maximo_quando_k_igual_n():
    escores = np.arange(1.0, 10.0)            # n = 9   ->  k = ceil(0.9*10) = 9
    assert quantil_conformal(escores, alpha=0.10) == 9.0


def test_quantil_conformal_amostra_pequena_demais_devolve_infinito():
    """Com n=5 e alpha=0.10, k = ceil(5.4) = 6 > n: nao ha garantia finita a dar."""
    assert quantil_conformal(np.arange(5.0), alpha=0.10) == float("inf")


def test_quantil_conformal_valida_entradas():
    with pytest.raises(ValueError):
        quantil_conformal(np.array([]), 0.1)
    with pytest.raises(ValueError):
        quantil_conformal(np.arange(10.0), 1.5)


def test_escores_cqr_sinal_dentro_e_fora():
    esc = escores_cqr(
        q_inf=np.array([-1.0, -1.0, -1.0]), q_sup=np.array([1.0, 1.0, 1.0]),
        y=np.array([0.0, 3.0, -2.0]),
    )
    assert esc[0] < 0                      # dentro
    assert esc[1] == pytest.approx(2.0)    # 2 acima do topo
    assert esc[2] == pytest.approx(1.0)    # 1 abaixo da base


# ---------------------------------------------------------------------------
# a garantia de cobertura (o coracao do modulo)
# ---------------------------------------------------------------------------
def _cobertura_media(largura_base: float, alpha=0.10, n_cal=200, n_rep=400, seed=0):
    """Repete o experimento: calibra num conjunto, mede a cobertura num teste
    GRANDE independente. Dados i.i.d. N(0,1) = permutaveis."""
    rng = np.random.default_rng(seed)
    coberturas = []
    for _ in range(n_rep):
        y_cal = rng.normal(size=n_cal)
        y_te = rng.normal(size=2000)
        Q = cqr_ajuste(np.full(n_cal, -largura_base), np.full(n_cal, largura_base), y_cal, alpha)
        dentro = (y_te >= -largura_base - Q) & (y_te <= largura_base + Q)
        coberturas.append(dentro.mean())
    return float(np.mean(coberturas))


def test_cqr_cobre_o_nominal_mesmo_com_modelo_base_estreito_demais():
    """Base cobre so ~24% (+-0.3): o CQR precisa alargar ate ~90%."""
    cob = _cobertura_media(largura_base=0.3, alpha=0.10)
    assert cob >= 0.90 - 0.01, f"cobertura {cob:.3f} abaixo da garantia de 0.90"
    assert cob <= 0.93


def test_cqr_encolhe_modelo_base_largo_demais_sem_sobre_cobrir():
    """Base cobre ~100% (+-5): Q negativo, e a cobertura volta para ~90%, nao 100%."""
    cob = _cobertura_media(largura_base=5.0, alpha=0.10)
    assert 0.89 <= cob <= 0.93, f"cobertura {cob:.3f}: o CQR deveria encolher, nao manter o excesso"


def test_cqr_ajuste_negativo_quando_o_base_sobre_cobre():
    rng = np.random.default_rng(1)
    y = rng.normal(size=500)
    assert cqr_ajuste(np.full(500, -5.0), np.full(500, 5.0), y, alpha=0.10) < 0


def test_cqr_multiplos_niveis_calibra_todos_os_niveis_simultaneamente():
    rng = np.random.default_rng(2)
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    n_cal, n_te = 1500, 20000
    y_cal, y_te = rng.normal(size=n_cal), rng.normal(size=n_te)
    base = stats.norm.ppf(taus) * 0.4          # base pessimo: sigma 0.4 (real = 1.0)
    q_cal, q_te = np.tile(base, (n_cal, 1)), np.tile(base, (n_te, 1))

    aj = cqr_multiplos_niveis(q_cal, y_cal, q_te, taus)
    for tau in (0.05, 0.10, 0.25):
        j = int(np.flatnonzero(np.isclose(taus, tau))[0])
        js = int(np.flatnonzero(np.isclose(taus, 1 - tau))[0])
        cob = np.mean((y_te >= aj[:, j]) & (y_te <= aj[:, js]))
        assert cob == pytest.approx(1 - 2 * tau, abs=0.02), f"nivel {1 - 2 * tau:.0%}: cobertura {cob:.3f}"


def test_cqr_multiplos_niveis_nao_produz_quantis_cruzados():
    rng = np.random.default_rng(3)
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    q_cal = np.sort(rng.normal(size=(300, len(taus))), axis=1)
    q_te = np.sort(rng.normal(size=(50, len(taus))), axis=1)
    aj = cqr_multiplos_niveis(q_cal, rng.normal(size=300), q_te, taus)
    assert (np.diff(aj, axis=1) >= -1e-12).all()


def test_cqr_multiplos_niveis_exige_taus_simetricos():
    taus = [0.1, 0.5, 0.8]     # 0.8 nao e o espelho de 0.1 (seria 0.9)
    with pytest.raises(ValueError, match="simetrico"):
        cqr_multiplos_niveis(np.zeros((10, 3)), np.zeros(10), np.zeros((5, 3)), taus)


def test_ordenar_quantis_elimina_cruzamento():
    q = np.array([[3.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert (ordenar_quantis(q) == np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])).all()


# ---------------------------------------------------------------------------
# ACI: adaptacao sob deriva, e ausencia de vazamento do futuro
# ---------------------------------------------------------------------------
def _fluxo(T=2500, delay=3, seed=0, escala_pos=3.0):
    """Fluxo cuja escala TRIPLICA na 2a metade (choque de regime). Os escores de
    calibracao vem da regiao de escala 1."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(size=T // 2), rng.normal(scale=escala_pos, size=T - T // 2)])
    t_dec = np.datetime64("2010-01-01") + np.arange(T).astype("timedelta64[D]")
    t_des = t_dec + np.timedelta64(delay, "D")
    q_inf, q_sup = np.full(T, -1.645), np.full(T, 1.645)   # base calibrado p/ escala 1
    esc_cal = escores_cqr(np.full(1000, -1.645), np.full(1000, 1.645), rng.normal(size=1000))
    return y, t_dec, t_des, q_inf, q_sup, esc_cal


def test_aci_recupera_cobertura_sob_choque_de_escala():
    y, t_dec, t_des, q_inf, q_sup, esc_cal = _fluxo()
    Q = quantil_conformal(esc_cal, 0.10)
    dentro_estatico = (y >= q_inf - Q) & (y <= q_sup + Q)

    lo, hi, _ = aci_intervalos(q_inf, q_sup, y, esc_cal, t_dec, t_des, alpha=0.10, gamma=0.05)
    dentro_aci = (y >= lo) & (y <= hi)

    meio = len(y) // 2
    assert dentro_estatico[meio:].mean() < 0.70, "o choque deveria derrubar a cobertura do conformal estatico"
    assert dentro_aci[meio:].mean() > dentro_estatico[meio:].mean() + 0.10, "ACI deveria recuperar cobertura"
    assert dentro_aci.mean() == pytest.approx(0.90, abs=0.04), "cobertura de longo prazo do ACI perto do nominal"


def test_aci_sem_vazamento_do_futuro_alpha_inicial_antes_do_primeiro_desfecho():
    """Nenhuma linha decidida ANTES de o primeiro desfecho se realizar pode ter
    alpha_t diferente do alpha inicial -- ainda nao ha feedback."""
    y, t_dec, t_des, q_inf, q_sup, esc_cal = _fluxo(T=200, delay=10)
    _, _, alpha_usado = aci_intervalos(q_inf, q_sup, y, esc_cal, t_dec, t_des, alpha=0.10, gamma=0.05)
    antes = t_dec < t_des.min()
    assert antes.sum() > 0
    assert (alpha_usado[antes] == 0.10).all()


def test_aci_resultado_alinhado_a_ordem_original_de_entrada():
    y, t_dec, t_des, q_inf, q_sup, esc_cal = _fluxo(T=300, delay=2)
    lo_a, hi_a, al_a = aci_intervalos(q_inf, q_sup, y, esc_cal, t_dec, t_des, alpha=0.10, gamma=0.03)

    perm = np.random.default_rng(9).permutation(len(y))
    lo_b, hi_b, al_b = aci_intervalos(
        q_inf[perm], q_sup[perm], y[perm], esc_cal, t_dec[perm], t_des[perm], alpha=0.10, gamma=0.03)
    assert np.allclose(lo_b, lo_a[perm])
    assert np.allclose(hi_b, hi_a[perm])
    assert np.allclose(al_b, al_a[perm])


def test_aci_com_gamma_zero_reproduz_o_conformal_estatico():
    y, t_dec, t_des, q_inf, q_sup, esc_cal = _fluxo(T=400)
    lo, hi, alpha_usado = aci_intervalos(q_inf, q_sup, y, esc_cal, t_dec, t_des, alpha=0.10, gamma=0.0)
    Q = quantil_conformal(esc_cal, 0.10)
    assert (alpha_usado == 0.10).all()
    assert np.allclose(lo, q_inf - Q) and np.allclose(hi, q_sup + Q)


def test_aci_com_grupos_usa_os_escores_de_cada_grupo():
    """Com gamma=0 (sem adaptacao), cada linha deve receber o Q do SEU grupo."""
    T = 100
    t_dec = np.datetime64("2010-01-01") + np.arange(T).astype("timedelta64[D]")
    t_des = t_dec + np.timedelta64(2, "D")
    y = np.zeros(T)
    q_inf, q_sup = np.full(T, -1.0), np.full(T, 1.0)
    esc = {0: np.linspace(-1, 1, 50), 1: np.linspace(-1, 5, 50)}      # grupo 1 exige Q maior
    grupo = np.where(np.arange(T) < 50, 0, 1)

    lo, hi, _ = aci_intervalos(q_inf, q_sup, y, esc, t_dec, t_des, alpha=0.10, gamma=0.0, grupo=grupo)
    Q0, Q1 = quantil_conformal(esc[0], 0.10), quantil_conformal(esc[1], 0.10)
    assert Q1 > Q0
    assert np.allclose(hi[:50], 1.0 + Q0) and np.allclose(hi[50:], 1.0 + Q1)
