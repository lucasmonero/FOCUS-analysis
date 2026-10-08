"""
Regras de pontuacao para previsao PROBABILISTICA.

A Etapa 4 mostrou que comparar previsoes por MAE/RMSE responde a pergunta
errada quando a entrega e uma distribuicao, e nao um ponto: MAE premia quem
acerta o centro, e e cego para quem declara honestamente a propria
incerteza. As metricas aqui sao as regras proprias ("proper scoring rules")
usadas na literatura de previsao de densidade.

Todas operam sobre quantis preditos -- nao exigem forma parametrica da
distribuicao, o que casa com a saida da regressao quantilica/conformal.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, tau: float) -> np.ndarray:
    """
    Perda pinball (quantilica) ponto a ponto para o nivel `tau`.

        L_tau(y, q) = tau * (y - q)      se y >= q
                      (1 - tau) * (q - y) se y <  q

    E assimetrica de proposito: para `tau=0.9`, subestimar (prever um quantil
    abaixo do realizado) custa 9x mais que superestimar -- e assim que o
    score força o modelo a acertar a CAUDA, nao a media.
    """
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau deve estar em (0, 1); recebido {tau}")
    dif = y - q_pred
    return np.where(dif >= 0, tau * dif, (tau - 1.0) * dif)


def pinball_por_quantil(y: np.ndarray, quantis_pred: np.ndarray, taus) -> dict[float, float]:
    """
    Perda pinball MEDIA por nivel de quantil.

    `quantis_pred` tem forma (n_obs, n_taus), na mesma ordem de `taus`.
    """
    y = np.asarray(y, dtype=float)
    quantis_pred = np.asarray(quantis_pred, dtype=float)
    taus = list(taus)
    if quantis_pred.shape != (len(y), len(taus)):
        raise ValueError(
            f"quantis_pred deve ter forma ({len(y)}, {len(taus)}); recebido {quantis_pred.shape}"
        )
    return {tau: float(pinball_loss(y, quantis_pred[:, j], tau).mean()) for j, tau in enumerate(taus)}


def crps_por_quantis(y: np.ndarray, quantis_pred: np.ndarray, taus) -> float:
    """
    CRPS (Continuous Ranked Probability Score) aproximado pela decomposicao
    quantilica:

        CRPS(F, y) = 2 * integral_0^1 L_tau(y, F^-1(tau)) d tau

    Aproximamos a integral pela media sobre a grade `taus` -- quanto mais
    densa a grade, melhor a aproximacao. E a metrica-resumo da qualidade da
    DISTRIBUICAO inteira (o analogo do MAE para previsao probabilistica):
    quanto menor, melhor, e ela so e minimizada pela densidade verdadeira.
    """
    perdas = pinball_por_quantil(y, quantis_pred, taus)
    return float(2.0 * np.mean(list(perdas.values())))


def cobertura(y: np.ndarray, q_inf: np.ndarray, q_sup: np.ndarray) -> float:
    """Fracao das observacoes que caem dentro do intervalo [q_inf, q_sup]."""
    y = np.asarray(y, dtype=float)
    dentro = (y >= np.asarray(q_inf, dtype=float)) & (y <= np.asarray(q_sup, dtype=float))
    return float(dentro.mean())


def sharpness(q_inf: np.ndarray, q_sup: np.ndarray) -> float:
    """
    Largura media do intervalo. Menor e melhor -- MAS so faz sentido ler
    junto com a cobertura: um intervalo estreitissimo que nao cobre nada nao
    vale nada. Calibracao primeiro, sharpness depois.
    """
    return float(np.mean(np.asarray(q_sup, dtype=float) - np.asarray(q_inf, dtype=float)))


def interval_score(y: np.ndarray, q_inf: np.ndarray, q_sup: np.ndarray, alpha: float) -> float:
    """
    Interval Score (Winkler) para um intervalo central de (1-alpha):

        IS = (u - l) + (2/alpha)(l - y)·1{y<l} + (2/alpha)(y - u)·1{y>u}

    E a regra propria que combina sharpness e calibracao num numero so:
    penaliza largura, mas penaliza MUITO mais deixar o realizado de fora.
    Menor e melhor.
    """
    y = np.asarray(y, dtype=float)
    l = np.asarray(q_inf, dtype=float)
    u = np.asarray(q_sup, dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha deve estar em (0, 1); recebido {alpha}")
    penal_inf = (2.0 / alpha) * np.clip(l - y, 0, None)
    penal_sup = (2.0 / alpha) * np.clip(y - u, 0, None)
    return float(np.mean((u - l) + penal_inf + penal_sup))


def crps_por_linha(y: np.ndarray, quantis_pred: np.ndarray, taus) -> np.ndarray:
    """
    CRPS de CADA observacao (vetor de tamanho n), pela mesma decomposicao
    quantilica de `crps_por_quantis`.

    Necessario para testar se uma previsao e melhor que outra: o teste de
    Diebold-Mariano opera sobre o DIFERENCIAL de perda ponto a ponto, nao
    sobre a media.
    """
    y = np.asarray(y, dtype=float)
    quantis_pred = np.asarray(quantis_pred, dtype=float)
    taus = list(taus)
    if quantis_pred.shape != (len(y), len(taus)):
        raise ValueError(
            f"quantis_pred deve ter forma ({len(y)}, {len(taus)}); recebido {quantis_pred.shape}"
        )
    perdas = np.column_stack([pinball_loss(y, quantis_pred[:, j], tau) for j, tau in enumerate(taus)])
    return 2.0 * perdas.mean(axis=1)
