"""
Predicao conformal para intervalos com garantia de cobertura em amostra finita.

Motivacao. Um intervalo construido com a dispersao publicada do Focus cobre
na mediana apenas ~40% das vezes quando declara 90% (Etapa 3). Reescalar o
sigma ou estimar quantis resolve a ESCALA, mas nao entrega garantia
nenhuma: se a forma ou o regime mudarem, a cobertura se perde sem aviso. A
predicao conformal entrega uma garantia que nao depende de a distribuicao
estar bem especificada:

    P( y in [lo, hi] ) >= 1 - alpha        (em amostra FINITA)

sob a unica hipotese de **permutabilidade** entre os erros de calibracao e
os de teste.

**Ressalva central -- permutabilidade.** Series macro com mudanca de regime
violam a permutabilidade: o erro de 2020 nao e "permutavel" com o de 2010.
Por isso este modulo NAO promete a garantia como teorema aplicavel a
serie: ela vale marginalmente sob permutabilidade, e o que se mede na
pratica e a cobertura EMPIRICA fora da amostra, inclusive condicional a
regime. A variante adaptativa (ACI, Gibbs & Candes 2021) existe justamente
para tratar deriva, e tambem e avaliada empiricamente.

Referencias: Romano, Patterson & Candes (2019), "Conformalized Quantile
Regression"; Gibbs & Candes (2021), "Adaptive Conformal Inference Under
Distribution Shift".
"""
from __future__ import annotations

import math

import numpy as np


def escores_cqr(q_inf: np.ndarray, q_sup: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Escore de conformidade do CQR: quanto o realizado excede o intervalo
    [q_inf, q_sup] -- negativo quando cai DENTRO (e o quao dentro).

        E_i = max(q_inf_i - y_i,  y_i - q_sup_i)
    """
    q_inf = np.asarray(q_inf, dtype=float)
    q_sup = np.asarray(q_sup, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.maximum(q_inf - y, y - q_sup)


def quantil_conformal(escores: np.ndarray, alpha: float) -> float:
    """
    Quantil de nivel ceil((1-alpha)(n+1))/n dos escores -- a correcao de
    amostra finita que da a garantia (o "+1" e o que diferencia isso de um
    quantil empirico comum, que teria cobertura ligeiramente abaixo).

    Se `ceil((1-alpha)(n+1)) > n` (amostra pequena demais para esse nivel),
    a unica escolha valida e um intervalo INFINITO: devolve `+inf`. Isso e
    honesto -- com poucos dados de calibracao, nao ha garantia finita a dar.
    """
    escores = np.asarray(escores, dtype=float)
    escores = escores[np.isfinite(escores)]
    n = len(escores)
    if n == 0:
        raise ValueError("sem escores de calibracao")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha deve estar em (0, 1); recebido {alpha}")

    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        return float("inf")
    return float(np.sort(escores)[k - 1])


def cqr_ajuste(q_inf_cal, q_sup_cal, y_cal, alpha: float) -> float:
    """Correcao Q do CQR: o intervalo final e [q_inf - Q, q_sup + Q]. Q pode ser
    NEGATIVO (encolhe o intervalo) quando o modelo base sobre-cobre."""
    return quantil_conformal(escores_cqr(q_inf_cal, q_sup_cal, y_cal), alpha)


def ordenar_quantis(quantis: np.ndarray) -> np.ndarray:
    """
    Rearranjo monotono (Chernozhukov et al. 2010): ordena os quantis de cada
    linha, eliminando cruzamentos. Nunca piora a perda pinball e e o remedio
    padrao para regressao quantilica com uma arvore por nivel.
    """
    return np.sort(np.asarray(quantis, dtype=float), axis=1)


def cqr_multiplos_niveis(
    quantis_cal: np.ndarray,
    y_cal: np.ndarray,
    quantis_teste: np.ndarray,
    taus,
) -> np.ndarray:
    """
    Aplica o CQR a CADA par simetrico de quantis (tau, 1-tau) de uma grade,
    devolvendo a matriz de quantis de teste ja conformalizados.

    O par (tau, 1-tau) define um intervalo central de nivel 1 - 2*tau, ou
    seja `alpha = 2*tau`. O quantil central (tau = 0.5), se presente, fica
    intacto. O resultado e reordenado para garantir monotonicidade.

    `taus` deve ser simetrico em torno de 0.5 (ex.: 0.05, 0.10, ..., 0.95).
    """
    taus = np.asarray(list(taus), dtype=float)
    quantis_cal = np.asarray(quantis_cal, dtype=float)
    quantis_teste = np.asarray(quantis_teste, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    if quantis_cal.shape[1] != len(taus) or quantis_teste.shape[1] != len(taus):
        raise ValueError("numero de colunas de quantis difere de len(taus)")

    ajustados = quantis_teste.copy()
    for j, tau in enumerate(taus):
        if tau >= 0.5:
            continue
        espelho = np.flatnonzero(np.isclose(taus, 1.0 - tau))
        if len(espelho) != 1:
            raise ValueError(f"taus nao e simetrico: falta o par de {tau}")
        j_sup = int(espelho[0])

        Q = cqr_ajuste(quantis_cal[:, j], quantis_cal[:, j_sup], y_cal, alpha=2.0 * tau)
        ajustados[:, j] = quantis_teste[:, j] - Q
        ajustados[:, j_sup] = quantis_teste[:, j_sup] + Q

    return ordenar_quantis(ajustados)


def aci_intervalos(
    q_inf: np.ndarray,
    q_sup: np.ndarray,
    y: np.ndarray,
    escores_cal: np.ndarray,
    tempo_decisao: np.ndarray,
    tempo_desfecho: np.ndarray,
    alpha: float,
    gamma: float = 0.02,
    grupo: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive Conformal Inference (Gibbs & Candes 2021) com **atraso de
    feedback**.

    Em vez de um `alpha` fixo, mantem um `alpha_t` que sobe quando o
    intervalo tem coberto DEMAIS e desce quando cobre de MENOS:

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),    err_t = 1{y_t fora}

    Ha uma sutileza essencial em previsao: a previsao feita em `t` para um
    alvo `h` meses a frente so tem o DESFECHO conhecido mais tarde. Logo o
    `alpha_t` usado numa previsao so pode incorporar erros de previsoes cujo
    alvo ja se realizou naquele instante -- por isso `tempo_desfecho` (data
    do alvo) e `tempo_decisao` (data_travamento) entram explicitamente.
    Ignorar isso vazaria informacao do futuro para dentro do ajuste.

    Parametros
    ----------
    q_inf, q_sup: quantis base (nivel alpha/2 e 1-alpha/2) de cada linha de teste.
    y: realizado de cada linha de teste (usado SO apos `tempo_desfecho`).
    escores_cal: escores de calibracao (`escores_cqr`) do modelo base. Um array
        1-D, ou -- se `grupo` for dado -- um dict `grupo -> array`.
    grupo: opcional, rotulo por linha selecionando QUAL conjunto de escores de
        calibracao vale para aquela linha. Necessario quando o modelo base e
        refeito por fold (cada fold tem os seus escores), mantendo o estado
        `alpha_t` unico ao longo de todo o fluxo temporal.
    tempo_decisao / tempo_desfecho: arrays de datas (mesma ordem de `y`).

    Devolve `(lo, hi, alpha_usado)`, alinhados a ordem ORIGINAL de entrada.
    """
    q_inf = np.asarray(q_inf, dtype=float)
    q_sup = np.asarray(q_sup, dtype=float)
    y = np.asarray(y, dtype=float)
    t_dec = np.asarray(tempo_decisao, dtype="datetime64[ns]")
    t_des = np.asarray(tempo_desfecho, dtype="datetime64[ns]")
    n = len(y)
    if grupo is None:
        conjuntos = {0: np.sort(np.asarray(escores_cal, dtype=float))}
        grupo = np.zeros(n, dtype=int)
    else:
        grupo = np.asarray(grupo)
        conjuntos = {g: np.sort(np.asarray(v, dtype=float)) for g, v in escores_cal.items()}

    def corte_para(alpha_t: float, g) -> float:
        escores = conjuntos[g]
        n_cal = len(escores)
        if alpha_t <= 0.0:
            return float("inf")            # nunca errar: intervalo infinito
        if alpha_t >= 1.0:
            return float("-inf")           # intervalo vazio/colapsado
        k = math.ceil((1.0 - alpha_t) * (n_cal + 1))
        return float("inf") if k > n_cal else float(escores[k - 1])

    ordem_decisao = np.argsort(t_dec, kind="stable")
    ordem_desfecho = np.argsort(t_des, kind="stable")

    # NaN (e nao np.empty): `lo[j]` e testado abaixo para saber se a previsao
    # j ja foi emitida -- memoria nao inicializada tornaria isso indefinido.
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    alpha_usado = np.full(n, np.nan)
    coberto = np.zeros(n, dtype=bool)

    alpha_t = alpha
    ponteiro = 0  # proximo desfecho (em ordem de data do alvo) ainda nao incorporado

    for i in ordem_decisao:
        # incorpora todo desfecho ja conhecido no instante desta decisao,
        # SE a previsao correspondente ja tiver sido emitida
        while ponteiro < n:
            j = ordem_desfecho[ponteiro]
            if t_des[j] > t_dec[i]:
                break
            if not np.isnan(lo[j]):                       # ja emitida
                alpha_t = alpha_t + gamma * (alpha - (0.0 if coberto[j] else 1.0))
                ponteiro += 1
            else:
                break  # desfecho de previsao ainda nao emitida: espera

        Q = corte_para(alpha_t, grupo[i])
        lo[i], hi[i] = q_inf[i] - Q, q_sup[i] + Q
        alpha_usado[i] = alpha_t
        coberto[i] = (y[i] >= lo[i]) and (y[i] <= hi[i])

    return lo, hi, alpha_usado
