"""
Teste de Diebold-Mariano (1995) com a correcao de amostra pequena de
Harvey-Leybourne-Newbold (1997), para comparar duas series de perda de
previsao em horizonte `h` passos a frente.

Formula isolada aqui (fora do notebook) porque e facil de errar
sutilmente -- sinal do diferencial, defasagem do HAC, o fator de correcao
HLN, os graus de liberdade da t -- e um erro de sinal na formula inverteria
silenciosamente a conclusao de "o modelo bate o Focus" para o oposto.
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats


def diebold_mariano(perda_a: np.ndarray, perda_b: np.ndarray, h: int) -> dict:
    """
    H0: as duas previsoes tem acuracia igual, E[d_t] = 0, onde
    `d_t = perda_a_t - perda_b_t`.

    `d_medio > 0` => perda_a maior que perda_b, ou seja, **b e a previsao
    melhor** (menor perda). `perda_a`/`perda_b` devem ser series de perda ja
    calculadas (ex.: erro quadratico ou absoluto), alinhadas ponto a ponto e
    ordenadas cronologicamente.

    O erro-padrao de `d_medio` usa covariancia HAC/Newey-West com
    `maxlags = h-1` -- o limite teorico de autocorrelacao mecanica de erros
    de projecao em h passos a frente sob eficiencia (ver `03_eda.ipynb`).
    A estatistica DM resultante e ajustada pela correcao de amostra pequena
    de Harvey-Leybourne-Newbold (1997) e comparada a uma t de Student com
    `n-1` graus de liberdade, em vez da normal padrao assintotica --
    necessario porque nossas amostras (n~150-250) sao pequenas para a
    aproximacao assintotica do DM original.

    Retorna um dict com `n`, `d_medio`, `dm_stat` (antes da correcao),
    `dm_ajustado_hln`, `p_valor` e `graus_liberdade`.
    """
    d = np.asarray(perda_a, dtype=float) - np.asarray(perda_b, dtype=float)
    n = len(d)
    if n < 3:
        raise ValueError(f"Amostra pequena demais para o teste DM (n={n})")
    if h < 1:
        raise ValueError(f"h deve ser >= 1 (recebido {h})")

    maxlags = max(h - 1, 0)
    X = np.ones(n)
    modelo = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    dm_stat = float(modelo.tvalues[0])
    d_medio = float(d.mean())

    fator_hln = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm_ajustado = dm_stat * fator_hln

    graus_liberdade = n - 1
    p_valor = float(2 * (1 - stats.t.cdf(abs(dm_ajustado), df=graus_liberdade)))

    return {
        "n": n,
        "d_medio": d_medio,
        "dm_stat": dm_stat,
        "dm_ajustado_hln": dm_ajustado,
        "p_valor": p_valor,
        "graus_liberdade": graus_liberdade,
    }
