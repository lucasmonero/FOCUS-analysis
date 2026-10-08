"""
Baselines probabilisticos da "escada" da Etapa 5.

Trabalha no espaco do ERRO (`erro = realizado - mediana_focus`): a
distribuicao preditiva do nivel e `mediana_focus + quantis(erro)`, e a
perda pinball e invariante a essa translacao, entao avaliar no espaco do
erro e equivalente a avaliar no nivel.

Degraus implementados aqui (os demais, B2/B3, vivem em `conformal.py` e no
notebook):

- **B0**  -- o que um usuario do boletim faria hoje: `Normal(0, sigma)` com o
  desvio-padrao publicado pelo Focus.
- **B0'** -- o MESMO sigma, reescalado por um fator `k` estimado no treino
  (por indicador x horizonte). E o concorrente honesto: a Etapa 3 mostrou
  que a dispersao do Focus ordena bem a incerteza (Spearman positivo em
  19/20 casos) mas erra a escala (k entre ~2,3 e ~4,7). Se esse remendo
  simples ja resolve, o resto da escada precisa provar que se paga.
- **B1**  -- quantis empiricos historicos do erro, sem olhar para o sigma:
  largura fixa, mas com a forma e o vies corretos.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from .scores import pinball_por_quantil

# Grade de quantis da escada: simetrica em torno de 0.5, de 5% a 95%.
# (Sem 2.5%/97.5%: com poucas linhas de calibracao por horizonte, o
# conformal nao consegue dar garantia finita nesses niveis extremos.)
TAUS_PADRAO = tuple(np.round(np.arange(0.05, 0.9501, 0.05), 2))


def quantis_normais(sigma: np.ndarray, taus, k: float = 1.0) -> np.ndarray:
    """
    Matriz (n, len(taus)) de quantis de `Normal(0, k * sigma)`.

    `sigma == 0` produz um intervalo de largura ZERO -- e exatamente o que um
    usuario ingenuo obteria de um consenso que convergiu no mesmo valor, e o
    que se quer medir. O chamador decide se descarta essas linhas.
    """
    sigma = np.asarray(sigma, dtype=float)
    z = stats.norm.ppf(np.asarray(list(taus), dtype=float))
    return np.outer(sigma, k * z)


def ajustar_fator_sigma(erro: np.ndarray, sigma: np.ndarray, taus=TAUS_PADRAO) -> float:
    """
    Fator `k` que minimiza a perda pinball media, sobre TODA a grade de
    quantis, de `Normal(0, k * sigma)` contra os erros observados.

    Usar a grade inteira (e nao so o nivel de 90%) e uma escolha deliberada:
    ajustar apenas um nivel poderia calibrar a cauda e descalibrar o centro
    (como ocorre na Selic de curto prazo, cuja distribuicao tem massa em
    zero). A funcao objetivo e convexa em `k` (soma de perdas pinball, cada
    uma convexa), entao a busca limitada e bem-comportada.

    Linhas com `sigma <= 0` sao ignoradas no ajuste.
    """
    erro = np.asarray(erro, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validas = np.isfinite(erro) & np.isfinite(sigma) & (sigma > 0)
    if validas.sum() < 5:
        raise ValueError(f"amostra pequena demais para ajustar k ({int(validas.sum())} linhas validas)")
    e, s = erro[validas], sigma[validas]
    taus = list(taus)

    def objetivo(k: float) -> float:
        perdas = pinball_por_quantil(e, quantis_normais(s, taus, k), taus)
        return float(np.mean(list(perdas.values())))

    return float(minimize_scalar(objetivo, bounds=(0.02, 100.0), method="bounded").x)


def quantis_historicos(erro_treino: np.ndarray, n_linhas: int, taus=TAUS_PADRAO) -> np.ndarray:
    """Matriz (n_linhas, len(taus)): os quantis empiricos do erro de treino, repetidos."""
    erro_treino = np.asarray(erro_treino, dtype=float)
    erro_treino = erro_treino[np.isfinite(erro_treino)]
    if len(erro_treino) == 0:
        raise ValueError("sem erros de treino")
    q = np.quantile(erro_treino, np.asarray(list(taus), dtype=float))
    return np.tile(q, (n_linhas, 1))


def escala_erros_resolvidos(
    datas_alvo,
    tempo_decisao,
    erro: np.ndarray,
    n_ultimos: int = 12,
    defasagem_dias: int = 60,
    minimo: int = 6,
) -> np.ndarray:
    """
    Para cada previsao, a media de |erro| das `n_ultimos` previsoes mais
    recentes cujo alvo JA ESTAVA RESOLVIDO no instante da decisao.

    Uma previsao com alvo em `T` so tem o erro conhecido depois que o
    realizado e publicado: consideramos `T + defasagem_dias` (conservador --
    cobre a defasagem mais longa do projeto, a da PNADC). Usar o erro de uma
    previsao ainda nao resolvida vazaria o futuro para dentro da escala.

    Deve ser chamado sobre UM horizonte por vez (a escala do erro difere
    muito entre horizontes). Devolve NaN onde ha menos de `minimo` erros
    resolvidos disponiveis. O resultado esta alinhado a ordem de entrada.
    """
    datas_alvo = np.asarray(datas_alvo, dtype="datetime64[ns]")
    t_dec = np.asarray(tempo_decisao, dtype="datetime64[ns]")
    erro = np.abs(np.asarray(erro, dtype=float))
    n = len(erro)

    ordem = np.argsort(datas_alvo, kind="stable")
    resolvido_em = datas_alvo[ordem] + np.timedelta64(int(defasagem_dias), "D")
    erro_ord = erro[ordem]

    saida = np.full(n, np.nan)
    for i in range(n):
        k = int(np.searchsorted(resolvido_em, t_dec[i], side="right"))  # quantos ja resolvidos
        if k >= minimo:
            saida[i] = erro_ord[max(0, k - n_ultimos):k].mean()
    return saida
