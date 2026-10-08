"""
Testes de calibracao de intervalos e densidades preditivas.

Usado na Etapa 3 para responder a pergunta central da avaliacao de
qualidade: **a dispersao que o Focus publica e uma medida honesta da
incerteza da projecao?** Discordancia entre analistas e incerteza da
previsao sao conceitos diferentes, e a primeira tende a subestimar a
segunda -- mas isso precisa ser medido, nao assumido.

Tres instrumentos, em ordem de exigencia:

1. **Kupiec** (cobertura incondicional): a frequencia de violacoes bate com
   a nominal? Um intervalo de 90% deveria ser furado ~10% das vezes.
2. **Christoffersen** (independencia + cobertura condicional): as violacoes
   sao independentes, ou se agrupam? Furar 10% das vezes mas com todas as
   falhas concentradas num pivot e muito pior do que 10% espalhado -- e e
   exatamente o padrao que se espera de um intervalo que ignora regime.
3. **PIT**: a densidade INTEIRA esta calibrada, nao so um intervalo.

As formulas usam `scipy.special.xlogy`, que trata 0*log(0)=0 -- necessario
porque contagens de transicao zeradas sao comuns em amostras curtas e
produziriam NaN com um `log` ingenuo.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlogy


def pit_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Transformada integral de probabilidade (PIT) supondo densidade normal:
    `u_t = Phi((y_t - mu_t) / sigma_t)`.

    Se a densidade preditiva estiver calibrada, `u` e Uniforme(0,1). Desvios
    tem leitura direta: U em forma de banana (acumulo nas pontas) = densidade
    estreita demais; acumulo no centro = larga demais; inclinada = viesada.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma deve ser estritamente positivo em todas as observacoes")
    return stats.norm.cdf((y - mu) / sigma)


def teste_uniformidade_pit(pit_valores: np.ndarray) -> dict:
    """Kolmogorov-Smirnov do PIT contra a Uniforme(0,1). p < 0.05 => densidade descalibrada."""
    pit_valores = np.asarray(pit_valores, dtype=float)
    pit_valores = pit_valores[np.isfinite(pit_valores)]
    estatistica, p_valor = stats.kstest(pit_valores, "uniform")
    return {
        "n": int(len(pit_valores)),
        "ks_estatistica": float(estatistica),
        "p_valor": float(p_valor),
        "calibrada_5pct": bool(p_valor >= 0.05),
    }


def teste_kupiec(violacoes: np.ndarray, p_nominal: float) -> dict:
    """
    Teste de razao de verossimilhanca de Kupiec (1995) para cobertura
    INCONDICIONAL.

    `violacoes`: vetor 0/1, 1 = o realizado caiu FORA do intervalo.
    `p_nominal`: taxa de violacao esperada (ex.: 0.10 para um intervalo de 90%).

    H0: a taxa de violacao observada e igual a nominal. `LR_uc ~ qui2(1)`.
    """
    v = np.asarray(violacoes, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        raise ValueError("vetor de violacoes vazio")
    if not 0.0 < p_nominal < 1.0:
        raise ValueError(f"p_nominal deve estar em (0, 1); recebido {p_nominal}")

    x = float(v.sum())
    pi_obs = x / n

    log_l0 = xlogy(x, p_nominal) + xlogy(n - x, 1.0 - p_nominal)
    log_l1 = xlogy(x, pi_obs) + xlogy(n - x, 1.0 - pi_obs)
    lr_uc = float(-2.0 * (log_l0 - log_l1))
    p_valor = float(stats.chi2.sf(lr_uc, df=1))

    return {
        "n": n,
        "violacoes": int(x),
        "taxa_violacao_observada": float(pi_obs),
        "taxa_violacao_nominal": float(p_nominal),
        "lr_uc": lr_uc,
        "p_valor": p_valor,
        "rejeita_5pct": bool(p_valor < 0.05),
    }


def teste_christoffersen(violacoes: np.ndarray, p_nominal: float) -> dict:
    """
    Teste de Christoffersen (1998): independencia das violacoes e cobertura
    CONDICIONAL.

    `LR_ind ~ qui2(1)` testa se a probabilidade de violar depende de ter
    violado no periodo anterior (agrupamento). `LR_cc = LR_uc + LR_ind ~
    qui2(2)` testa as duas coisas juntas.

    Agrupamento de violacoes e o sintoma classico de um intervalo de largura
    fixa aplicado a um mundo com regimes: ele acerta a media de cobertura ao
    longo do tempo, mas falha sistematicamente junto, bem quando importa.
    """
    v = np.asarray(violacoes, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        raise ValueError("sao necessarias ao menos 2 observacoes para testar independencia")

    anterior, atual = v[:-1], v[1:]
    n00 = float(np.sum((anterior == 0) & (atual == 0)))
    n01 = float(np.sum((anterior == 0) & (atual == 1)))
    n10 = float(np.sum((anterior == 1) & (atual == 0)))
    n11 = float(np.sum((anterior == 1) & (atual == 1)))

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l_restrito = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    log_l_livre = (
        xlogy(n00, 1.0 - pi_01) + xlogy(n01, pi_01)
        + xlogy(n10, 1.0 - pi_11) + xlogy(n11, pi_11)
    )
    lr_ind = float(-2.0 * (log_l_restrito - log_l_livre))

    resultado_uc = teste_kupiec(v, p_nominal)
    lr_cc = float(resultado_uc["lr_uc"] + lr_ind)

    return {
        "n": resultado_uc["n"],
        "lr_ind": lr_ind,
        "p_valor_ind": float(stats.chi2.sf(lr_ind, df=1)),
        "lr_cc": lr_cc,
        "p_valor_cc": float(stats.chi2.sf(lr_cc, df=2)),
        "rejeita_independencia_5pct": bool(stats.chi2.sf(lr_ind, df=1) < 0.05),
        "rejeita_cobertura_condicional_5pct": bool(stats.chi2.sf(lr_cc, df=2) < 0.05),
    }


def avaliar_intervalo(
    y: np.ndarray, q_inf: np.ndarray, q_sup: np.ndarray, nivel_nominal: float
) -> dict:
    """
    Conveniencia: junta cobertura observada, Kupiec e Christoffersen para um
    intervalo de cobertura nominal `nivel_nominal` (ex.: 0.90).

    ATENCAO: `y`, `q_inf` e `q_sup` devem estar em ordem CRONOLOGICA -- o
    teste de independencia de Christoffersen le a sequencia de violacoes como
    serie temporal; embaralhar destruiria justamente o agrupamento que ele
    deveria detectar.
    """
    y = np.asarray(y, dtype=float)
    q_inf = np.asarray(q_inf, dtype=float)
    q_sup = np.asarray(q_sup, dtype=float)
    if not 0.0 < nivel_nominal < 1.0:
        raise ValueError(f"nivel_nominal deve estar em (0, 1); recebido {nivel_nominal}")

    valido = np.isfinite(y) & np.isfinite(q_inf) & np.isfinite(q_sup)
    y, q_inf, q_sup = y[valido], q_inf[valido], q_sup[valido]

    violacoes = ((y < q_inf) | (y > q_sup)).astype(float)
    p_nominal = 1.0 - nivel_nominal

    kupiec = teste_kupiec(violacoes, p_nominal)
    christ = teste_christoffersen(violacoes, p_nominal)

    return {
        "nivel_nominal": float(nivel_nominal),
        "cobertura_observada": float(1.0 - kupiec["taxa_violacao_observada"]),
        "n": kupiec["n"],
        "violacoes": kupiec["violacoes"],
        "largura_media": float(np.mean(q_sup - q_inf)),
        "kupiec_lr": kupiec["lr_uc"],
        "kupiec_p": kupiec["p_valor"],
        "christoffersen_ind_p": christ["p_valor_ind"],
        "christoffersen_cc_p": christ["p_valor_cc"],
        "rejeita_calibracao_5pct": bool(kupiec["rejeita_5pct"]),
    }
