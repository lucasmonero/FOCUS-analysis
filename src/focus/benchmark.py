"""
Benchmark ingenuo ("nao prever mudanca") para avaliar previsoes do Focus.

A regua minima de qualquer projecao e bater "o proximo valor sera igual ao
ultimo que conheco". Mas o benchmark so e justo se respeitar o MESMO
conjunto de informacao do analista: o ultimo valor **ja publicado** na
`data_travamento`, e nao `realizado(P-h)`.

Isso ja produziu um resultado espurio neste projeto. A PNADC (Taxa de
desocupacao) e divulgada com ~60 dias de defasagem; um benchmark que usava
`realizado(P-1)` dava a impressao de que o Focus errava 2,2x mais que nao
prever mudanca -- quando, respeitando a defasagem, ele erra 0,79x. O
analista simplesmente nao tinha o dado do mes anterior.

Por isso a logica mora aqui, com teste, e nao inline nos notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pontual import calcular_data_disponibilidade, juntar_as_of


def previsao_ingenua_mensal(
    painel: pd.DataFrame,
    realizado_df: pd.DataFrame,
    nome_regra: str,
    coluna_ancora: str = "data_travamento",
) -> pd.Series:
    """
    Para cada linha do painel, o ultimo valor realizado JA PUBLICADO em
    `coluna_ancora`, segundo a regra de defasagem `nome_regra` (ver
    `configuracao.DEFASAGENS_PUBLICACAO`).

    `realizado_df` tem colunas `data` (primeiro dia do mes de referencia,
    convencao do SGS) e `valor`. O resultado esta alinhado ao indice de
    `painel` (a ordem original e preservada), com NaN onde nada havia sido
    publicado ainda.
    """
    real = realizado_df.copy()
    real["fim_periodo"] = real["data"] + pd.offsets.MonthEnd(0)
    real = calcular_data_disponibilidade(real, "fim_periodo", nome_regra).sort_values("data")

    ancoras = painel[[coluna_ancora]].copy()
    ancoras["_pos"] = np.arange(len(ancoras))
    unido = juntar_as_of(ancoras, coluna_ancora, real, ["valor"], prefixo="ingenuo")
    unido = unido.sort_values("_pos")

    return pd.Series(unido["ingenuo_valor"].to_numpy(), index=painel.index, name="previsao_ingenua")


def previsao_ingenua_selic(painel: pd.DataFrame) -> pd.Series:
    """
    Para a Selic o "ultimo valor publicado" e a meta vigente na
    `data_travamento` -- a decisao do Copom e publica no proprio dia, sem
    defasagem, e as flags de regime ja carregam a ultima meta conhecida.
    """
    return painel["regime_ultima_meta_selic_conhecida"].rename("previsao_ingenua")


def razao_mae(erro_focus: pd.Series, realizado: pd.Series, previsao_ingenua: pd.Series) -> float:
    """MAE do Focus dividido pelo MAE do benchmark ingenuo. Abaixo de 1, o Focus agrega valor."""
    validos = previsao_ingenua.notna() & erro_focus.notna()
    mae_focus = erro_focus[validos].abs().mean()
    mae_ingenuo = (realizado[validos] - previsao_ingenua[validos]).abs().mean()
    return float(mae_focus / mae_ingenuo)
