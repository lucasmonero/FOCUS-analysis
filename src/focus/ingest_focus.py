"""
Extratores das entidades Focus (Expectativas de Mercado).

Os indicadores MENSAIS (IPCA, Cambio, IGP-M, Taxa de desocupacao, ...)
compartilham exatamente a mesma forma no OData -- mensais, chaveados por
`DataReferencia` -- entao um unico par de funcoes genericas
(`buscar_focus_mensal` / `buscar_focus_top5_mensal`) cobre todos eles.

A Selic e o caso a parte: chaveada por REUNIAO do Copom (`Reuniao`), nao por
mes, e por isso tem funcoes proprias.

Todos retornam DataFrames ja normalizados (colunas em PT) e filtrados por
`base_calculo`/`tipo_calculo` conforme as regras de `configuracao.py`.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .configuracao import (
    AMOSTRA_INICIO,
    BASE_CALCULO_FOCUS,
    ENDPOINT_FOCUS_MENSAL,
    ENDPOINT_FOCUS_SELIC,
    ENDPOINT_FOCUS_TOP5_MENSAL,
    ENDPOINT_FOCUS_TOP5_SELIC,
    TIPO_CALCULO_TOP5,
)
from .odata import buscar_odata_por_ano


def _ano_inicio(amostra_inicio: str) -> int:
    return pd.Timestamp(amostra_inicio).year


def _ano_fim_atual() -> int:
    return dt.date.today().year


def _asserir_sem_duplicatas(df: pd.DataFrame, chaves: list[str], nome: str) -> None:
    duplicadas = df.duplicated(subset=chaves, keep=False)
    if duplicadas.any():
        n_dup = int(duplicadas.sum())
        raise ValueError(
            f"{nome}: {n_dup} linhas duplicadas em {chaves}. "
            "Verifique se o filtro de base_calculo/tipo_calculo foi aplicado."
        )


# ---------------------------------------------------------------------------
# Indicadores mensais (genericos)
# ---------------------------------------------------------------------------
def buscar_focus_mensal(indicador: str, amostra_inicio: str = AMOSTRA_INICIO) -> pd.DataFrame:
    """
    Mediana/media mensal de um indicador qualquer do Focus (so baseCalculo=0).

    `indicador` deve ser o rotulo exato usado pelo BCB -- ex.: 'IPCA',
    'Câmbio', 'IGP-M', 'Taxa de desocupação' (ver
    `configuracao.INDICADORES_MENSAIS`).
    """
    df = buscar_odata_por_ano(
        ENDPOINT_FOCUS_MENSAL,
        ano_inicio=_ano_inicio(amostra_inicio),
        ano_fim=_ano_fim_atual(),
        filtro_extra=f"Indicador eq '{indicador}'",
    )
    if df.empty:
        return df

    df["base_calculo"] = pd.to_numeric(df["base_calculo"])
    df = df.loc[df["base_calculo"] == BASE_CALCULO_FOCUS].copy()
    df["data_pesquisa"] = pd.to_datetime(df["data_pesquisa"])

    _asserir_sem_duplicatas(
        df, ["indicador", "data_pesquisa", "referencia"], f"focus_mensal[{indicador}]"
    )
    return df.sort_values(["referencia", "data_pesquisa"]).reset_index(drop=True)


def buscar_focus_top5_mensal(indicador: str, amostra_inicio: str = AMOSTRA_INICIO) -> pd.DataFrame:
    """Top 5 (curto prazo) mensal de um indicador qualquer, so tipoCalculo='C'."""
    df = buscar_odata_por_ano(
        ENDPOINT_FOCUS_TOP5_MENSAL,
        ano_inicio=_ano_inicio(amostra_inicio),
        ano_fim=_ano_fim_atual(),
        filtro_extra=f"Indicador eq '{indicador}'",
    )
    if df.empty:
        return df

    df = df.loc[df["tipo_calculo"] == TIPO_CALCULO_TOP5].copy()
    df["data_pesquisa"] = pd.to_datetime(df["data_pesquisa"])

    _asserir_sem_duplicatas(
        df, ["indicador", "data_pesquisa", "referencia"], f"focus_top5_mensal[{indicador}]"
    )
    return df.sort_values(["referencia", "data_pesquisa"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Selic (chaveada por reuniao do Copom)
# ---------------------------------------------------------------------------
def buscar_focus_selic(amostra_inicio: str = AMOSTRA_INICIO) -> pd.DataFrame:
    """Mediana/media da Selic por reuniao do Copom, so baseCalculo=0."""
    df = buscar_odata_por_ano(
        ENDPOINT_FOCUS_SELIC,
        ano_inicio=_ano_inicio(amostra_inicio),
        ano_fim=_ano_fim_atual(),
        filtro_extra="Indicador eq 'Selic'",
    )
    if df.empty:
        return df

    df["base_calculo"] = pd.to_numeric(df["base_calculo"])
    df = df.loc[df["base_calculo"] == BASE_CALCULO_FOCUS].copy()
    df["data_pesquisa"] = pd.to_datetime(df["data_pesquisa"])

    _asserir_sem_duplicatas(df, ["indicador", "data_pesquisa", "reuniao"], "focus_selic")
    return df.sort_values(["reuniao", "data_pesquisa"]).reset_index(drop=True)


def buscar_focus_top5_selic(amostra_inicio: str = AMOSTRA_INICIO) -> pd.DataFrame:
    """Top 5 (curto prazo) da Selic por reuniao, so tipoCalculo='C'.

    Atencao: esta entidade usa nomes de campo em MINUSCULAS no OData
    (`indicador`, `reuniao`, ...), ao contrario das demais -- por isso o
    filtro abaixo usa 'indicador' e nao 'Indicador'.
    """
    df = buscar_odata_por_ano(
        ENDPOINT_FOCUS_TOP5_SELIC,
        ano_inicio=_ano_inicio(amostra_inicio),
        ano_fim=_ano_fim_atual(),
        filtro_extra="indicador eq 'Selic'",
    )
    if df.empty:
        return df

    df = df.loc[df["tipo_calculo"] == TIPO_CALCULO_TOP5].copy()
    df["data_pesquisa"] = pd.to_datetime(df["data_pesquisa"])

    _asserir_sem_duplicatas(df, ["indicador", "data_pesquisa", "reuniao"], "focus_top5_selic")
    return df.sort_values(["reuniao", "data_pesquisa"]).reset_index(drop=True)
