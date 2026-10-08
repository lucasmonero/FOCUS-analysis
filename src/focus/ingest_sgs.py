"""
Extrator das series do SGS (Sistema Gerenciador de Series Temporais) usadas
como "realizado" e como features macro: IPCA, Meta Selic, IBC-Br, IC-Br e
meta de inflacao.

O endpoint publico `api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados`
rejeita janelas maiores que ~10 anos para series diarias -- fatiamos por
janela de `SGS_JANELA_MAX_ANOS_SERIE_DIARIA` anos e concatenamos. Series
mensais nao tem esse limite na pratica, mas fatiamos do mesmo jeito por
uniformidade e para manter cada requisicao pequena/idempotente.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .configuracao import (
    AMOSTRA_INICIO,
    SERIES_SGS,
    SGS_JANELA_MAX_ANOS_SERIE_DIARIA,
)
from .rede import buscar_json, pausar_entre_requisicoes

URL_SGS_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"


def _janelas_de_datas(inicio: dt.date, fim: dt.date, max_anos: int) -> list[tuple[dt.date, dt.date]]:
    """Quebra [inicio, fim] em janelas de no maximo `max_anos` anos."""
    janelas = []
    cursor = inicio
    while cursor < fim:
        # a API rejeita janelas de EXATAMENTE `max_anos` anos: termina um dia antes
        proximo = min(dt.date(cursor.year + max_anos, cursor.month, cursor.day) - dt.timedelta(days=1), fim)
        janelas.append((cursor, proximo))
        cursor = proximo + dt.timedelta(days=1)
    return janelas


def buscar_serie_sgs(codigo: int, data_inicio: str, data_fim: str | None = None) -> pd.DataFrame:
    """
    Busca uma serie SGS completa, fatiada em janelas seguras, e retorna um
    DataFrame com colunas `data` (datetime) e `valor` (float), ordenado.
    """
    inicio = pd.Timestamp(data_inicio).date()
    fim = pd.Timestamp(data_fim).date() if data_fim else dt.date.today()

    partes: list[pd.DataFrame] = []
    for jan_inicio, jan_fim in _janelas_de_datas(inicio, fim, SGS_JANELA_MAX_ANOS_SERIE_DIARIA):
        url = URL_SGS_BASE.format(codigo=codigo)
        params = {
            "formato": "json",
            "dataInicial": jan_inicio.strftime("%d/%m/%Y"),
            "dataFinal": jan_fim.strftime("%d/%m/%Y"),
        }
        payload = buscar_json(url, params=params)
        if payload:
            partes.append(pd.DataFrame.from_records(payload))
        pausar_entre_requisicoes()

    if not partes:
        return pd.DataFrame(columns=["data", "valor"])

    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.drop_duplicates(subset="data").sort_values("data").reset_index(drop=True)
    return df


def buscar_todas_series_sgs(amostra_inicio: str = AMOSTRA_INICIO) -> dict[str, pd.DataFrame]:
    """Busca todas as series configuradas em `SERIES_SGS` e retorna nome -> DataFrame."""
    # Margem de 2 anos antes do inicio da amostra: momentos (3m/6m) e
    # revisoes precisam de historico anterior a AMOSTRA_INICIO para nao
    # nascer com NaN nos primeiros meses.
    inicio_com_margem = (pd.Timestamp(amostra_inicio) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    resultado = {}
    for codigo, (nome, _e_diaria) in SERIES_SGS.items():
        resultado[nome] = buscar_serie_sgs(codigo, inicio_com_margem)
    return resultado
