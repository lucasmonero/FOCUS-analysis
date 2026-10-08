"""Ingestao completa das fontes publicas (Focus/OData, SGS, Copom, yfinance) para Parquet."""
from __future__ import annotations

import pandas as pd

from .armazenamento import obter_ou_buscar
from .configuracao import (
    AMOSTRA_INICIO,
    DIR_BRUTO,
    DIR_REFERENCIA,
    INDICADORES_MENSAIS,
    SERIES_SGS,
    TICKERS_MERCADO,
    URL_HISTORICO_TAXAS_JUROS,
)
from .copom import construir_calendario_copom, validar_calendario_copom
from .ingest_focus import (
    buscar_focus_mensal,
    buscar_focus_selic,
    buscar_focus_top5_mensal,
    buscar_focus_top5_selic,
)
from .ingest_market import buscar_ticker_yfinance
from .ingest_sgs import buscar_serie_sgs


def _fechamento(ticker: str) -> pd.DataFrame:
    return buscar_ticker_yfinance(ticker)[["data", "Close"]].rename(columns={"Close": "fechamento"})


def ingerir() -> None:
    """Rebaixa todas as fontes (sem cache) e regrava `dados/bruto` e `dados/referencia`."""

    def salvar(pasta, nome, fonte, busca, coluna_data):
        obter_ou_buscar(pasta, nome, fonte, busca, coluna_data=coluna_data, forcar=True)

    calendario = obter_ou_buscar(
        DIR_REFERENCIA, "calendario_copom", URL_HISTORICO_TAXAS_JUROS,
        lambda: construir_calendario_copom(AMOSTRA_INICIO), coluna_data="data_reuniao", forcar=True,
    )
    erros = validar_calendario_copom(calendario)["anos_com_contagem_errada"]
    if erros:
        raise ValueError(f"Calendario do Copom com contagem de reunioes inesperada: {erros}")

    for indicador, cfg in INDICADORES_MENSAIS.items():
        nome = cfg["nome"]
        salvar(DIR_BRUTO, f"focus_{nome}_mensal", f"Olinda OData (mensal / {indicador})",
               lambda i=indicador: buscar_focus_mensal(i, AMOSTRA_INICIO), "data_pesquisa")
        salvar(DIR_BRUTO, f"focus_{nome}_top5_mensal", f"Olinda OData (top5 mensal / {indicador})",
               lambda i=indicador: buscar_focus_top5_mensal(i, AMOSTRA_INICIO), "data_pesquisa")
    salvar(DIR_BRUTO, "focus_selic", "Olinda OData (Selic)",
           lambda: buscar_focus_selic(AMOSTRA_INICIO), "data_pesquisa")
    salvar(DIR_BRUTO, "focus_top5_selic", "Olinda OData (Selic Top 5)",
           lambda: buscar_focus_top5_selic(AMOSTRA_INICIO), "data_pesquisa")

    # 2 anos de margem: momentos (3m/6m) e revisoes precisam de historico antes da amostra
    inicio = (pd.Timestamp(AMOSTRA_INICIO) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    for codigo, (nome, _) in SERIES_SGS.items():
        salvar(DIR_BRUTO, f"sgs_{nome}", f"SGS/BCB (codigo {codigo})",
               lambda c=codigo: buscar_serie_sgs(c, inicio), "data")

    for ticker, nome in TICKERS_MERCADO.items():
        salvar(DIR_BRUTO, f"mercado_{nome}", f"yfinance ({ticker})",
               lambda t=ticker: _fechamento(t), "data")
