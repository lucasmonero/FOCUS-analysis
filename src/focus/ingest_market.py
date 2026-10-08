"""
Extrator de series de mercado externo via yfinance: USD/BRL e o rendimento
do Treasury de 10 anos dos EUA (^TNX).

O yfinance falha SILENCIOSAMENTE devolvendo um DataFrame vazio quando a
requisicao tem problema (rate limit, ticker temporariamente indisponivel,
etc.) em vez de levantar excecao -- por isso tratamos frame vazio como erro
explicito, com retry, em vez de deixar passar como "sem dados".
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from .configuracao import MERCADO_INICIO, TICKERS_MERCADO


class ErroBuscaMercado(RuntimeError):
    pass


def buscar_ticker_yfinance(
    ticker: str,
    inicio: str = MERCADO_INICIO,
    tentativas: int = 3,
    pausa_entre_tentativas: float = 3.0,
) -> pd.DataFrame:
    """Busca um ticker via yfinance com retry manual (a lib nao expõe backoff nativo)."""
    ultimo_erro: Exception | None = None
    for tentativa in range(1, tentativas + 1):
        try:
            df = yf.download(
                ticker,
                start=inicio,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001 - yfinance levanta tipos variados
            ultimo_erro = exc
            time.sleep(pausa_entre_tentativas)
            continue

        if df is None or df.empty:
            ultimo_erro = ErroBuscaMercado(f"yfinance devolveu DataFrame vazio para {ticker!r}")
            time.sleep(pausa_entre_tentativas)
            continue

        # yfinance devolve colunas MultiIndex (ex.: ('Close', 'USDBRL=X'));
        # achatamos para o nivel de campo (Close, High, ...).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index().rename(columns={"Date": "data"})
        df["data"] = pd.to_datetime(df["data"]).dt.tz_localize(None)
        return df

    raise ErroBuscaMercado(f"Falha ao buscar {ticker!r} apos {tentativas} tentativas") from ultimo_erro


def buscar_todos_tickers(inicio: str = MERCADO_INICIO) -> dict[str, pd.DataFrame]:
    """Busca todos os tickers configurados em `TICKERS_MERCADO`."""
    resultado = {}
    for ticker, nome in TICKERS_MERCADO.items():
        df = buscar_ticker_yfinance(ticker, inicio=inicio)
        resultado[nome] = df[["data", "Close"]].rename(columns={"Close": "fechamento"})
    return resultado
