"""
Persistencia em Parquet com manifesto de execucao (cache-aware), para que
reexecucoes do notebook de ingestao nao rebaixem as APIs do BCB
desnecessariamente.

O manifesto (`_manifesto.json`) registra, por dataset: URL/fonte, contagem de
linhas, data minima/maxima, hash do conteudo e timestamp da coleta. Uma
dataset e considerado "fresco" (e portanto pulado numa reexecucao) se o
arquivo parquet existe e o manifesto tem menos de `validade_horas` horas --
salvo `forcar=True`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _caminho_manifesto(diretorio: Path) -> Path:
    return diretorio / "_manifesto.json"


def _carregar_manifesto(diretorio: Path) -> dict[str, Any]:
    caminho = _caminho_manifesto(diretorio)
    if caminho.exists():
        return json.loads(caminho.read_text(encoding="utf-8"))
    return {}


def _salvar_manifesto(diretorio: Path, manifesto: dict[str, Any]) -> None:
    caminho = _caminho_manifesto(diretorio)
    caminho.write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Hash estavel do conteudo (ordenado) para deteccao de mudancas reais."""
    df_ordenado = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    conteudo = pd.util.hash_pandas_object(df_ordenado, index=False).values.tobytes()
    return hashlib.sha256(conteudo).hexdigest()[:16]


def dataset_esta_fresco(
    diretorio: Path,
    nome: str,
    validade_horas: float = 24.0,
) -> bool:
    """Verifica se o parquet existe e o manifesto e recente o suficiente."""
    caminho_parquet = diretorio / f"{nome}.parquet"
    if not caminho_parquet.exists():
        return False

    manifesto = _carregar_manifesto(diretorio)
    entrada = manifesto.get(nome)
    if not entrada:
        return False

    coletado_em = datetime.fromisoformat(entrada["coletado_em"])
    return datetime.now(timezone.utc) - coletado_em < timedelta(hours=validade_horas)


def salvar_dataset(
    df: pd.DataFrame,
    diretorio: Path,
    nome: str,
    fonte: str,
    coluna_data: str | None = None,
) -> Path:
    """Persiste um DataFrame em Parquet e atualiza o manifesto do diretorio."""
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho_parquet = diretorio / f"{nome}.parquet"
    df.to_parquet(caminho_parquet, engine="pyarrow", index=False, compression="snappy")

    manifesto = _carregar_manifesto(diretorio)
    entrada: dict[str, Any] = {
        "fonte": fonte,
        "linhas": int(len(df)),
        "colunas": list(df.columns),
        "coletado_em": datetime.now(timezone.utc).isoformat(),
        "hash_conteudo": _hash_dataframe(df) if len(df) else None,
    }
    if coluna_data and coluna_data in df.columns and len(df):
        entrada["data_minima"] = str(pd.to_datetime(df[coluna_data]).min())
        entrada["data_maxima"] = str(pd.to_datetime(df[coluna_data]).max())

    manifesto[nome] = entrada
    _salvar_manifesto(diretorio, manifesto)
    return caminho_parquet


def carregar_dataset(diretorio: Path, nome: str) -> pd.DataFrame:
    caminho_parquet = diretorio / f"{nome}.parquet"
    if not caminho_parquet.exists():
        raise FileNotFoundError(
            f"Dataset '{nome}' nao encontrado em {diretorio}. Rode a ingestao primeiro."
        )
    return pd.read_parquet(caminho_parquet, engine="pyarrow")


def resumo_manifesto(diretorio: Path) -> pd.DataFrame:
    """Retorna o manifesto do diretorio como DataFrame legivel (para EDA rapido)."""
    manifesto = _carregar_manifesto(diretorio)
    if not manifesto:
        return pd.DataFrame(
            columns=["dataset", "fonte", "linhas", "coletado_em", "data_minima", "data_maxima"]
        )
    linhas = []
    for nome, entrada in manifesto.items():
        linhas.append(
            {
                "dataset": nome,
                "fonte": entrada.get("fonte"),
                "linhas": entrada.get("linhas"),
                "coletado_em": entrada.get("coletado_em"),
                "data_minima": entrada.get("data_minima"),
                "data_maxima": entrada.get("data_maxima"),
            }
        )
    return pd.DataFrame(linhas)


def obter_ou_buscar(
    diretorio: Path,
    nome: str,
    fonte: str,
    funcao_busca,
    coluna_data: str | None = None,
    validade_horas: float = 24.0,
    forcar: bool = False,
) -> pd.DataFrame:
    """
    Cache-aware fetch: se o dataset `nome` em `diretorio` estiver fresco (ver
    `dataset_esta_fresco`) e `forcar=False`, carrega do Parquet em vez de
    chamar `funcao_busca()` -- e o que torna reexecucoes do notebook de
    ingestao baratas (nao rebaixa as APIs do BCB/yfinance sem necessidade).

    `funcao_busca` e uma callable sem argumentos que devolve o DataFrame
    buscado da fonte original.
    """
    if not forcar and dataset_esta_fresco(diretorio, nome, validade_horas=validade_horas):
        return carregar_dataset(diretorio, nome)

    df = funcao_busca()
    salvar_dataset(df, diretorio, nome, fonte=fonte, coluna_data=coluna_data)
    return df
