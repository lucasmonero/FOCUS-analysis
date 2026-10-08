"""
Cliente OData "cru" para o servico Expectativas do BCB (Olinda), com
paginacao segura e normalizacao de nomes de campo.

Por que nao usar so o wrapper `python-bcb`? Ele e otimo para exploracao
interativa, mas esconde a paginacao e o tratamento de erro HTTP que
precisamos controlar explicitamente para uma ingestao de producao robusta
(ver `rede.py` e a estrategia de fatiamento por ano abaixo).

Estrategia de paginacao
------------------------
O Olinda limita o tamanho de pagina (~10 mil linhas) e o parametro `$skip`
degrada mal em offsets profundos sobre o historico completo (~20+ anos de
Focus). Em vez de paginar com `$skip` sobre TODO o historico, fatiamos a
requisicao por ANO no campo de data relevante (`Data`) e so entao paginamos
com `$skip` DENTRO de cada fatia anual, ate receber uma pagina mais curta que
`$top`. Isso mantem cada requisicao individual pequena, torna os retries
idempotentes e limita o dano de uma falha a um unico ano.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from .configuracao import ODATA_BASE_URL
from .rede import buscar_json, pausar_entre_requisicoes

TAMANHO_PAGINA = 10_000

from .dicionario import CAMPOS_FOCUS
from .dicionario import normalizar_campos as _normalizar_campos_generico


def normalizar_campos(registros: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renomeia as chaves de cada registro para o esquema canonico em PT."""
    return _normalizar_campos_generico(registros, CAMPOS_FOCUS)


def _montar_url(entidade: str) -> str:
    return f"{ODATA_BASE_URL}/{entidade}"


def buscar_odata_paginado(
    entidade: str,
    filtro: str | None = None,
    top: int = TAMANHO_PAGINA,
    orderby: str | None = None,
) -> list[dict[str, Any]]:
    """
    Pagina uma consulta OData com `$skip`/`$top` ate esgotar os resultados.
    Deve ser usada apenas sobre fatias JA pequenas (ex.: um ano), nunca sobre
    o historico completo de uma vez.
    """
    url = _montar_url(entidade)
    registros: list[dict[str, Any]] = []
    skip = 0

    while True:
        params = {"$format": "json", "$top": top, "$skip": skip}
        if filtro:
            params["$filter"] = filtro
        if orderby:
            params["$orderby"] = orderby

        payload = buscar_json(url, params=params)
        pagina = payload.get("value", [])
        registros.extend(pagina)

        if len(pagina) < top:
            break
        skip += top
        pausar_entre_requisicoes()

    return registros


def buscar_odata_por_ano(
    entidade: str,
    ano_inicio: int,
    ano_fim: int,
    campo_data: str = "Data",
    filtro_extra: str | None = None,
    orderby: str | None = None,
) -> pd.DataFrame:
    """
    Busca uma entidade OData fatiando por ano civil no campo `campo_data`
    (formato `YYYY-MM-DD` no filtro OData) e concatena tudo num DataFrame
    normalizado.

    `filtro_extra` e combinado com "and" ao filtro de ano (ex.:
    `"Indicador eq 'IPCA'"`).
    """
    ano_atual = dt.date.today().year
    ano_fim = min(ano_fim, ano_atual + 1)  # nunca buscar alem do ano corrente+1

    todos_registros: list[dict[str, Any]] = []
    for ano in range(ano_inicio, ano_fim + 1):
        inicio = f"{ano}-01-01"
        fim = f"{ano + 1}-01-01"
        filtro_ano = f"{campo_data} ge '{inicio}' and {campo_data} lt '{fim}'"
        filtro_completo = f"{filtro_ano} and {filtro_extra}" if filtro_extra else filtro_ano

        registros_ano = buscar_odata_paginado(entidade, filtro=filtro_completo, orderby=orderby)
        todos_registros.extend(registros_ano)
        pausar_entre_requisicoes()

    registros_normalizados = normalizar_campos(todos_registros)
    return pd.DataFrame.from_records(registros_normalizados)
