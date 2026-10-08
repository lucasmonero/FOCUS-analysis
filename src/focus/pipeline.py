"""Pipeline mensal: ingestao -> dbt (transformacao + testes) -> OOF -> site.

    python -m focus.pipeline                  # tudo
    python -m focus.pipeline --sem-ingestao   # reaproveita dados/bruto (CI, desenvolvimento)
"""
from __future__ import annotations

import argparse
import json
import os

import duckdb
from dbt.cli.main import dbtRunner

from .configuracao import (
    DEFASAGENS_PUBLICACAO,
    DIR_PROCESSADO,
    HORIZONTES_MESES,
    INDICADORES_MENSAIS,
    RAIZ_PROJETO,
    SERIES_SGS,
)
from .ingestao import ingerir
from .media_condicional import gerar_oof
from .painel import gerar_site


def _vars_dbt() -> str:
    return json.dumps({
        "indicadores": {c["nome"]: SERIES_SGS[c["sgs"]][0] for c in INDICADORES_MENSAIS.values()},
        "horizontes": list(HORIZONTES_MESES),
        "defasagem": {k: r.dias_apos_fim_periodo or 0 for k, r in DEFASAGENS_PUBLICACAO.items()},
    })


def transformar() -> None:
    """`dbt build` (modelos + testes) e exporta os paineis para `dados/processado`."""
    resultado = dbtRunner().invoke(
        ["build", "--project-dir", "dbt", "--profiles-dir", "dbt", "--vars", _vars_dbt()]
    )
    if not resultado.success:
        raise RuntimeError("dbt build falhou (modelo ou teste de qualidade); nada foi publicado")

    con = duckdb.connect("dbt/focus.duckdb")
    for nome in (c["nome"] for c in INDICADORES_MENSAIS.values()):
        con.execute(
            f"copy (select * from painel_mensal where indicador = '{nome}') "
            f"to '{DIR_PROCESSADO / f'painel_{nome}.parquet'}' (format parquet)"
        )
    con.execute(f"copy painel_selic to '{DIR_PROCESSADO / 'painel_selic.parquet'}' (format parquet)")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sem-ingestao", action="store_true")
    args = parser.parse_args()

    os.chdir(RAIZ_PROJETO)  # dbt e os caminhos do painel sao relativos a raiz
    if not args.sem_ingestao:
        ingerir()
    transformar()
    gerar_oof()
    gerar_site(RAIZ_PROJETO / "site")


if __name__ == "__main__":
    main()
