"""
Utilitarios genericos de junção "point-in-time" (as-of), usados na Etapa 2
para unir cada feature ao painel respeitando a defasagem de publicação da
serie de origem -- o controle central de viés look-ahead do projeto.

Mantido em `src/` (e nao dentro do notebook) porque e mecanico, reutilizado
para toda feature construida em `02_transformacao.ipynb`, e e o unico lugar
onde um bug realmente compromete a validade cientifica do projeto: se
`data_disponibilidade` ficar depois de `data_travamento` para qualquer
linha, o modelo esta "vendo o futuro".
"""
from __future__ import annotations

import pandas as pd

from .configuracao import DEFASAGENS_PUBLICACAO, RegraDefasagem


def calcular_data_disponibilidade(
    serie: pd.DataFrame,
    coluna_data: str,
    nome_serie: str,
) -> pd.DataFrame:
    """
    Adiciona a coluna `data_disponibilidade` a uma serie, aplicando a regra
    de defasagem configurada em `DEFASAGENS_PUBLICACAO[nome_serie]`.

    `data_disponibilidade = coluna_data + dias_apos_fim_periodo`. Para series
    mensais (ex.: IPCA), `coluna_data` deve ja representar o FIM do periodo
    de referencia (ou o primeiro dia do mes seguinte, como o SGS devolve --
    ver ajuste em `02_transformacao.ipynb`).
    """
    if nome_serie not in DEFASAGENS_PUBLICACAO:
        raise KeyError(
            f"Sem regra de defasagem configurada para '{nome_serie}'. "
            "Adicione uma entrada em configuracao.DEFASAGENS_PUBLICACAO."
        )

    regra: RegraDefasagem = DEFASAGENS_PUBLICACAO[nome_serie]
    dias = regra.dias_apos_fim_periodo or 0

    serie = serie.copy()
    serie["data_disponibilidade"] = pd.to_datetime(serie[coluna_data]) + pd.Timedelta(days=dias)
    return serie


def juntar_as_of(
    painel: pd.DataFrame,
    coluna_ancora: str,
    serie: pd.DataFrame,
    colunas_valor: list[str],
    prefixo: str,
    validar_invariante: bool = True,
) -> pd.DataFrame:
    """
    Une `serie` (deve ja ter `data_disponibilidade`, ver
    `calcular_data_disponibilidade`) ao `painel` via merge_asof "backward":
    para cada linha do painel, pega o ultimo valor da serie cuja
    `data_disponibilidade` seja <= `coluna_ancora` (a data de travamento da
    pesquisa Focus).

    As colunas trazidas de `serie` sao renomeadas com o `prefixo` dado, para
    evitar colisao de nomes ao acumular dezenas de features no mesmo painel.

    Se `validar_invariante=True` (padrao), verifica ao final que nenhuma
    linha resultante tem `data_disponibilidade > coluna_ancora` -- e o teste
    de nao-vazamento mais importante do projeto.
    """
    colunas_novas = [f"{prefixo}_{col}" for col in colunas_valor] + [f"{prefixo}_data_disponibilidade"]
    colisoes = [c for c in colunas_novas if c in painel.columns]
    if colisoes:
        raise ValueError(
            f"juntar_as_of: prefixo='{prefixo}' colide com coluna(s) ja existente(s) no painel: "
            f"{colisoes}. Use um prefixo diferente para cada fonte unida ao mesmo painel "
            "(reusar o mesmo prefixo duas vezes sobrescreve/duplica 'data_disponibilidade')."
        )

    painel_ordenado = painel.sort_values(coluna_ancora).reset_index(drop=True)
    serie_ordenada = serie.sort_values("data_disponibilidade").reset_index(drop=True)

    resultado = pd.merge_asof(
        painel_ordenado,
        serie_ordenada[["data_disponibilidade", *colunas_valor]],
        left_on=coluna_ancora,
        right_on="data_disponibilidade",
        direction="backward",
    )

    mapa_renomeio = {col: f"{prefixo}_{col}" for col in colunas_valor}
    mapa_renomeio["data_disponibilidade"] = f"{prefixo}_data_disponibilidade"
    resultado = resultado.rename(columns=mapa_renomeio)

    if validar_invariante:
        coluna_disp = f"{prefixo}_data_disponibilidade"
        violacoes = resultado[coluna_disp] > resultado[coluna_ancora]
        # Linhas sem match (data_disponibilidade NaT) nao violam nada.
        violacoes = violacoes.fillna(False)
        if violacoes.any():
            n = int(violacoes.sum())
            raise ValueError(
                f"Viés look-ahead detectado ao juntar '{prefixo}': {n} linha(s) com "
                f"{coluna_disp} > {coluna_ancora}."
            )

    return resultado
