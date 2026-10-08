"""
Calendario de reunioes do Copom, construido a partir da API institucional
`historicotaxasjuros` (fora do Olinda/OData).

Essa API devolve o historico completo de decisoes do Copom com data da
reuniao e meta Selic decidida -- o suficiente para (a) mapear cada
`Reuniao` do Focus (`"R6/2026"`) para uma data de calendario e (b) usar
`MetaSelic` diretamente como "realizado" da Selic, evitando a armadilha da
serie SGS 432 (indexada pela data de VIGENCIA da meta, que pode estar no
futuro em relacao a data de coleta).
"""
from __future__ import annotations

import pandas as pd

from .configuracao import COPOM_REUNIOES_POR_ANO_ESPERADO, URL_HISTORICO_TAXAS_JUROS
from .dicionario import CAMPOS_COPOM, normalizar_campos
from .rede import buscar_json


def buscar_historico_copom() -> pd.DataFrame:
    """Busca o historico bruto de reunioes do Copom (todas as datas disponiveis)."""
    payload = buscar_json(URL_HISTORICO_TAXAS_JUROS)
    registros = payload.get("conteudo", [])
    registros_norm = normalizar_campos(registros, CAMPOS_COPOM)
    df = pd.DataFrame.from_records(registros_norm)

    df["data_reuniao"] = pd.to_datetime(df["data_reuniao"]).dt.tz_localize(None).dt.normalize()
    if "data_inicio_vigencia" in df.columns:
        df["data_inicio_vigencia"] = (
            pd.to_datetime(df["data_inicio_vigencia"]).dt.tz_localize(None).dt.normalize()
        )
    df = df.sort_values("data_reuniao").reset_index(drop=True)
    return df


def construir_calendario_copom(amostra_inicio: str) -> pd.DataFrame:
    """
    Constroi o calendario `rotulo_reuniao -> data_reuniao, meta_selic` para
    reunioes ordinarias a partir de `amostra_inicio`.

    `rotulo_reuniao` segue a convencao do Focus: `"R{n}/{ano}"`, onde `n` e o
    rank (1-indexado) da reuniao dentro do ano civil -- valido porque, a
    partir de 2006, nao ha reunioes extraordinarias no historico do Copom.
    """
    df = buscar_historico_copom()

    df_ordinarias = df.loc[
        (~df["reuniao_extraordinaria"].astype(bool))
        & (df["data_reuniao"] >= pd.Timestamp(amostra_inicio))
    ].copy()

    df_ordinarias["ano"] = df_ordinarias["data_reuniao"].dt.year
    df_ordinarias = df_ordinarias.sort_values(["ano", "data_reuniao"])
    df_ordinarias["rank_no_ano"] = df_ordinarias.groupby("ano").cumcount() + 1
    df_ordinarias["rotulo_reuniao"] = (
        "R" + df_ordinarias["rank_no_ano"].astype(str) + "/" + df_ordinarias["ano"].astype(str)
    )

    colunas = [
        "rotulo_reuniao",
        "numero_reuniao",
        "data_reuniao",
        "meta_selic",
        "vies",
        "ano",
        "rank_no_ano",
    ]
    return df_ordinarias[colunas].reset_index(drop=True)


def validar_calendario_copom(
    calendario: pd.DataFrame,
    rotulos_observados_no_focus: set[str] | None = None,
) -> dict[str, object]:
    """
    Roda as verificacoes de sanidade descritas no plano:
    1. Todo ano CIVIL COMPLETO (nao o corrente) deve ter exatamente
       `COPOM_REUNIOES_POR_ANO_ESPERADO` reunioes.
    2. Se `rotulos_observados_no_focus` for passado, todo rotulo presente no
       Focus deve existir no calendario derivado -- MAS o Focus pesquisa
       expectativas para reunioes que ainda NAO ocorreram (ate ~2 anos a
       frente), entao um "rotulo ausente" so e um problema genuino se ele
       se refere a uma reuniao que, pela contagem observada, ja deveria ter
       acontecido. Rotulos de reunioes futuras (ano > ultimo ano do
       calendario, ou rank acima do maximo ja realizado no ano corrente) sao
       esperados e reportados separadamente, sem disparar erro.

    Retorna um dict com o resultado de cada checagem, para uso tanto em
    notebook (relatorio legivel) quanto em teste automatizado (assert).
    """
    ano_corrente = pd.Timestamp.today().year
    contagem_por_ano = calendario.groupby("ano").size()
    anos_completos = contagem_por_ano[contagem_por_ano.index < ano_corrente]
    anos_com_contagem_errada = anos_completos[
        anos_completos != COPOM_REUNIOES_POR_ANO_ESPERADO
    ].to_dict()

    resultado: dict[str, object] = {
        "anos_com_contagem_errada": anos_com_contagem_errada,
        "contagem_por_ano": contagem_por_ano.to_dict(),
    }

    if rotulos_observados_no_focus is not None:
        rotulos_calendario = set(calendario["rotulo_reuniao"])
        ausentes = rotulos_observados_no_focus - rotulos_calendario

        max_ano_calendario = int(calendario["ano"].max())
        max_rank_por_ano = calendario.groupby("ano")["rank_no_ano"].max().to_dict()

        orfaos_genuinos: list[str] = []
        reunioes_futuras: list[str] = []
        for rotulo in sorted(ausentes):
            numero_str, ano_str = rotulo.lstrip("R").split("/")
            numero, ano = int(numero_str), int(ano_str)
            e_futura = ano > max_ano_calendario or (
                ano == max_ano_calendario and numero > max_rank_por_ano.get(ano, 0)
            )
            (reunioes_futuras if e_futura else orfaos_genuinos).append(rotulo)

        resultado["rotulos_orfaos_no_focus"] = orfaos_genuinos
        resultado["rotulos_reunioes_futuras"] = reunioes_futuras

    return resultado
