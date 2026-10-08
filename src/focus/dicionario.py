"""
Dicionario central de nomes de campo: mapeia os nomes originais (e
inconsistentes) devolvidos pelas APIs do BCB para o esquema canonico em
portugues usado em todo o pipeline (`src/focus`, notebooks e testes).

Motivacao: a API Olinda mistura convencoes de caixa entre entidades --
`ExpectativasMercadoSelic` usa `Indicador/Reuniao/Mediana` (Pascal/CamelCase),
enquanto `ExpectativasMercadoTop5Selic` usa `indicador/reuniao/mediana`
(camelCase minusculo) para os MESMOS conceitos. Sem normalizar num unico
lugar, um join entre as duas quebra silenciosamente.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Campos das entidades OData de Expectativas (Focus)
# ---------------------------------------------------------------------------
CAMPOS_FOCUS = {
    "Indicador": "indicador",
    "indicador": "indicador",
    "Data": "data_pesquisa",
    "data": "data_pesquisa",
    "DataReferencia": "referencia",
    "dataReferencia": "referencia",
    "Reuniao": "reuniao",
    "reuniao": "reuniao",
    "tipoCalculo": "tipo_calculo",
    "TipoCalculo": "tipo_calculo",
    "Media": "media",
    "media": "media",
    "Mediana": "mediana",
    "mediana": "mediana",
    "DesvioPadrao": "desvio_padrao",
    "desvioPadrao": "desvio_padrao",
    "Minimo": "minimo",
    "minimo": "minimo",
    "Maximo": "maximo",
    "maximo": "maximo",
    "numeroRespondentes": "num_respondentes",
    "NumeroRespondentes": "num_respondentes",
    "baseCalculo": "base_calculo",
    "BaseCalculo": "base_calculo",
    "coeficienteVariacao": "coef_variacao",
    "CoeficienteVariacao": "coef_variacao",
    "IndicadorDetalhe": "indicador_detalhe",
    "indicadorDetalhe": "indicador_detalhe",
    "Suavizada": "suavizada",
    "suavizada": "suavizada",
}

# ---------------------------------------------------------------------------
# Campos da API institucional `historicotaxasjuros` (calendario do Copom)
# ---------------------------------------------------------------------------
CAMPOS_COPOM = {
    "NumeroReuniaoCopom": "numero_reuniao",
    "ReuniaoExtraordinaria": "reuniao_extraordinaria",
    "DataReuniaoCopom": "data_reuniao",
    "Vies": "vies",
    "UsoMetaSelic": "uso_meta_selic",
    "DataInicioVigencia": "data_inicio_vigencia",
    "DataFimVigencia": "data_fim_vigencia",
    "MetaSelic": "meta_selic",
    "TaxaTban": "taxa_tban",
    "TaxaSelicEfetivaVigencia": "taxa_selic_efetiva_vigencia",
    "TaxaSelicEfetivaAnualizada": "taxa_selic_efetiva_anualizada",
    "descisaoMonocraticaPres": "decisao_monocratica_presidente",
}

# ---------------------------------------------------------------------------
# Series SGS: codigo -> nome canonico (ver tambem configuracao.SERIES_SGS)
# ---------------------------------------------------------------------------
CAMPOS_SGS = {
    "data": "data",
    "valor": "valor",
}

# ---------------------------------------------------------------------------
# Tickers de mercado (yfinance)
# ---------------------------------------------------------------------------
CAMPOS_MERCADO = {
    "Close": "fechamento",
    "High": "maxima",
    "Low": "minima",
    "Open": "abertura",
    "Volume": "volume",
}


def normalizar_campos(registros: list[dict], mapa: dict[str, str]) -> list[dict]:
    """Renomeia as chaves de uma lista de dicts segundo `mapa` (identidade se ausente)."""
    return [{mapa.get(k, k): v for k, v in registro.items()} for registro in registros]
