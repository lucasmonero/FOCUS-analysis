"""
Configuracao central do pipeline: caminhos, codigos de series, horizontes de
projecao e regras de defasagem de publicacao (usadas no controle de look-ahead
bias na Etapa 2).

Todas as datas/constantes que definem "o que e a amostra" moram aqui para que
os notebooks e os modulos de src/ nunca hardcodem esses valores em mais de um
lugar.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos (raiz do projeto = dois niveis acima deste arquivo: src/focus/..)
# ---------------------------------------------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
DIR_DADOS = RAIZ_PROJETO / "dados"
DIR_BRUTO = DIR_DADOS / "bruto"
DIR_INTERMEDIARIO = DIR_DADOS / "intermediario"
DIR_PROCESSADO = DIR_DADOS / "processado"
DIR_REFERENCIA = DIR_DADOS / "referencia"

for _d in (DIR_BRUTO, DIR_INTERMEDIARIO, DIR_PROCESSADO, DIR_REFERENCIA):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Janela amostral
# ---------------------------------------------------------------------------
# 2006-01 = inicio do regime estavel de 8 reunioes Copom/ano e com IBC-Br
# (2003) e IC-Br (2005) ja totalmente disponiveis -> nenhuma feature exige
# backfill/imputacao no inicio da amostra.
AMOSTRA_INICIO = "2006-01-01"

# Horizontes de projecao (em meses) analisados em todo o pipeline.
HORIZONTES_MESES = (1, 3, 6, 12)

# ---------------------------------------------------------------------------
# Focus / Expectativas de Mercado (API OData "Olinda")
# ---------------------------------------------------------------------------
# ATENCAO: o nome correto do endpoint mensal e SINGULAR ("Expectativa"), nao
# "Expectativas...Mensais" como aparece em varias referencias na internet.
# `ExpectativasMercadoMensais` retorna HTTP 400.
ODATA_BASE_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"

ENDPOINT_FOCUS_MENSAL = "ExpectativaMercadoMensais"
ENDPOINT_FOCUS_TOP5_MENSAL = "ExpectativasMercadoTop5Mensais"
ENDPOINT_FOCUS_SELIC = "ExpectativasMercadoSelic"
ENDPOINT_FOCUS_TOP5_SELIC = "ExpectativasMercadoTop5Selic"

# baseCalculo: 0 = janela movel de 30 dias corridos: 1 = janela movel de 5
# dias uteis. Usamos 0 (amostra mais profunda / maior numero de
# respondentes). As duas linhas coexistem no mesmo (Data, DataReferencia) -
# se nao filtrar, o painel dobra silenciosamente.
BASE_CALCULO_FOCUS = 0

# tipoCalculo nas series Top5: 'C' e o unico valor observado para
# IPCA/Selic, mas filtramos explicitamente por robustez.
TIPO_CALCULO_TOP5 = "C"

# ---------------------------------------------------------------------------
# Calendario do Copom (API institucional do site do BCB, fora do Olinda)
# ---------------------------------------------------------------------------
URL_HISTORICO_TAXAS_JUROS = "https://www.bcb.gov.br/api/servico/sitebcb/historicotaxasjuros"

# A partir de 2006 ha exatamente 8 reunioes ordinarias por ano e nenhuma
# reuniao extraordinaria (as 3 extraordinarias da historia sao anteriores a
# 2003) -> permite validar o calendario derivado programaticamente.
COPOM_REUNIOES_POR_ANO_ESPERADO = 8

# ---------------------------------------------------------------------------
# SGS (Sistema Gerenciador de Series Temporais)
# ---------------------------------------------------------------------------
SGS_IPCA_MENSAL = 433          # IPCA - variacao % mensal
SGS_META_SELIC = 432           # Meta Selic definida pelo Copom (% a.a., diaria)
SGS_SELIC_ACUM_MES_ANUALIZADA = 4189  # Selic acumulada no mes, anualizada
SGS_IBCBR = 24364              # IBC-Br - indice de atividade economica
SGS_ICBR = 27574               # IC-Br - indice de commodities do BCB
SGS_META_INFLACAO = 13521      # Meta de inflacao (CMN, % a.a.)
SGS_IGPM = 189                 # IGP-M - variacao % mensal
SGS_CAMBIO_MEDIA_MENSAL = 3698 # Cambio R$/US$ (venda) - MEDIA do periodo, mensal
SGS_DESOCUPACAO = 24369        # Taxa de desocupacao - PNAD Continua (trimestre movel)

# Series diarias no SGS: a API rejeita janelas > 10 anos por requisicao.
SGS_JANELA_MAX_ANOS_SERIE_DIARIA = 10

# codigo -> (nome_canonico, e_diaria)
SERIES_SGS = {
    SGS_IPCA_MENSAL: ("ipca_mensal", False),
    SGS_META_SELIC: ("meta_selic", True),
    SGS_SELIC_ACUM_MES_ANUALIZADA: ("selic_acum_mes_anualizada", False),
    SGS_IBCBR: ("ibcbr", False),
    SGS_ICBR: ("icbr", False),
    SGS_META_INFLACAO: ("meta_inflacao", False),
    SGS_IGPM: ("igpm", False),
    SGS_CAMBIO_MEDIA_MENSAL: ("cambio", False),
    SGS_DESOCUPACAO: ("desocupacao", False),
}

# ---------------------------------------------------------------------------
# Indicadores mensais do Focus analisados, e a serie SGS do respectivo
# "realizado". Todos compartilham a MESMA forma no OData (mensais, chaveados
# por `DataReferencia`), entao o painel do IPCA generaliza direto para eles.
# A Selic e o caso a parte: chaveada por reuniao do Copom, nao por mes.
# ---------------------------------------------------------------------------
INDICADORES_MENSAIS = {
    "IPCA": {"nome": "ipca", "sgs": SGS_IPCA_MENSAL, "regra": "ipca_mensal", "unidade": "% a.m."},
    "Câmbio": {"nome": "cambio", "sgs": SGS_CAMBIO_MEDIA_MENSAL, "regra": "cambio", "unidade": "R$/US$"},
    "IGP-M": {"nome": "igpm", "sgs": SGS_IGPM, "regra": "igpm", "unidade": "% a.m."},
    "Taxa de desocupação": {"nome": "desocupacao", "sgs": SGS_DESOCUPACAO, "regra": "desocupacao", "unidade": "%"},
}

# ---------------------------------------------------------------------------
# Mercado externo (yfinance)
# ---------------------------------------------------------------------------
TICKERS_MERCADO = {
    "USDBRL=X": "usdbrl",
    "^TNX": "tnx",  # rendimento do Treasury de 10 anos dos EUA (%, x10)
}
MERCADO_INICIO = "2003-01-01"  # cobre a amostra completa com folga


# ---------------------------------------------------------------------------
# Defasagens de publicacao (dias corridos apos o fim do periodo de
# referencia) usadas para construir `data_disponibilidade` na Etapa 2.
# Valores conservadores (limite superior) para nunca "espiar" um dado antes
# de ele existir publicamente.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegraDefasagem:
    descricao: str
    dias_apos_fim_periodo: int | None = None  # None => mesmo dia do evento


DEFASAGENS_PUBLICACAO = {
    "ipca_mensal": RegraDefasagem(
        "IBGE divulga o IPCA por volta do dia 9-11 do mes seguinte; usamos "
        "dia 15 como limite seguro.",
        dias_apos_fim_periodo=15,
    ),
    "ibcbr": RegraDefasagem(
        "IBC-Br e divulgado com defasagem de ~45-50 dias apos o fim do mes.",
        dias_apos_fim_periodo=60,
    ),
    "icbr": RegraDefasagem(
        "IC-Br e divulgado no inicio do mes seguinte.",
        dias_apos_fim_periodo=15,
    ),
    "meta_selic": RegraDefasagem(
        "A decisao do Copom e publica na propria data da reuniao.",
        dias_apos_fim_periodo=0,
    ),
    "meta_inflacao": RegraDefasagem(
        "Definida pelo CMN com anos de antecedencia; sempre conhecida.",
        dias_apos_fim_periodo=0,
    ),
    "usdbrl": RegraDefasagem(
        "Cotacao de fechamento do dia util anterior (evita espiar o "
        "fechamento do proprio dia da pesquisa).",
        dias_apos_fim_periodo=0,
    ),
    "tnx": RegraDefasagem(
        "Fechamento do dia util anterior.",
        dias_apos_fim_periodo=0,
    ),
    "igpm": RegraDefasagem(
        "Caso INVERTIDO em relacao ao IPCA: o IGP-M apura do dia 21 do mes "
        "anterior ao dia 20 do mes de referencia e e divulgado ainda DENTRO "
        "do proprio mes de referencia (~dia 30). Usamos +5 dias apos o fim "
        "do periodo como limite seguro (levemente conservador).",
        dias_apos_fim_periodo=5,
    ),
    "cambio": RegraDefasagem(
        "PTAX e conhecida diariamente; a media mensal fecha no ultimo dia "
        "util do mes de referencia.",
        dias_apos_fim_periodo=1,
    ),
    "desocupacao": RegraDefasagem(
        "PNAD Continua (trimestre movel) e divulgada pelo IBGE com ~45-50 "
        "dias de defasagem; usamos 60 como limite seguro.",
        dias_apos_fim_periodo=60,
    ),
}
