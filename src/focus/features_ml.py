"""
Selecao e preparo da matriz de features (X) para a Etapa 4 (modelagem) a
partir de `painel_ipca`/`painel_selic`.

Mantido em `src/` (em vez de repetido em `04_ml_modeling.ipynb` e
`05_shap_interacoes.ipynb`) porque as duas etapas precisam construir
EXATAMENTE a mesma matriz a partir do mesmo painel -- duplicar essa lista
em dois notebooks seria um risco real de dessincronia (alguem atualiza um
e esquece o outro).

A lista de features e um INCLUDE list explicito (em vez de "todas as
colunas menos X") de proposito: e a forma mais auditavel de garantir que
nenhuma coluna de vazamento (`erro_focus`, `ipca_realizado`,
`meta_selic_realizado`, identificadores/datas) entre na matriz, mesmo que o
esquema do painel mude no futuro.
"""
from __future__ import annotations

import pandas as pd

# Features numericas continuas, todas conhecidas em `data_travamento` (ver
# Etapa 2). `mediana_focus` e legitima como feature: nao estamos prevendo o
# Focus a partir dele mesmo, estamos testando se o proprio nivel/dispersao
# do consenso ajuda a explicar o ERRO subsequente.
FEATURES_NUMERICAS = [
    "mediana_focus",
    "desvio_padrao_focus",
    "num_respondentes_focus",
    "cv_focus",
    "amplitude_focus",
    "assimetria_focus",
    "spread_top5_mediana",
    "revisao_focus_1m",
    "revisao_focus_3m",
    "regime_ultima_meta_selic_conhecida",
    "regime_direcao_copom",
    "regime_meses_desde_pivot",
    "regime_bps_acumulado_ciclo",
    "macro_momento_ipca_3m",
    "macro_momento_ipca_6m",
    "macro_aceleracao_ipca",
    "macro_ipca_12m",
    "meta_inflacao_vigente",
    "hiato_ipca_meta",
    "mmselic_nivel_selic",
    "mmselic_momento_selic_3m",
    "mmselic_momento_selic_6m",
    "mmibcbr_momento_ibcbr_3m",
    "mmibcbr_momento_ibcbr_6m",
    "mmicbr_momento_icbr_3m",
    "mmicbr_momento_icbr_6m",
    "juro_real_ex_ante",
    "usdbrl_retorno_1m",
    "usdbrl_retorno_3m",
    "usdbrl_vol_21d",
    "nivel_tnx",
    "variacao_tnx_1m",
    "variacao_tnx_3m",
]

FEATURES_BOOLEANAS = ["regime_flag_pivot_copom"]
FEATURES_CATEGORICAS = ["regime_fase_ciclo"]

# Categorias FIXAS da fase do ciclo, com "afrouxamento" como referencia.
# Sem isso, `get_dummies(drop_first=True)` descarta a primeira categoria
# PRESENTE na amostra -- e uma amostra sem "afrouxamento" (ex.: a de
# desocupacao, so 2021+) mudaria a categoria de referencia em silencio,
# tornando os coeficientes nao comparaveis entre indicadores.
CATEGORIAS_FASE_CICLO = ["afrouxamento", "aperto", "manutencao"]

# Colunas deliberadamente EXCLUIDAS (documentado para nao serem "descobertas"
# e adicionadas de volta por engano):
#   - media_focus, minimo_focus, maximo_focus, mediana_top5: redundantes com
#     mediana_focus + cv_focus + amplitude_focus + assimetria_focus + spread_top5_mediana.
#   - realizado, erro_focus: vazamento direto do alvo (erro_focus = realizado - mediana).
#   - referencia_periodo, reuniao, data_reuniao, data_travamento, horizonte,
#     horizonte_m: identificadores/datas, nao features numericas.


def preparar_matriz_features(
    painel: pd.DataFrame, incluir_horizonte: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retorna `(X, nomes_colunas)`, com:
      - features numericas mantidas como float (NaN preservado -- XGBoost
        lida nativamente; ElasticNet precisa de imputacao, feita no Pipeline
        de `04_ml_modeling.ipynb`, nunca aqui, para nao vazar estatisticas
        de todo o painel para dentro de um fold de treino/teste);
      - a booleana de pivot convertida para 0/1;
      - a categorica de fase do ciclo convertida em dummies com categorias
        FIXAS (`afrouxamento` como referencia) -- o conjunto de colunas e
        identico em qualquer painel, inclusive um que nao contenha todas as
        fases;
      - se `incluir_horizonte=True`, a coluna `horizonte` entra como feature
        (necessario ao empilhar horizontes num unico modelo).
    """
    faltando = [
        c for c in FEATURES_NUMERICAS + FEATURES_BOOLEANAS + FEATURES_CATEGORICAS
        if c not in painel.columns
    ]
    if faltando:
        raise KeyError(f"Colunas esperadas ausentes do painel: {faltando}")

    X_num = painel[FEATURES_NUMERICAS].astype(float)
    X_bool = painel[FEATURES_BOOLEANAS].astype(int)

    fase = pd.Categorical(painel["regime_fase_ciclo"], categories=CATEGORIAS_FASE_CICLO)
    desconhecidas = set(painel["regime_fase_ciclo"].dropna()) - set(CATEGORIAS_FASE_CICLO)
    if desconhecidas:
        raise ValueError(f"Categorias de fase_ciclo nao previstas: {sorted(desconhecidas)}")
    X_cat = pd.get_dummies(
        pd.DataFrame({"regime_fase_ciclo": fase}, index=painel.index),
        prefix="fase", drop_first=True,
    ).astype(int)

    X = pd.concat([X_num, X_bool, X_cat], axis=1)
    X.index = painel.index

    if incluir_horizonte:
        X["horizonte"] = painel["horizonte"].astype(float)

    return X, list(X.columns)
