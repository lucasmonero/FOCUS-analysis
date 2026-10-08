"""
Testa a montagem da matriz de features para a Etapa 4 -- garante,
principalmente, que colunas de vazamento nunca entram em X.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focus.features_ml import (
    FEATURES_BOOLEANAS,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    preparar_matriz_features,
)


def _painel_sintetico(n=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dados = {c: rng.normal(size=n) for c in FEATURES_NUMERICAS}
    dados[FEATURES_BOOLEANAS[0]] = rng.integers(0, 2, size=n).astype(bool)
    dados[FEATURES_CATEGORICAS[0]] = rng.choice(["aperto", "afrouxamento", "manutencao"], size=n)
    # colunas que NAO devem aparecer em X (vazamento/metadado)
    dados["erro_focus"] = rng.normal(size=n)
    dados["ipca_realizado"] = rng.normal(size=n)
    dados["realizado"] = rng.normal(size=n)
    dados["referencia_periodo"] = pd.date_range("2020-01-01", periods=n, freq="MS")
    dados["horizonte"] = 12
    return pd.DataFrame(dados)


def test_colunas_de_vazamento_nunca_entram_em_x():
    painel = _painel_sintetico()
    X, cols = preparar_matriz_features(painel)
    for proibida in ["erro_focus", "realizado", "ipca_realizado", "meta_selic_realizado", "referencia_periodo", "horizonte"]:
        assert proibida not in cols, f"'{proibida}' vazou para dentro da matriz de features!"


def test_todas_as_features_numericas_presentes():
    painel = _painel_sintetico()
    X, cols = preparar_matriz_features(painel)
    for esperada in FEATURES_NUMERICAS:
        assert esperada in cols


def test_categorica_vira_dummy_sem_a_coluna_original():
    painel = _painel_sintetico()
    X, cols = preparar_matriz_features(painel)
    assert "regime_fase_ciclo" not in cols
    assert any(c.startswith("fase_") for c in cols)


def test_booleana_convertida_para_inteiro():
    painel = _painel_sintetico()
    X, cols = preparar_matriz_features(painel)
    assert X["regime_flag_pivot_copom"].dtype.kind in "iu"  # int/uint, nao bool


def test_indice_de_x_preserva_indice_do_painel():
    painel = _painel_sintetico().set_index(pd.RangeIndex(100, 120))
    X, _ = preparar_matriz_features(painel)
    assert list(X.index) == list(painel.index)


def test_coluna_faltando_levanta_erro_claro():
    painel = _painel_sintetico().drop(columns=["mediana_focus"])
    with pytest.raises(KeyError, match="mediana_focus"):
        preparar_matriz_features(painel)


def test_colunas_de_dummy_sao_identicas_mesmo_sem_todas_as_fases():
    """Regressao: com drop_first sobre categorias presentes, uma amostra sem
    'afrouxamento' (caso da desocupacao) gerava um conjunto de colunas
    diferente e trocava a categoria de referencia em silencio."""
    completo = _painel_sintetico(30)
    completo["regime_fase_ciclo"] = np.tile(["afrouxamento", "aperto", "manutencao"], 10)
    parcial = _painel_sintetico(30)
    parcial["regime_fase_ciclo"] = np.tile(["aperto", "manutencao"], 15)  # sem afrouxamento

    _, cols_completo = preparar_matriz_features(completo)
    _, cols_parcial = preparar_matriz_features(parcial)
    assert cols_completo == cols_parcial
    assert "fase_aperto" in cols_parcial and "fase_manutencao" in cols_parcial


def test_referencia_e_sempre_afrouxamento():
    """Linha em 'afrouxamento' deve ter TODAS as dummies zeradas."""
    painel = _painel_sintetico(3)
    painel["regime_fase_ciclo"] = ["afrouxamento", "aperto", "manutencao"]
    X, _ = preparar_matriz_features(painel)
    assert X.loc[0, ["fase_aperto", "fase_manutencao"]].sum() == 0
    assert X.loc[1, "fase_aperto"] == 1 and X.loc[2, "fase_manutencao"] == 1


def test_categoria_desconhecida_levanta_erro():
    painel = _painel_sintetico(3)
    painel["regime_fase_ciclo"] = ["aperto", "aperto", "categoria_nova"]
    with pytest.raises(ValueError, match="nao previstas"):
        preparar_matriz_features(painel)


def test_incluir_horizonte_adiciona_a_coluna():
    painel = _painel_sintetico(5)
    X, cols = preparar_matriz_features(painel, incluir_horizonte=True)
    assert "horizonte" in cols
    X2, cols2 = preparar_matriz_features(painel, incluir_horizonte=False)
    assert "horizonte" not in cols2
