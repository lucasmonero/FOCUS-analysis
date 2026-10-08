"""
Testa o benchmark ingenuo. O teste mais importante e o de defasagem: ele
reproduz exatamente a armadilha que inverteu a conclusao sobre a Taxa de
desocupacao (razao 2,19 com benchmark injusto vs 0,79 com o justo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focus.benchmark import previsao_ingenua_mensal, previsao_ingenua_selic, razao_mae


def _realizado_mensal():
    datas = pd.date_range("2024-01-01", periods=8, freq="MS")
    return pd.DataFrame({"data": datas, "valor": np.arange(1.0, 9.0)})


def test_usa_apenas_dado_ja_publicado_respeitando_a_defasagem():
    """Desocupacao tem defasagem de 60 dias: em 2024-04-30, o mes de fevereiro
    (fim 02-29, disponivel 04-29) ja saiu, mas marco (fim 03-31, disponivel
    05-30) NAO. O benchmark tem que devolver o valor de fevereiro (2.0)."""
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2024-04-30")]})
    resultado = previsao_ingenua_mensal(painel, _realizado_mensal(), "desocupacao")
    assert resultado.iloc[0] == 2.0


def test_defasagem_curta_enxerga_dado_mais_recente_que_a_longa():
    """A MESMA data de travamento, com defasagem de IPCA (15d), ja enxerga
    marco -- prova que a regra de defasagem esta de fato sendo aplicada."""
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2024-04-30")]})
    curta = previsao_ingenua_mensal(painel, _realizado_mensal(), "ipca_mensal").iloc[0]
    longa = previsao_ingenua_mensal(painel, _realizado_mensal(), "desocupacao").iloc[0]
    assert curta > longa


def test_preserva_ordem_e_indice_do_painel():
    """juntar_as_of reordena internamente; o resultado precisa voltar
    alinhado a ordem ORIGINAL do painel, senao os valores se descolam."""
    datas = [pd.Timestamp("2024-07-31"), pd.Timestamp("2024-03-31"), pd.Timestamp("2024-05-31")]
    painel = pd.DataFrame({"data_travamento": datas}, index=[10, 20, 30])
    resultado = previsao_ingenua_mensal(painel, _realizado_mensal(), "ipca_mensal")

    assert list(resultado.index) == [10, 20, 30]
    # ordem cronologica das ancoras: 03-31 < 05-31 < 07-31  => valores crescentes
    assert resultado.loc[20] < resultado.loc[30] < resultado.loc[10]


def test_nan_quando_nada_foi_publicado_ainda():
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2023-06-01")]})
    resultado = previsao_ingenua_mensal(painel, _realizado_mensal(), "ipca_mensal")
    assert pd.isna(resultado.iloc[0])


def test_previsao_ingenua_selic_usa_ultima_meta_conhecida():
    painel = pd.DataFrame({"regime_ultima_meta_selic_conhecida": [13.75, 14.0]})
    assert list(previsao_ingenua_selic(painel)) == [13.75, 14.0]


def test_razao_mae_abaixo_de_um_quando_o_focus_e_melhor():
    realizado = pd.Series([1.0, 2.0, 3.0])
    focus = realizado - pd.Series([0.1, 0.1, 0.1])       # erro_focus = +0.1
    erro_focus = realizado - focus
    ingenua = pd.Series([0.0, 0.0, 0.0])                 # erro ingenuo = 1,2,3
    assert razao_mae(erro_focus, realizado, ingenua) < 1.0


def test_razao_mae_ignora_linhas_sem_benchmark():
    realizado = pd.Series([1.0, 2.0])
    erro_focus = pd.Series([0.5, 0.5])
    ingenua = pd.Series([np.nan, 1.0])
    assert razao_mae(erro_focus, realizado, ingenua) == pytest.approx(0.5 / 1.0)
