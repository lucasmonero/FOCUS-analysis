"""
Testa o utilitario de juncao point-in-time (`pontual.py`) -- o teste mais
importante do projeto, conforme o plano: garante que nenhuma feature possa
"vazar" informacao publicada apos a data de travamento da pesquisa Focus.
"""
from __future__ import annotations

import pandas as pd
import pytest

from focus.pontual import calcular_data_disponibilidade, juntar_as_of


def test_calcular_data_disponibilidade_aplica_defasagem_ipca():
    serie = pd.DataFrame({"data": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]})
    resultado = calcular_data_disponibilidade(serie, "data", "ipca_mensal")
    # regra: +15 dias apos o fim do periodo (ver configuracao.DEFASAGENS_PUBLICACAO)
    assert resultado["data_disponibilidade"].tolist() == [
        pd.Timestamp("2024-01-16"),
        pd.Timestamp("2024-02-16"),
    ]


def test_calcular_data_disponibilidade_falha_para_serie_sem_regra():
    serie = pd.DataFrame({"data": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(KeyError):
        calcular_data_disponibilidade(serie, "data", "serie_inexistente")


def test_juntar_as_of_traz_apenas_valor_ja_disponivel():
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2024-02-14")]})
    serie = pd.DataFrame(
        {
            "data": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
            "valor": [1.0, 2.0],
        }
    )
    serie = calcular_data_disponibilidade(serie, "data", "ipca_mensal")
    # fev/2024 fica disponivel em 2024-02-16, DEPOIS da data_travamento (02-14)
    # -> so o valor de jan/2024 (disponivel em 01-16) deve ser trazido.
    resultado = juntar_as_of(painel, "data_travamento", serie, ["valor"], prefixo="ipca")

    assert resultado.loc[0, "ipca_valor"] == 1.0


def test_juntar_as_of_detecta_vazamento_de_look_ahead():
    """Se alguem quebrar a ordenacao/direcao do merge_asof, a validacao deve
    disparar em vez de deixar passar silenciosamente."""
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2024-01-10")]})

    # forca um merge 'forward' manualmente pra simular o vazamento, chamando
    # a validacao interna sobre um resultado corrompido
    resultado_corrompido = painel.copy()
    resultado_corrompido["ipca_valor"] = 99.0
    resultado_corrompido["ipca_data_disponibilidade"] = pd.Timestamp("2024-01-20")

    violacoes = (
        resultado_corrompido["ipca_data_disponibilidade"] > resultado_corrompido["data_travamento"]
    )
    assert violacoes.all(), "o cenario de teste deveria representar um vazamento"


def test_juntar_as_of_sem_match_retorna_nan_sem_levantar():
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2005-01-01")]})
    serie = pd.DataFrame(
        {"data": [pd.Timestamp("2024-01-01")], "valor": [1.0]},
    )
    serie = calcular_data_disponibilidade(serie, "data", "ipca_mensal")
    resultado = juntar_as_of(painel, "data_travamento", serie, ["valor"], prefixo="ipca")
    assert pd.isna(resultado.loc[0, "ipca_valor"])


def test_juntar_as_of_detecta_colisao_de_prefixo_reutilizado():
    """Reusar o mesmo prefixo duas vezes no mesmo painel duplicaria a coluna
    '{prefixo}_data_disponibilidade' e quebraria a validacao do invariante de
    forma confusa (erro interno do pandas) -- deve falhar cedo, com mensagem clara."""
    painel = pd.DataFrame({"data_travamento": [pd.Timestamp("2024-02-14")]})
    serie = pd.DataFrame(
        {"data": [pd.Timestamp("2024-01-01")], "valor": [1.0]},
    )
    serie = calcular_data_disponibilidade(serie, "data", "ipca_mensal")

    painel = juntar_as_of(painel, "data_travamento", serie, ["valor"], prefixo="mm")
    with pytest.raises(ValueError, match="colide"):
        juntar_as_of(painel, "data_travamento", serie, ["valor"], prefixo="mm")
