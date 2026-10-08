"""
Testa a logica de construcao e validacao do calendario do Copom SEM rede:
`construir_calendario_copom` e testado via monkeypatch de
`buscar_historico_copom`; `validar_calendario_copom` e testado com
DataFrames sinteticos.
"""
from __future__ import annotations

import pandas as pd

from focus import copom as copom_mod
from focus.copom import construir_calendario_copom, validar_calendario_copom


def _historico_sintetico() -> pd.DataFrame:
    """8 reunioes ordinarias em 2006 + 1 extraordinaria (deve ser excluida)."""
    linhas = []
    for i, (mes, dia) in enumerate(
        [(1, 18), (3, 8), (4, 19), (6, 7), (7, 19), (9, 6), (10, 18), (12, 6)], start=1
    ):
        linhas.append(
            {
                "numero_reuniao": 100 + i,
                "reuniao_extraordinaria": False,
                "data_reuniao": pd.Timestamp(2006, mes, dia),
                "meta_selic": 18.0 - i * 0.5,
                "vies": "n/a",
            }
        )
    # reuniao extraordinaria no meio do ano -- nao deve contar no rank ordinario
    linhas.append(
        {
            "numero_reuniao": 999,
            "reuniao_extraordinaria": True,
            "data_reuniao": pd.Timestamp(2006, 5, 15),
            "meta_selic": 20.0,
            "vies": "n/a",
        }
    )
    return pd.DataFrame(linhas)


def test_construir_calendario_rotula_reunioes_ordinarias_em_ordem(monkeypatch):
    monkeypatch.setattr(copom_mod, "buscar_historico_copom", lambda: _historico_sintetico())

    calendario = construir_calendario_copom("2006-01-01")

    assert len(calendario) == 8, "reuniao extraordinaria nao deveria contar no calendario ordinario"
    assert calendario["rotulo_reuniao"].tolist() == [f"R{n}/2006" for n in range(1, 9)]
    # a extraordinaria (2006-05-15) fica FORA do calendario ordinario
    assert pd.Timestamp(2006, 5, 15) not in calendario["data_reuniao"].values


def test_construir_calendario_respeita_amostra_inicio(monkeypatch):
    monkeypatch.setattr(copom_mod, "buscar_historico_copom", lambda: _historico_sintetico())

    calendario = construir_calendario_copom("2006-07-01")

    assert (calendario["data_reuniao"] >= pd.Timestamp("2006-07-01")).all()
    assert len(calendario) == 4  # 2006-07-19, 09-06, 10-18, 12-06


def _calendario_8_por_ano(anos: list[int]) -> pd.DataFrame:
    linhas = []
    for ano in anos:
        for rank in range(1, 9):
            linhas.append(
                {
                    "rotulo_reuniao": f"R{rank}/{ano}",
                    "data_reuniao": pd.Timestamp(ano, 1, 1) + pd.Timedelta(days=30 * rank),
                    "meta_selic": 10.0,
                    "ano": ano,
                    "rank_no_ano": rank,
                }
            )
    return pd.DataFrame(linhas)


def test_validar_calendario_aceita_8_reunioes_por_ano_completo():
    calendario = _calendario_8_por_ano([2006, 2007, 2008])
    resultado = validar_calendario_copom(calendario)
    assert resultado["anos_com_contagem_errada"] == {}


def test_validar_calendario_detecta_contagem_errada():
    calendario = _calendario_8_por_ano([2006, 2007])
    # remove uma reuniao de 2006 -> deveria disparar erro (ano completo, nao corrente)
    calendario = calendario[~((calendario["ano"] == 2006) & (calendario["rank_no_ano"] == 8))]
    resultado = validar_calendario_copom(calendario)
    assert 2006 in resultado["anos_com_contagem_errada"]
    assert resultado["anos_com_contagem_errada"][2006] == 7


def test_validar_calendario_distingue_orfao_genuino_de_reuniao_futura():
    # calendario cobre 2006 (ano completo e PASSADO) e 2007 (topo do calendario,
    # analogo ao "ano corrente" numa busca real).
    calendario = _calendario_8_por_ano([2006, 2007])
    # remove a 5a reuniao de 2006 do calendario, simulando uma lacuna real de
    # dado (2006 e um ano ja completo, entao isso NAO pode ser uma reuniao
    # "ainda nao realizada" -- e um orfao genuino).
    calendario = calendario[~((calendario["ano"] == 2006) & (calendario["rank_no_ano"] == 5))]

    rotulos_focus = {
        "R1/2006", "R2/2006",       # existem no calendario
        "R1/2008", "R2/2008",       # ano alem do topo do calendario -> esperado, nao e erro
        "R9/2007",                  # rank alem do maximo do TOPO do calendario -> reuniao futura
        "R5/2006",                  # lacuna dentro de um ano ja completo -> orfao genuino (bug real)
    }
    resultado = validar_calendario_copom(calendario, rotulos_observados_no_focus=rotulos_focus)

    assert resultado["rotulos_orfaos_no_focus"] == ["R5/2006"]
    assert set(resultado["rotulos_reunioes_futuras"]) == {"R1/2008", "R2/2008", "R9/2007"}
