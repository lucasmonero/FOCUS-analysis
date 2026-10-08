"""
Testa o cache-aware fetch (`obter_ou_buscar`) que torna reexecucoes do
notebook de ingestao baratas: a segunda chamada dentro da janela de validade
do manifesto NAO deve chamar a funcao de busca de novo.
"""
from __future__ import annotations

import pandas as pd

from focus.armazenamento import obter_ou_buscar


def test_obter_ou_buscar_usa_cache_na_segunda_chamada(tmp_path):
    chamadas = {"n": 0}

    def buscar():
        chamadas["n"] += 1
        return pd.DataFrame({"data": [pd.Timestamp("2024-01-01")], "valor": [1.0]})

    df1 = obter_ou_buscar(tmp_path, "serie_teste", fonte="teste", funcao_busca=buscar, coluna_data="data")
    df2 = obter_ou_buscar(tmp_path, "serie_teste", fonte="teste", funcao_busca=buscar, coluna_data="data")

    assert chamadas["n"] == 1, "a segunda chamada deveria ter usado o cache, nao rebaixado a fonte"
    pd.testing.assert_frame_equal(df1, df2)


def test_obter_ou_buscar_com_forcar_ignora_cache(tmp_path):
    chamadas = {"n": 0}

    def buscar():
        chamadas["n"] += 1
        return pd.DataFrame({"data": [pd.Timestamp("2024-01-01")], "valor": [float(chamadas["n"])]})

    obter_ou_buscar(tmp_path, "serie_teste", fonte="teste", funcao_busca=buscar, coluna_data="data")
    obter_ou_buscar(tmp_path, "serie_teste", fonte="teste", funcao_busca=buscar, coluna_data="data", forcar=True)

    assert chamadas["n"] == 2
