"""
Testa a asserção de unicidade dos extratores Focus, que existe justamente
para pegar o bug de `baseCalculo`/`tipoCalculo` nao filtrado (duas linhas
por chave, painel dobrando silenciosamente).
"""
from __future__ import annotations

import pandas as pd
import pytest

from focus.ingest_focus import _asserir_sem_duplicatas


def test_asserir_sem_duplicatas_passa_com_chaves_unicas():
    df = pd.DataFrame({"data_pesquisa": ["2024-01-01", "2024-01-02"], "referencia": ["01/2024", "01/2024"]})
    _asserir_sem_duplicatas(df, ["data_pesquisa", "referencia"], "teste")  # nao deve levantar


def test_asserir_sem_duplicatas_detecta_duplicata_de_base_calculo_nao_filtrada():
    """Simula o bug real: duas linhas por (data, referencia) por causa de
    baseCalculo=0 e baseCalculo=1 nao terem sido filtradas."""
    df = pd.DataFrame(
        {
            "data_pesquisa": ["2024-01-01", "2024-01-01"],
            "referencia": ["01/2024", "01/2024"],
            "base_calculo": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="duplicadas"):
        _asserir_sem_duplicatas(df, ["data_pesquisa", "referencia"], "teste")
