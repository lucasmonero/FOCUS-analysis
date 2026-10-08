"""
Testa a normalizacao de campos que cobre a inconsistencia de caixa entre
entidades OData do Focus (ex.: `Mediana` vs `mediana`, `Reuniao` vs
`reuniao`) -- sem essa normalizacao, um join entre `ExpectativasMercadoSelic`
e `ExpectativasMercadoTop5Selic` quebra silenciosamente.
"""
from __future__ import annotations

from focus.dicionario import CAMPOS_FOCUS, normalizar_campos


def test_normaliza_campos_pascal_case():
    registros = [{"Indicador": "IPCA", "Data": "2024-01-01", "Mediana": 0.5}]
    resultado = normalizar_campos(registros, CAMPOS_FOCUS)
    assert resultado == [{"indicador": "IPCA", "data_pesquisa": "2024-01-01", "mediana": 0.5}]


def test_normaliza_campos_camel_case_minusculo():
    registros = [{"indicador": "Selic", "reuniao": "R1/2026", "mediana": 13.75}]
    resultado = normalizar_campos(registros, CAMPOS_FOCUS)
    assert resultado == [{"indicador": "Selic", "reuniao": "R1/2026", "mediana": 13.75}]


def test_pascal_e_camel_convergem_para_o_mesmo_nome_canonico():
    """As duas entidades Selic usam caixas diferentes para o mesmo conceito;
    apos normalizar, as chaves resultantes devem ser identicas."""
    registro_pascal = {"Reuniao": "R1/2026", "Mediana": 13.75}
    registro_camel = {"reuniao": "R1/2026", "mediana": 13.75}

    norm_pascal = normalizar_campos([registro_pascal], CAMPOS_FOCUS)[0]
    norm_camel = normalizar_campos([registro_camel], CAMPOS_FOCUS)[0]

    assert set(norm_pascal.keys()) == set(norm_camel.keys()) == {"reuniao", "mediana"}


def test_campo_desconhecido_permanece_inalterado():
    registros = [{"CampoNovoDesconhecido": 123}]
    resultado = normalizar_campos(registros, CAMPOS_FOCUS)
    assert resultado == [{"CampoNovoDesconhecido": 123}]
