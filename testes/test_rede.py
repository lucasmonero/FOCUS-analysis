"""
Testa a peculiaridade mais perigosa (e menos obvia) descoberta na API do
BCB: o parser OData do Olinda rejeita espacos codificados como '+' no
$filter, entao `rede.buscar_json` precisa forcar '%20'. Este teste trava
essa correcao contra regressao (ex.: alguem "simplificando" o codigo de
volta para `sess.get(url, params=params)`).
"""
from __future__ import annotations

from focus.rede import buscar_json


class _RespostaFalsa:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    text = '{"value": []}'
    url = "http://exemplo.test/entidade"

    def json(self):
        return {"value": []}


class _SessaoFalsa:
    def __init__(self):
        self.url_capturada = None

    def get(self, url, timeout=60):
        self.url_capturada = url
        return _RespostaFalsa()


def test_buscar_json_codifica_espacos_como_percent20():
    sessao = _SessaoFalsa()
    buscar_json(
        "http://exemplo.test/entidade",
        params={"$filter": "Data ge '2020-01-01' and Data lt '2021-01-01'"},
        sessao=sessao,
    )
    assert sessao.url_capturada is not None
    assert "+" not in sessao.url_capturada, "espacos nao devem virar '+' (quebra o parser OData do BCB)"
    assert "%20" in sessao.url_capturada


def test_buscar_json_sem_params_nao_adiciona_querystring():
    sessao = _SessaoFalsa()
    buscar_json("http://exemplo.test/entidade", sessao=sessao)
    assert sessao.url_capturada == "http://exemplo.test/entidade"
