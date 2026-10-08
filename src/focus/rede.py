"""
Camada HTTP compartilhada: sessao com retry/backoff para todas as chamadas a
APIs do BCB (Olinda OData, SGS, site institucional) e ao yfinance.

O Olinda limita taxa de forma agressiva e, em erro, costuma devolver corpo
HTML em vez de JSON -- por isso validamos Content-Type antes de parsear e
levantamos um erro tipado com a URL, em vez de deixar o json.loads() estourar
com uma mensagem opaca.
"""
from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ErroBuscaBCB(RuntimeError):
    """Erro ao buscar dados de uma API do BCB, com a URL de origem anexada."""

    def __init__(self, mensagem: str, url: str, status_code: int | None = None):
        super().__init__(f"{mensagem} | url={url} | status={status_code}")
        self.url = url
        self.status_code = status_code


def criar_sessao(
    total_tentativas: int = 5,
    fator_backoff: float = 1.5,
    status_para_retry: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Cria uma requests.Session com retry exponencial configurado."""
    sessao = requests.Session()
    retry = Retry(
        total=total_tentativas,
        backoff_factor=fator_backoff,
        status_forcelist=list(status_para_retry),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adaptador = HTTPAdapter(max_retries=retry)
    sessao.mount("https://", adaptador)
    sessao.mount("http://", adaptador)
    sessao.headers.update(
        {
            "User-Agent": "focus-analise/0.1 (pesquisa acadêmica; contato via repo)",
            "Accept": "application/json",
        }
    )
    return sessao


# Sessao unica reutilizada por todos os extratores do pacote.
SESSAO = criar_sessao()


def _uma_tentativa_buscar_json(
    url: str,
    params: dict[str, Any] | None,
    timeout: int,
    sess: requests.Session,
) -> Any:
    # O parser OData do BCB (Olinda) trata espacos codificados como '+'
    # (a codificacao padrao de requests.params para dict) como caractere
    # literal em vez de espaco, quebrando o `$filter` com um erro opaco
    # ("Edm.Boolean and Edm.String not compatible"). Codificamos manualmente
    # com '%20' (urllib.parse.quote) para contornar essa peculiaridade.
    if params:
        querystring = urlencode(params, quote_via=quote)
        url_completa = f"{url}?{querystring}"
        resp = sess.get(url_completa, timeout=timeout)
    else:
        resp = sess.get(url, timeout=timeout)

    if resp.status_code >= 400:
        raise ErroBuscaBCB(
            f"HTTP {resp.status_code} ao buscar dados. Corpo (300 chars): "
            f"{resp.text[:300]!r}",
            url=resp.url,
            status_code=resp.status_code,
        )

    content_type = resp.headers.get("Content-Type", "")
    if "json" not in content_type.lower():
        # Alguns endpoints do BCB nao setam Content-Type=json mesmo devolvendo
        # JSON valido; tentamos parsear mesmo assim e so falhamos se o parse
        # realmente estourar.
        try:
            return json.loads(resp.text)
        except json.JSONDecodeError as exc:
            raise ErroBuscaBCB(
                f"Resposta nao-JSON (Content-Type={content_type!r}). "
                f"Corpo (300 chars): {resp.text[:300]!r}",
                url=resp.url,
                status_code=resp.status_code,
            ) from exc

    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise ErroBuscaBCB(
            f"Falha ao decodificar JSON. Corpo (300 chars): {resp.text[:300]!r}",
            url=resp.url,
            status_code=resp.status_code,
        ) from exc


def buscar_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 60,
    sessao: requests.Session | None = None,
    tentativas: int = 4,
    pausa_entre_tentativas: float = 2.0,
) -> Any:
    """
    Busca uma URL e retorna o JSON decodificado, com validacao defensiva de
    Content-Type e retry manual.

    O retry do `HTTPAdapter` (ver `criar_sessao`) so cobre codigos de status
    HTTP (429/500/...). Na pratica, o SGS as vezes devolve uma pagina HTML de
    erro ("Requisição inválida!") com status 200 de forma transitoria (a
    mesma URL, chamada de novo alguns segundos depois, funciona) -- esse
    caso escapa do Retry do adapter e precisa de retry manual aqui.
    """
    ultimo_erro: ErroBuscaBCB | None = None
    sess = sessao or SESSAO
    for tentativa in range(1, tentativas + 1):
        try:
            return _uma_tentativa_buscar_json(url, params, timeout, sess)
        except ErroBuscaBCB as exc:
            ultimo_erro = exc
            if tentativa < tentativas:
                time.sleep(pausa_entre_tentativas * tentativa)
    assert ultimo_erro is not None
    raise ultimo_erro


def pausar_entre_requisicoes(segundos: float = 0.2) -> None:
    """Pequena pausa cortes entre chamadas em loop para nao sobrecarregar o Olinda."""
    time.sleep(segundos)
