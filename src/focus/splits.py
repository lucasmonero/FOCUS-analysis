"""
Validacao cruzada "purgada" para series temporais com horizonte de projecao
sobrepondo periodos vizinhos -- a peca de corretude mais critica da Etapa 4.

Por que nao um TimeSeriesSplit comum? Erros de projecao em `h` passos a
frente sao autocorrelacionados ate a defasagem `h-1` mesmo sob eficiencia
plena do Focus (ver `03_eda.ipynb`, secao de autocorrelacao) -- e mais ainda
quando ha vies real. Um fold de treino cronologicamente colado ao inicio do
fold de teste carrega essa correlacao serial para dentro do treino,
inflando artificialmente a performance de validacao. O embargo remove do
treino qualquer observacao cujo periodo-alvo caia dentro dos `embargo_meses`
imediatamente anteriores ao inicio de cada bloco de teste.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    Walk-forward CV com embargo temporal.

    O eixo de datas e dividido em `n_splits + 1` blocos cronologicos
    contiguos (por posicao, apos ordenar): o primeiro bloco vira o treino
    inicial e cada bloco seguinte vira um fold de teste, com o treino de
    cada fold sendo tudo o que veio antes dele, MENOS as observacoes cuja
    data caia dentro do embargo imediatamente anterior ao inicio do teste.

    Parametros
    ----------
    n_splits: numero de folds de teste (>= 1).
    embargo_meses: largura do embargo, em meses de calendario. Para prever
        erro em horizonte `h`, use `embargo_meses=h` (ou `h-1`, o limite
        teorico da autocorrelacao mecanica sob eficiencia -- `h` e a
        escolha conservadora adotada no projeto).
    """

    def __init__(self, n_splits: int = 5, embargo_meses: int = 0):
        if n_splits < 1:
            raise ValueError("n_splits deve ser >= 1")
        if embargo_meses < 0:
            raise ValueError("embargo_meses nao pode ser negativo")
        self.n_splits = n_splits
        self.embargo_meses = embargo_meses

    def split(self, datas) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        `datas`: array-like de datas (uma por observacao, mesma ordem que o
        X/y que serao indexados externamente). Gera tuplas
        (indices_treino, indices_teste) como arrays de posicoes inteiras
        (0-indexed), no formato que `sklearn` espera para um `cv` custom.
        """
        serie_datas = pd.Series(pd.to_datetime(pd.Series(datas).to_numpy())).reset_index(drop=True)
        if len(serie_datas) == 0:
            raise ValueError("vetor de datas vazio")

        # O fatiamento e feito sobre DATAS UNICAS, nao sobre linhas. Quando o
        # painel empilha varios horizontes, o mesmo periodo-alvo aparece
        # varias vezes; fatiar por linha deixaria uma fronteira de bloco
        # cortar esse grupo ao meio, colocando linhas do MESMO alvo em treino
        # e teste ao mesmo tempo -- vazamento direto. Agrupando por data,
        # todas as linhas de um alvo caem sempre do mesmo lado.
        # Quando as datas ja sao unicas (um horizonte so), o resultado e
        # identico ao fatiamento por linha.
        datas_unicas = np.sort(serie_datas.unique())
        n_datas = len(datas_unicas)

        tamanho_bloco = n_datas // (self.n_splits + 1)
        if tamanho_bloco == 0:
            raise ValueError(
                f"Amostra pequena demais ({n_datas} datas unicas) para "
                f"{self.n_splits} folds -- reduza n_splits."
            )

        valores_datas = serie_datas.to_numpy()

        for i in range(self.n_splits):
            inicio_teste = tamanho_bloco * (i + 1)
            fim_teste = n_datas if i == self.n_splits - 1 else tamanho_bloco * (i + 2)
            datas_teste = datas_unicas[inicio_teste:fim_teste]

            idx_teste = np.flatnonzero(np.isin(valores_datas, datas_teste))

            data_inicio_teste = pd.Timestamp(datas_teste[0])
            corte_embargo = data_inicio_teste - pd.DateOffset(months=self.embargo_meses)
            idx_treino = np.flatnonzero(valores_datas < np.datetime64(corte_embargo))

            if len(idx_treino) == 0:
                continue  # embargo grande demais para este fold inicial -- pula

            yield idx_treino, idx_teste

    def get_n_splits(self) -> int:
        return self.n_splits


def gerar_folds(datas, n_splits: int = 5, embargo_meses: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materializa os folds de `PurgedTimeSeriesSplit` como uma lista (formato
    aceito diretamente por `cv=` em ElasticNetCV, cross_val_predict etc.)."""
    return list(PurgedTimeSeriesSplit(n_splits=n_splits, embargo_meses=embargo_meses).split(datas))


def folds_seguros(datas, n_splits: int, embargo_meses: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Como `gerar_folds`, mas degrada com elegancia em amostras pequenas.

    Tenta `n_splits` folds e, se a amostra for pequena demais, tenta
    `n_splits-1`, `n_splits-2`, ... ate obter ao menos um fold nao-vazio.
    Comum nos folds INTERNOS de uma CV aninhada, onde o treino externo ainda
    e curto. Devolve `[]` se nem com 1 fold for possivel.

    Sempre reaproveita o `PurgedTimeSeriesSplit` ja testado -- nunca uma
    logica de fallback paralela que pudesse nao respeitar o embargo.
    """
    for n in range(n_splits, 0, -1):
        try:
            folds = gerar_folds(datas, n_splits=n, embargo_meses=embargo_meses)
        except ValueError:
            continue
        if folds:
            return folds
    return []


def separar_fit_val(
    datas, embargo_meses: int, n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Separa uma fatia final PURGADA para usar como validacao de early
    stopping, reaproveitando o proprio `PurgedTimeSeriesSplit`: pede
    `n_splits` folds e usa o ULTIMO (a fatia cronologicamente mais recente,
    ja com embargo) como validacao e o treino dele como conjunto de ajuste.

    Devolve `(idx_fit, idx_val)` ou `None` se a amostra for pequena demais.
    """
    folds = folds_seguros(datas, n_splits, embargo_meses)
    return folds[-1] if folds else None
