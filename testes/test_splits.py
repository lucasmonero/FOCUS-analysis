"""
Testa o PurgedTimeSeriesSplit -- a peca de corretude mais critica da Etapa 4.
Sem essas garantias, uma alegacao de "o modelo bate o Focus" pode ser
artefato de vazamento de autocorrelacao serial entre treino e teste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focus.splits import PurgedTimeSeriesSplit, gerar_folds


def _datas_mensais(n: int, inicio: str = "2010-01-01") -> pd.Series:
    return pd.Series(pd.date_range(inicio, periods=n, freq="MS"))


def test_sem_sobreposicao_entre_treino_e_teste():
    datas = _datas_mensais(60)
    for idx_treino, idx_teste in PurgedTimeSeriesSplit(n_splits=5, embargo_meses=3).split(datas):
        assert set(idx_treino).isdisjoint(set(idx_teste))


def test_teste_e_sempre_posterior_ao_treino_no_tempo():
    """Nenhum fold pode ter uma observacao de treino com data POSTERIOR a
    uma observacao de teste do mesmo fold (senao seria olhar o futuro)."""
    datas = _datas_mensais(60)
    for idx_treino, idx_teste in PurgedTimeSeriesSplit(n_splits=5, embargo_meses=2).split(datas):
        data_max_treino = datas.iloc[idx_treino].max()
        data_min_teste = datas.iloc[idx_teste].min()
        assert data_max_treino < data_min_teste


def test_embargo_e_respeitado():
    """Nenhuma observacao de treino pode cair dentro dos `embargo_meses`
    imediatamente anteriores ao inicio do bloco de teste."""
    datas = _datas_mensais(60)
    embargo = 4
    for idx_treino, idx_teste in PurgedTimeSeriesSplit(n_splits=5, embargo_meses=embargo).split(datas):
        data_min_teste = datas.iloc[idx_teste].min()
        corte = data_min_teste - pd.DateOffset(months=embargo)
        data_max_treino = datas.iloc[idx_treino].max()
        assert data_max_treino < corte, (
            f"observacao de treino em {data_max_treino} viola o embargo de {embargo}m "
            f"antes do teste iniciar em {data_min_teste}"
        )


def test_embargo_maior_reduz_tamanho_do_treino():
    """Aumentar o embargo so pode remover observacoes de treino, nunca adicionar."""
    datas = _datas_mensais(60)
    folds_sem_embargo = gerar_folds(datas, n_splits=5, embargo_meses=0)
    folds_com_embargo = gerar_folds(datas, n_splits=5, embargo_meses=6)
    for (tr_sem, te_sem), (tr_com, te_com) in zip(folds_sem_embargo, folds_com_embargo):
        assert set(te_sem) == set(te_com), "o embargo nao deveria alterar o bloco de teste"
        assert len(tr_com) <= len(tr_sem)


def test_folds_cobrem_blocos_de_teste_crescentes_no_tempo():
    """Cada fold de teste deve ser cronologicamente posterior ao anterior
    (walk-forward, nao k-fold aleatorio)."""
    datas = _datas_mensais(60)
    folds = gerar_folds(datas, n_splits=4, embargo_meses=0)
    inicios_teste = [datas.iloc[idx_teste].min() for _, idx_teste in folds]
    assert inicios_teste == sorted(inicios_teste)


def test_funciona_com_datas_fora_de_ordem_na_entrada():
    """A ordem de entrada do array de datas nao deveria importar -- o split
    reordena internamente antes de fatiar."""
    datas_ordenadas = _datas_mensais(40)
    rng = np.random.default_rng(0)
    permutacao = rng.permutation(40)
    datas_embaralhadas = datas_ordenadas.iloc[permutacao].reset_index(drop=True)

    for idx_treino, idx_teste in PurgedTimeSeriesSplit(n_splits=3, embargo_meses=2).split(datas_embaralhadas):
        data_max_treino = datas_embaralhadas.iloc[idx_treino].max()
        data_min_teste = datas_embaralhadas.iloc[idx_teste].min()
        assert data_max_treino < data_min_teste


def test_n_splits_invalido_levanta_erro():
    with pytest.raises(ValueError):
        PurgedTimeSeriesSplit(n_splits=0)


def test_embargo_negativo_levanta_erro():
    with pytest.raises(ValueError):
        PurgedTimeSeriesSplit(n_splits=3, embargo_meses=-1)


def test_amostra_pequena_demais_levanta_erro_claro():
    datas = _datas_mensais(3)
    with pytest.raises(ValueError, match="pequena demais"):
        list(PurgedTimeSeriesSplit(n_splits=5, embargo_meses=0).split(datas))


def test_compativel_com_formato_esperado_pelo_sklearn():
    """cv customizado do sklearn espera uma lista de tuplas de arrays de
    indices inteiros -- confirma o formato e que os indices sao validos
    para indexar o array original."""
    datas = _datas_mensais(50)
    folds = gerar_folds(datas, n_splits=4, embargo_meses=1)
    assert isinstance(folds, list)
    for idx_treino, idx_teste in folds:
        assert isinstance(idx_treino, np.ndarray)
        assert isinstance(idx_teste, np.ndarray)
        assert idx_treino.max() < len(datas)
        assert idx_teste.max() < len(datas)


# ---------------------------------------------------------------------------
# Agrupamento por data: necessario quando o painel empilha horizontes, caso em
# que o MESMO periodo-alvo aparece varias vezes (uma por horizonte).
# ---------------------------------------------------------------------------
def _datas_empilhadas(n_datas: int, n_horizontes: int = 4) -> pd.Series:
    """Simula o painel com horizontes empilhados: cada data-alvo repetida
    `n_horizontes` vezes, em ordem embaralhada (como um concat faria)."""
    base = pd.date_range("2010-01-01", periods=n_datas, freq="MS")
    empilhado = pd.Series(list(base) * n_horizontes)
    return empilhado.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_nenhuma_data_alvo_aparece_em_treino_e_teste_ao_mesmo_tempo():
    """O vazamento que o pooling entre horizontes introduziria: uma fronteira
    de bloco cortando ao meio o grupo de linhas de um mesmo periodo-alvo."""
    datas = _datas_empilhadas(60)
    for idx_treino, idx_teste in PurgedTimeSeriesSplit(n_splits=5, embargo_meses=3).split(datas):
        datas_treino = set(datas.iloc[idx_treino])
        datas_teste = set(datas.iloc[idx_teste])
        assert datas_treino.isdisjoint(datas_teste), (
            "uma data-alvo apareceu simultaneamente em treino e teste"
        )


def test_bloco_de_teste_leva_todas_as_linhas_da_data():
    """Se uma data esta no bloco de teste, TODAS as suas linhas (todos os
    horizontes) precisam estar juntas ali."""
    n_horizontes = 4
    datas = _datas_empilhadas(60, n_horizontes=n_horizontes)
    for _, idx_teste in PurgedTimeSeriesSplit(n_splits=5, embargo_meses=3).split(datas):
        contagem = datas.iloc[idx_teste].value_counts()
        assert (contagem == n_horizontes).all(), (
            "alguma data do bloco de teste veio com linhas faltando"
        )


def test_embargo_respeitado_tambem_com_datas_repetidas():
    datas = _datas_empilhadas(60)
    embargo = 6
    for idx_treino, idx_teste in PurgedTimeSeriesSplit(n_splits=4, embargo_meses=embargo).split(datas):
        corte = datas.iloc[idx_teste].min() - pd.DateOffset(months=embargo)
        assert datas.iloc[idx_treino].max() < corte


def test_resultado_identico_ao_fatiamento_por_linha_quando_datas_sao_unicas():
    """Garantia de compatibilidade: com um horizonte so (datas unicas), o
    agrupamento por data nao muda nada."""
    datas = _datas_mensais(48)
    folds = gerar_folds(datas, n_splits=4, embargo_meses=2)
    tamanhos_teste = [len(te) for _, te in folds]
    # 48 datas / (4+1) = 9 por bloco, com o ultimo fold levando o resto
    assert tamanhos_teste[:-1] == [9, 9, 9]
    assert sum(tamanhos_teste) == 48 - 9  # o primeiro bloco fica so para treino


# ---------------------------------------------------------------------------
# folds_seguros / separar_fit_val
# ---------------------------------------------------------------------------
from focus.splits import folds_seguros, separar_fit_val


def test_folds_seguros_igual_a_gerar_folds_quando_a_amostra_comporta():
    datas = _datas_mensais(60)
    a = gerar_folds(datas, n_splits=5, embargo_meses=3)
    b = folds_seguros(datas, n_splits=5, embargo_meses=3)
    assert len(a) == len(b)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        assert list(tr_a) == list(tr_b) and list(te_a) == list(te_b)


def test_folds_seguros_degrada_em_vez_de_estourar_erro():
    """Amostra pequena demais para 5 folds (5 datas < n_splits+1 = 6):
    gerar_folds levanta erro, folds_seguros devolve menos folds."""
    datas = _datas_mensais(5)
    with pytest.raises(ValueError):
        gerar_folds(datas, n_splits=5, embargo_meses=0)
    folds = folds_seguros(datas, n_splits=5, embargo_meses=0)
    assert len(folds) >= 1


def test_folds_seguros_devolve_lista_vazia_quando_impossivel():
    assert folds_seguros(_datas_mensais(1), n_splits=5, embargo_meses=0) == []


def test_folds_seguros_preserva_o_embargo_ao_degradar():
    """O fallback nunca pode relaxar o embargo."""
    datas = _datas_mensais(14)
    for idx_treino, idx_teste in folds_seguros(datas, n_splits=5, embargo_meses=2):
        corte = datas.iloc[idx_teste].min() - pd.DateOffset(months=2)
        assert datas.iloc[idx_treino].max() < corte


def test_separar_fit_val_devolve_fatia_final_purgada():
    datas = _datas_mensais(60)
    idx_fit, idx_val = separar_fit_val(datas, embargo_meses=3)
    assert datas.iloc[idx_fit].max() < datas.iloc[idx_val].min() - pd.DateOffset(months=3)
    # a validacao e a fatia mais recente
    assert datas.iloc[idx_val].max() == datas.max()


def test_separar_fit_val_none_para_amostra_minuscula():
    assert separar_fit_val(_datas_mensais(1), embargo_meses=0) is None
