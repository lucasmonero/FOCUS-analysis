{# Parquet bruto/referencia lido direto pelo DuckDB #}
{% macro raw(nome, pasta='bruto') %}read_parquet('dados/{{ pasta }}/{{ nome }}.parquet'){% endmacro %}

{# Instante a partir do qual o dado fica publico: fim do periodo + defasagem da regra #}
{% macro disponivel_em(data, regra) %}
cast({{ data }} as timestamp) + interval '{{ var("defasagem")[regra] }} days'
{% endmacro %}

{# Mes de calendario como inteiro, para diferenca em meses #}
{% macro mes(col) %}(extract(year from {{ col }}) * 12 + extract(month from {{ col }})){% endmacro %}

{# Variacao em janelas de uma serie SGS: 3 e 6 meses (linhas = meses; para serie diaria, 91/182 linhas) #}
{% macro feat_momento(serie, regra, diaria=false) %}
select
    {{ disponivel_em('data' if diaria else 'last_day(data)', regra) }} as disp,
    valor as nivel,
    valor - lag(valor, {{ 91 if diaria else 3 }}) over o as momento_3m,
    valor - lag(valor, {{ 182 if diaria else 6 }}) over o as momento_6m
from {{ raw(serie) }}
window o as (order by data)
{% endmacro %}

{# Retorno em 21/63 linhas e volatilidade de 21 retornos diarios de uma serie de mercado #}
{% macro feat_mercado(serie, regra) %}
with r as (
    select data, fechamento,
           fechamento / lag(fechamento) over o - 1 as ret_1d
    from {{ raw(serie) }}
    window o as (order by data)
)
select
    {{ disponivel_em('data', regra) }} as disp,
    fechamento as nivel,
    (fechamento / lag(fechamento, 21) over o - 1) * 100 as retorno_1m,
    (fechamento / lag(fechamento, 63) over o - 1) * 100 as retorno_3m,
    case when count(ret_1d) over w21 = 21 then stddev_samp(ret_1d) over w21 * sqrt(21) * 100 end as vol_21d
from r
window o as (order by data), w21 as (order by data rows between 20 preceding and current row)
{% endmacro %}

{# Une as features ao painel pelo dado JA publicado em data_travamento (ASOF: sem look-ahead) #}
{% macro juntar_features(painel) %}
select
    p.*,
    rg.ultima_meta_selic_conhecida as regime_ultima_meta_selic_conhecida,
    rg.direcao_copom as regime_direcao_copom,
    rg.flag_pivot_copom as regime_flag_pivot_copom,
    rg.meses_desde_pivot as regime_meses_desde_pivot,
    rg.fase_ciclo as regime_fase_ciclo,
    rg.bps_acumulado_ciclo as regime_bps_acumulado_ciclo,
    ip.momento_ipca_3m as macro_momento_ipca_3m,
    ip.momento_ipca_6m as macro_momento_ipca_6m,
    ip.aceleracao_ipca as macro_aceleracao_ipca,
    ip.ipca_12m as macro_ipca_12m,
    mi.valor as meta_inflacao_vigente,
    ip.ipca_12m - mi.valor as hiato_ipca_meta,
    se.nivel as mmselic_nivel_selic,
    se.momento_3m as mmselic_momento_selic_3m,
    se.momento_6m as mmselic_momento_selic_6m,
    ib.momento_3m as mmibcbr_momento_ibcbr_3m,
    ib.momento_6m as mmibcbr_momento_ibcbr_6m,
    ic.momento_3m as mmicbr_momento_icbr_3m,
    ic.momento_6m as mmicbr_momento_icbr_6m,
    -- IPCA 12m realizado como proxy da inflacao esperada: juro real ex-post
    se.nivel - ip.ipca_12m as juro_real_ex_ante,
    us.retorno_1m as usdbrl_retorno_1m,
    us.retorno_3m as usdbrl_retorno_3m,
    us.vol_21d as usdbrl_vol_21d,
    tn.nivel as nivel_tnx,
    tn.retorno_1m as variacao_tnx_1m,
    tn.retorno_3m as variacao_tnx_3m
from {{ painel }} p
asof left join {{ ref('feat_regime') }} rg on p.data_travamento >= rg.disp
asof left join {{ ref('feat_ipca') }} ip on p.data_travamento >= ip.disp
asof left join {{ ref('feat_meta_inflacao') }} mi on p.data_travamento >= mi.disp
asof left join {{ ref('feat_selic') }} se on p.data_travamento >= se.disp
asof left join {{ ref('feat_ibcbr') }} ib on p.data_travamento >= ib.disp
asof left join {{ ref('feat_icbr') }} ic on p.data_travamento >= ic.disp
asof left join {{ ref('feat_usdbrl') }} us on p.data_travamento >= us.disp
asof left join {{ ref('feat_tnx') }} tn on p.data_travamento >= tn.disp
{% endmacro %}

{# Ultima pesquisa Focus de cada alvo, por horizonte em meses, com as colunas padronizadas do painel #}
{% macro ultima_pesquisa(pesquisas, alvo, mes_alvo) %}
select
    {{ alvo }}, horizonte,
    data_pesquisa as data_travamento,
    media as media_focus, mediana as mediana_focus, desvio_padrao as desvio_padrao_focus,
    minimo as minimo_focus, maximo as maximo_focus, num_respondentes as num_respondentes_focus
from (
    select *, {{ mes(mes_alvo) }} - {{ mes('data_pesquisa') }} as horizonte
    from {{ pesquisas }}
)
where horizonte in ({{ var('horizontes') | join(',') }})
qualify row_number() over (partition by {{ alvo }}, horizonte order by data_pesquisa desc) = 1
{% endmacro %}

{# Dispersao, spread vs Top 5 e revisoes (1m, 3m) do consenso para o mesmo alvo #}
{% macro enriquecer_consenso(painel, top5, pesquisas, chave) %}
select
    p.*,
    p.desvio_padrao_focus / nullif(abs(p.mediana_focus), 0) as cv_focus,
    p.maximo_focus - p.minimo_focus as amplitude_focus,
    (p.media_focus - p.mediana_focus) / nullif(p.desvio_padrao_focus, 0) as assimetria_focus,
    t.mediana as mediana_top5,
    t.mediana - p.mediana_focus as spread_top5_mediana,
    p.mediana_focus - r1.mediana as revisao_focus_1m,
    p.mediana_focus - r3.mediana as revisao_focus_3m
from {{ painel }} p
left join {{ top5 }} t on t.{{ chave }} = p.{{ chave }}
     {% if chave == 'referencia_periodo' %}and t.indicador = p.indicador{% endif %}
     and t.data_pesquisa = p.data_travamento
asof left join {{ pesquisas }} r1
     on r1.{{ chave }} = p.{{ chave }}
     {% if chave == 'referencia_periodo' %}and r1.indicador = p.indicador{% endif %}
     and p.data_travamento - interval '21 days' >= r1.data_pesquisa
asof left join {{ pesquisas }} r3
     on r3.{{ chave }} = p.{{ chave }}
     {% if chave == 'referencia_periodo' %}and r3.indicador = p.indicador{% endif %}
     and p.data_travamento - interval '63 days' >= r3.data_pesquisa
{% endmacro %}
