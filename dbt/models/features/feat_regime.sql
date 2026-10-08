-- Ciclo de juros do Copom: direcao vigente = ultima variacao NAO nula da meta
with c as (
    select data_reuniao, meta_selic,
           meta_selic - lag(meta_selic) over o as delta
    from {{ ref('stg_copom') }}
    window o as (order by data_reuniao)
),
d as (
    select *,
           coalesce(last_value(nullif(sign(delta), 0) ignore nulls) over o, 0)::int as direcao_copom
    from c
    window o as (order by data_reuniao rows between unbounded preceding and current row)
),
f as (
    select *,
           direcao_copom != lag(direcao_copom) over o and direcao_copom != 0 as flag_pivot_copom
    from d
    window o as (order by data_reuniao)
),
g as (
    select *, sum(coalesce(flag_pivot_copom, false)::int) over (order by data_reuniao) as grupo from f
)
select
    {{ disponivel_em('data_reuniao', 'meta_selic') }} as disp,
    meta_selic as ultima_meta_selic_conhecida,
    direcao_copom,
    coalesce(flag_pivot_copom, false) as flag_pivot_copom,
    round(datediff('day', min(data_reuniao) over (partition by grupo), data_reuniao) / 30.4375, 1) as meses_desde_pivot,
    case direcao_copom when 1 then 'aperto' when -1 then 'afrouxamento' else 'manutencao' end as fase_ciclo,
    coalesce(sum(delta) over (partition by grupo order by data_reuniao), 0) * 100 as bps_acumulado_ciclo
from g
