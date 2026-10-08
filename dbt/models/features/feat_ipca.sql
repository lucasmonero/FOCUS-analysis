with r as (
    select
        {{ disponivel_em('last_day(data)', 'ipca_mensal') }} as disp, valor,
        case when count(valor) over w3 = 3 then avg(valor) over w3 end as m3,
        case when count(valor) over w6 = 6 then avg(valor) over w6 end as m6,
        case when count(valor) over w12 = 12 then (exp(sum(ln(1 + valor / 100)) over w12) - 1) * 100 end as ipca_12m
    from {{ raw('sgs_ipca_mensal') }}
    window w3 as (order by data rows between 2 preceding and current row),
           w6 as (order by data rows between 5 preceding and current row),
           w12 as (order by data rows between 11 preceding and current row)
)
select disp, m3 as momento_ipca_3m, m6 as momento_ipca_6m, m3 - m6 as aceleracao_ipca, ipca_12m from r
