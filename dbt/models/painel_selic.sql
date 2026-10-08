-- Uma linha por (reuniao do Copom, horizonte): a ULTIMA pesquisa do mes P-h.
with painel as (
    select u.*, 'selic' as indicador,
           s.data_reuniao, s.realizado,
           round(datediff('day', u.data_travamento, s.data_reuniao) / 30.4375, 2) as horizonte_m,
           s.realizado - u.mediana_focus as erro_focus
    from (
        {{ ultima_pesquisa(ref('stg_focus_selic'), 'reuniao', 'data_reuniao') }}
    ) u
    join (select distinct reuniao, data_reuniao, realizado from {{ ref('stg_focus_selic') }}) s using (reuniao)
),
enriquecido as (
    {{ enriquecer_consenso('painel', ref('stg_focus_top5_selic'), ref('stg_focus_selic'), 'reuniao') }}
)
{{ juntar_features('enriquecido') }}
order by data_travamento, horizonte
