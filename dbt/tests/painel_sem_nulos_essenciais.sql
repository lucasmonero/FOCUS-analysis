select indicador, data_travamento from {{ ref('painel_mensal') }}
where realizado is null or mediana_focus is null or erro_focus is null
union all
select indicador, data_travamento from {{ ref('painel_selic') }}
where realizado is null or mediana_focus is null or erro_focus is null
