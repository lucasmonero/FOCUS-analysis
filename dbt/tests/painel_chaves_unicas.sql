-- Chave duplicada indica join multiplicando linhas (ex.: Top 5 ou calendario duplicados)
select 'mensal' as painel, indicador, referencia_periodo as alvo, horizonte, count(*) as n
from {{ ref('painel_mensal') }} group by all having n > 1
union all
select 'selic', 'selic', data_reuniao, horizonte, count(*)
from {{ ref('painel_selic') }} group by all having count(*) > 1
