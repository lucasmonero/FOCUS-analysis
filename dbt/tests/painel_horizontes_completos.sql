-- Todo indicador precisa ter todos os horizontes configurados
with esperado as (select unnest([{{ var('horizontes') | join(',') }}]) as horizonte),
presente as (
    select indicador, horizonte from {{ ref('painel_mensal') }}
    union select indicador, horizonte from {{ ref('painel_selic') }}
)
select i.indicador, e.horizonte
from (select distinct indicador from presente) i cross join esperado e
left join presente p using (indicador, horizonte)
where p.horizonte is null
