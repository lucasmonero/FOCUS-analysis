-- O Focus sai toda segunda; sem pesquisa nos ultimos 45 dias a ingestao falhou ou parou
select max(data_pesquisa) as ultima_pesquisa
from {{ ref('stg_focus_mensal') }}
having max(data_pesquisa) < current_date - interval '45 days'
