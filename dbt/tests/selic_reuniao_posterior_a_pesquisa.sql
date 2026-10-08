select * from {{ ref('painel_selic') }} where data_reuniao <= data_travamento
