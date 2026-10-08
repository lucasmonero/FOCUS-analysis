-- Features macro/mercado nulas em mais de 5% das linhas indicam serie vazia ou join quebrado
select 'mensal' as painel from {{ ref('painel_mensal') }}
having avg((macro_ipca_12m is null or usdbrl_retorno_3m is null or nivel_tnx is null)::int) > 0.05
union all
select 'selic' from {{ ref('painel_selic') }}
having avg((macro_ipca_12m is null or usdbrl_retorno_3m is null or nivel_tnx is null)::int) > 0.05
