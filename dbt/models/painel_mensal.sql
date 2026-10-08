-- Uma linha por (indicador, periodo de referencia, horizonte): a ULTIMA pesquisa do mes P-h.
with painel as (
    select u.*, r.realizado, r.realizado - u.mediana_focus as erro_focus
    from (
        {{ ultima_pesquisa(ref('stg_focus_mensal'), 'indicador, referencia_periodo', 'referencia_periodo') }}
    ) u
    join {{ ref('stg_realizado_mensal') }} r using (indicador, referencia_periodo)
),
enriquecido as (
    {{ enriquecer_consenso('painel', ref('stg_focus_top5_mensal'), ref('stg_focus_mensal'), 'referencia_periodo') }}
)
{{ juntar_features('enriquecido') }}
order by data_travamento, horizonte
