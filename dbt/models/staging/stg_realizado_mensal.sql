{% for nome, sgs in var('indicadores').items() %}
select '{{ nome }}' as indicador, data as referencia_periodo, valor as realizado
from {{ raw('sgs_' ~ sgs) }}
{% if not loop.last %}union all{% endif %}
{% endfor %}
