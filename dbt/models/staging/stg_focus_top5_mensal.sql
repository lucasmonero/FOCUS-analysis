{% for nome in var('indicadores') %}
select * replace ('{{ nome }}' as indicador), strptime(referencia, '%m/%Y') as referencia_periodo
from {{ raw('focus_' ~ nome ~ '_top5_mensal') }}
{% if not loop.last %}union all by name{% endif %}
{% endfor %}
