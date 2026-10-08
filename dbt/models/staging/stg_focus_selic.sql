select s.*, c.data_reuniao, c.meta_selic as realizado
from {{ raw('focus_selic') }} s
join {{ ref('stg_copom') }} c on c.rotulo_reuniao = s.reuniao
