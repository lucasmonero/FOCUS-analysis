select {{ disponivel_em('data', 'meta_inflacao') }} as disp, valor from {{ raw('sgs_meta_inflacao') }}
