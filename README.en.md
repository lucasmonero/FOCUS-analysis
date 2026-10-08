# Focus Bulletin Quality and Probabilistic Alternative

Evaluates the quality of the Focus bulletin's (BCB — Brazil's central bank) projections for **IPCA, Selic, FX rate, IGP-M and the unemployment rate**, and proposes a **calibrated probabilistic alternative** to the bulletin's deterministic (median) projection.

## The question and the central finding

Focus publishes a point forecast (the median) and a dispersion across analysts. The project shows that:

1. **The point is hard to beat.** Focus outperforms a "no change" naive forecast for IPCA, Selic and IGP-M, and converges to a random walk for the FX rate at long horizons. Mean-correction models (ElasticNet/XGBoost) only significantly improve IPCA at `h=1`.
2. **Uncertainty is the real problem.** A "90%" interval built from the published standard deviation covers, at the median across the 20 indicator × horizon cases, **only 40%** of outcomes. The Kupiec test rejects calibration in **20 of 20** cases, and gets worse with the horizon.
3. **Disagreement among analysts is not uncertainty.** But it does rank *relative* uncertainty well (positive correlation in 19 of 20 cases) — it only gets the scale wrong (the 90% interval would need to be 1.3× to 7.3× wider, median ≈ 3.2×).
4. **The best alternative is the simplest one:** Focus's own σ, rescaled per indicator × horizon, improves CRPS by 8–13% and raises 90% coverage from 31–58% to ~80%. Historical quantiles, quantile XGBoost and CQR don't pay for their added complexity.
5. **Nothing stays calibrated during cycle turning points.** The "90%" interval covers ~60–80%, and no ex-ante signal tested (Copom cycle state; scale of recent errors) corrects it in time. Treat the interval as a floor on uncertainty, not as guaranteed coverage.

## Structure

```
src/focus/      testable plumbing (120 tests): networking, OData, SGS, Copom calendar,
                point-in-time join, purged CV, naive benchmark, calibration, scores, conformal
notebooks/      full analysis (in Portuguese)
dados/          Parquet: raw and reference are versioned; processed is regenerated
dbt/            SQL transformation (point-in-time panels) and data-quality tests
testes/         pytest, no network dependency
docs/plano.md   revised work plan
src/focus/painel.py   BI dashboard generator (PT and EN), published to GitHub Pages
```

| Notebook | Content | Status |
|---|---|---|
| `01_ingestao` | 5 Focus indicators + realized values (SGS) + market data + Copom calendar | ✔ |
| `02_transformacao` | Multi-indicator point-in-time panels, features, naive benchmark | ✔ |
| `03_eda` | Point forecast quality **and dispersion calibration** | ✔ |
| `04_media_condicional` | Is Focus's point forecast correctable? (documented negative result) | ✔ |
| `05_probabilistico` | Probabilistic alternative ladder B0 → B3 | ✔ |
| `06_diagnostico_virada` | Why does no method achieve coverage during cycle turning points? | ✔ |

## How to run

```bash
pip install -e ".[dev]"
python -m pytest testes/
python -m focus.pipeline        # ingestion -> dbt -> OOF -> site/ (PT and EN dashboard)
```

The `atualizar.yml` workflow runs this pipeline on the 12th of every month (or manually), commits
`dados/bruto` and publishes `site/` to GitHub Pages. `--sem-ingestao` reuses the versioned data.

The notebooks (`pip install -e ".[notebooks]"`) document the analysis and are not part of the pipeline:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_ingestao.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_transformacao.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
# ... 04, 05 and 06 follow the same pattern
```

In the ingestion notebook, re-runs are cache-aware: within 24h it won't re-hit the APIs.

## Pitfalls documented in the code

Discovered and fixed during development — each one would have produced a plausible but wrong result:

- **FX rate:** Focus projects the **monthly average**, not the end-of-period value (measured: 26% lower MAE). The wrong choice would fabricate a bias and attribute it to Focus.
- **Naive benchmark:** must use the last value **already published**, not `actual(P−h)`. For the unemployment rate (PNADC, ~60-day lag), the unfair benchmark gave a ratio of 2.19; the fair one, 0.79.
- **BCB's OData** rejects spaces encoded as `+` in `$filter`; it requires `%20`.
- **Selic series (SGS 432)** is indexed by the effective date (which can be in the future); the realized value comes from the Copom calendar instead.
- **Pooling across horizons** repeats each target 4×; purged CV slices by **unique date** so the same target never leaks between train and test.

## Limitations

- The **unemployment rate** has only 214 rows (2021-10 onward), dominated by a single trend — conclusions hold for that regime only.
- Focus's standard deviation is dispersion across analysts, not a measure of uncertainty that the BCB intends to publish as such.

---
*This is an English translation of [README.md](README.md); the canonical version (and all notebooks/code comments) is in Portuguese.*
