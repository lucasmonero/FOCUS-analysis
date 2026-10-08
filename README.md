# Qualidade do Boletim Focus e Alternativa Probabilística

*[English version](README.en.md)*

Avalia a qualidade das projeções do boletim Focus (BCB) para **IPCA, Selic, Câmbio, IGP-M e Taxa de desocupação** e propõe uma **alternativa probabilística calibrada** à projeção determinística (mediana) do boletim.

## A pergunta e o achado central

O Focus publica um ponto (a mediana) e uma dispersão entre analistas. O projeto mostra que:

1. **O ponto é difícil de bater.** O Focus supera "não prever mudança" em IPCA, Selic e IGP-M, e converge ao passeio aleatório no Câmbio em horizontes longos. Modelos de correção da média (ElasticNet/XGBoost) só melhoram significativamente o IPCA em `h=1`.
2. **A incerteza é o problema.** Um intervalo de "90%" construído com o desvio-padrão publicado cobre, na mediana dos 20 casos indicador × horizonte, **apenas 40%** das realizações. O teste de Kupiec rejeita a calibração em **20 de 20** casos, e piora com o horizonte.
3. **Discordância entre analistas não é incerteza.** Mas ela ordena bem a incerteza *relativa* (correlação positiva em 19 de 20 casos) — só erra a escala (o intervalo de 90% precisaria ser de 1,3× a 7,3× mais largo, mediana ≈ 3,2×).
4. **A melhor alternativa é a mais simples:** o próprio σ do Focus reescalado por indicador × horizonte melhora o CRPS em 8–13% e leva a cobertura de 90% de 31–58% para ~80%. Quantis históricos, XGBoost quantílico e CQR não se pagam.
5. **Em regime de virada, nada fica calibrado.** O intervalo "de 90%" cobre ~60–80%, e nenhum sinal ex-ante testado (estado do ciclo do Copom; escala dos erros recentes) o corrige a tempo. Tratar o intervalo como piso de incerteza, não como cobertura garantida.

## Estrutura

```
src/focus/      plumbing testável (120 testes): rede, OData, SGS, calendário do Copom,
                junção point-in-time, CV purgada, benchmark ingênuo, calibração, scores, conformal
notebooks/      análise completa em português
dados/          Parquet: bruto e referência versionados; processado é regenerado
dbt/            transformação em SQL (painéis point-in-time) e testes de qualidade
testes/         pytest, sem dependência de rede
docs/plano.md   plano de trabalho revisado
```

| Notebook | Conteúdo | Estado |
|---|---|---|
| `01_ingestao` | 5 indicadores Focus + realizados (SGS) + mercado + calendário do Copom | ✔ |
| `02_transformacao` | Painéis point-in-time multi-indicador, features, benchmark ingênuo | ✔ |
| `03_eda` | Qualidade do ponto **e calibração da dispersão** | ✔ |
| `04_media_condicional` | O ponto do Focus é corrigível? (resultado negativo documentado) | ✔ |
| `05_probabilistico` | Escada de alternativas probabilísticas B0 → B3 | ✔ |
| `06_diagnostico_virada` | Por que nenhum método cobre em regime de virada? | ✔ |

## Como rodar

```bash
pip install -e ".[dev]"
python -m pytest testes/
python -m focus.pipeline        # ingestão -> dbt -> OOF -> site/ (painel PT e EN)
```

O workflow `atualizar.yml` roda esse pipeline todo dia 12 (ou manualmente), versiona `dados/bruto`
e publica `site/` no GitHub Pages. `--sem-ingestao` reaproveita os dados versionados.

Os notebooks (`pip install -e ".[notebooks]"`) documentam a análise e não fazem parte do pipeline:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_ingestao.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_transformacao.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
# ... 04, 05 e 06 seguem o mesmo padrão
```

No notebook de ingestão, reexecuções são cache-aware: dentro de 24h não rebaixam as APIs.

## Armadilhas documentadas no código

Descobertas e corrigidas durante o desenvolvimento — cada uma produziria um resultado plausível porém errado:

- **Câmbio:** o Focus projeta a **média mensal**, não o fim de período (medido: MAE 26% menor). A escolha errada fabricaria um viés atribuído ao Focus.
- **Benchmark ingênuo:** deve usar o último dado **já publicado**, não `realizado(P−h)`. Na desocupação (PNADC, ~60 dias de defasagem), o benchmark injusto dava razão 2,19; o justo, 0,79.
- **OData do BCB** rejeita espaços codificados como `+` no `$filter`; exige `%20`.
- **Selic 432** é indexada pela data de vigência (pode estar no futuro); o realizado vem do calendário do Copom.
- **Pooling entre horizontes** repete cada alvo 4×; o CV purgado fatia por **data única** para não vazar o mesmo alvo entre treino e teste.

## Limitações

- A **desocupação** tem só 214 linhas (2021-10+), dominadas por uma única tendência — conclusões valem para esse regime.
- O desvio-padrão do Focus é dispersão entre analistas, não uma medida de incerteza que o BCB pretenda publicar como tal.
