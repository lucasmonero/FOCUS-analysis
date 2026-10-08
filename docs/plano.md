# Replanejamento — Qualidade do Focus e Alternativa Probabilística

## Contexto

O projeto foi construído até a Etapa 4 (ingestão, painel *point-in-time*, EDA, modelagem do erro). Os resultados foram sólidos, mas expuseram um problema de **desenho**, não de execução: o projeto estava medindo o Focus na dimensão errada.

**O que os dados já mostraram:**

1. **O ponto do Focus é muito difícil de bater na média condicional.** O XGBoost só trouxe ganho estatisticamente significativo no IPCA `h=1` (DM, p=0,011). Na Selic, corrigir o Focus **piorou significativamente** em `h=1, 3, 6` (p=0,031 / 0,002 / 0,041). O ElasticNet não bateu o baseline em nenhum horizonte.
2. **Mas o erro é fortemente dependente de regime.** O viés do IPCA cresce monotonicamente com o horizonte (ME 0,044 → 0,083) e se concentra em regimes de pivot; a autocorrelação dos erros confirma comportamento adaptativo.
3. **A amostra por horizonte é pequena** (130–250 obs), o que fez os modelos de média condicional overfitarem.
4. **O Focus publica dispersão, mas entrega um ponto.** O usuário do boletim não recebe nenhum intervalo calibrado — e essa dispersão nunca foi testada.

**Conclusão que redefine o projeto:** o valor não está em corrigir o ponto do Focus (ele já é quase eficiente no primeiro momento) — está em **quantificar honestamente a incerteza em torno dele**. O segundo momento é onde há sinal; o primeiro, quase não há.

**Objetivo final revisado:**
1. Avaliar a qualidade da projeção do Focus **através de vários indicadores** — incluindo, pela primeira vez, se a dispersão que ele publica é uma medida honesta de incerteza.
2. Entregar uma **alternativa probabilística calibrada** à projeção determinística do boletim.

---

## Decisões travadas (confirmadas com o usuário)

| Decisão | Escolha |
|---|---|
| Natureza da alternativa | **Previsão probabilística calibrada** (quantis + conformal), não um ponto concorrente |
| Indicadores | **IPCA, Selic, Câmbio, IGP-M, Taxa de desocupação** |
| Nova avaliação de qualidade | **Calibração da dispersão do Focus** (discordância entre analistas ≠ incerteza) |

Fora de escopo por decisão explícita (baratos de adicionar depois): benchmarks ingênuos (random walk/AR), acurácia direcional, decomposição de Theil.

---

## Verificações feitas ao vivo nesta rodada

| Item | Achado |
|---|---|
| Indicadores no `ExpectativaMercadoMensais` | 9 disponíveis: IPCA (+5 subcomponentes), Câmbio, IGP-M, Taxa de desocupação |
| SGS dos novos realizados | `189` IGP-M, `3698` Câmbio (média mensal), `24369` Desocupação PNADC — todos retornam dados |
| **Convenção do Câmbio** | **Ambígua e crítica.** Focus `07/2026` = 5,13; SGS 3698 (média) = 5,1139; SGS 3695 (fim de período) = 5,0767. A média está mais próxima, mas **uma observação não decide** |
| XGBoost quantílico | `objective="reg:quantileerror"` com `quantile_alpha` vetorial funciona no 3.4.1 — **um modelo devolve todos os quantis de uma vez** |
| Subcomponentes do IPCA | SGS `4449` ≈ Administrados (confirmado por cruzamento); os demais não identificados — ficam fora do escopo |

---

## Arquitetura revisada

Reaproveita tudo que já existe em `src/focus/` (43 testes passando). Nada é descartado; o que muda é generalização e ênfase.

```
notebooks/
├─ 01_ingestao.ipynb        ESTENDER  → 5 indicadores + novas séries SGS
├─ 02_transformacao.ipynb   GENERALIZAR → painel multi-indicador
├─ 03_eda.ipynb             REESCREVER → multi-indicador + calibração da dispersão
├─ 04_media_condicional.ipynb  CONDENSAR → resultado honesto + pooling entre horizontes
├─ 05_probabilistico.ipynb  NOVO ★    → a alternativa: quantis + conformal
└─ 06_shap_interacoes.ipynb NOVO      → o que dirige a LARGURA do intervalo
```

---

## Etapa 1 — Ingestão multi-indicador (`01_ingestao.ipynb`)

Generalizar `src/focus/ingest_focus.py`: hoje tem `buscar_focus_ipca_mensal()` hardcoded; vira `buscar_focus_mensal(indicador)`, mantendo os filtros de `baseCalculo=0` / `tipoCalculo='C'` e a asserção de unicidade que já existe. `ingest_sgs.buscar_serie_sgs(codigo, ...)` já é genérica — só entram códigos novos em `configuracao.SERIES_SGS`.

**Verificação bloqueante — convenção do Câmbio:** antes de construir o painel, comparar o MAE do Focus contra SGS 3698 (média mensal) **e** 3695 (fim de período) ao longo de toda a amostra, e adotar a que minimizar erro sistemático. Escolher errado **fabricaria um viés artificial e o atribuiria indevidamente ao Focus** — o mesmo tipo de erro que a armadilha da vigência prospectiva da SGS 432 já quase causou na Selic.

O cache por manifesto (`armazenamento.obter_ou_buscar`) já cobre reexecuções.

## Etapa 2 — Painel multi-indicador (`02_transformacao.ipynb`)

Câmbio, IGP-M e Desocupação têm **a mesma forma do IPCA** (mensal, chaveado por `DataReferencia`), então o caminho do IPCA generaliza direto com um mapa `indicador → (código SGS, regra de defasagem)`. A Selic continua o caso especial (chaveada por reunião do Copom).

Toda a lógica *point-in-time* (`pontual.juntar_as_of`, com a asserção de não-vazamento) é reaproveitada sem mudança.

**Defasagens novas a configurar em `DEFASAGENS_PUBLICACAO`** — cada uma é uma fonte real de vazamento se errada:
- **IGP-M**: fecha no dia 20 e é divulgado **ainda dentro do mês de referência** — caso invertido em relação ao IPCA, exige atenção.
- **Câmbio**: PTAX é conhecida diariamente; a média mensal fecha no último dia útil.
- **Desocupação (PNADC)**: trimestre móvel divulgado com ~45 dias de defasagem — a mais longa do conjunto.

## Etapa 3 — Qualidade e calibração (`03_eda.ipynb`)

Reescrever o EDA atual para rodar sobre os 5 indicadores, preservando o que já funciona (perfil de erro por horizonte e regime, Mincer-Zarnowitz, Newey-West, autocorrelação, gráfico de trajetória realizado × previsto).

**Seção nova — o núcleo da avaliação de qualidade:** o Focus publica `desvio_padrao`, mín/máx e Top 5. A pergunta nunca testada é se a **discordância entre analistas** é uma medida honesta da **incerteza da projeção** — são coisas diferentes, e a literatura sugere que a primeira subestima a segunda.

Construir a distribuição implícita do Focus (`Normal(mediana, desvio_padrao)`) e testar:
- **Cobertura empírica** vs. nominal (50/80/90/95%), por indicador × horizonte
- **Kupiec** (cobertura incondicional) e **Christoffersen** (cobertura condicional + independência das violações)
- **Histograma PIT** — se calibrada, a transformada do realizado pela CDF implícita é uniforme

Hipótese a testar: intervalos sistematicamente estreitos demais. Se confirmada, **é a justificativa direta e quantificada da Etapa 5**.

## Etapa 4 — Média condicional, condensado (`04_media_condicional.ipynb`)

Renomeia e enxuga o `04_ml_modeling.ipynb` atual. Duas mudanças:
- **Pooling entre horizontes** (`horizonte` vira feature): 970 linhas por indicador em vez de ~245, atacando diretamente a limitação amostral.
- Apresentar como **resultado negativo documentado**: o primeiro momento do erro é quase não previsível, e é *por isso* que o projeto se move para a distribuição. Mantém ElasticNet, XGBoost e Diebold-Mariano já prontos.

**Ajuste necessário em `src/focus/splits.py`:** ao empilhar horizontes, cada período-alvo aparece 4 vezes. O `PurgedTimeSeriesSplit` atual fatia por posição após ordenar — uma fronteira de bloco pode **cortar o grupo de um mesmo período-alvo ao meio**, vazando entre treino e teste. Precisa de uma variante que fatie por **data-alvo única**, mantendo todas as linhas de uma data do mesmo lado, mais um teste que trave exatamente isso.

## Etapa 5 — A alternativa probabilística (`05_probabilistico.ipynb`) ★

O centro do projeto. Alvo continua `erro_focus`; a saída vira a distribuição preditiva do nível: `mediana_focus + quantis(erro | estado macro)`.

**Modelos, em ordem crescente de rigor:**

| | Abordagem | O que responde |
|---|---|---|
| **B0** | Focus + sua própria dispersão | O que um usuário do boletim faria hoje (mostrado descalibrado na Etapa 3) |
| **B1** | Focus + quantis históricos incondicionais do erro | Intervalo honesto, mas de largura fixa |
| **B2** | Regressão quantílica condicional (XGBoost multi-quantil) | Intervalo que **se adapta ao regime** — estreito em calmaria, largo em pivot |
| **B3** | **CQR** (Conformalized Quantile Regression) sobre B2 | Garantia de cobertura em amostra finita ★ entrega final |

**Por que CQR resolve a limitação central:** a garantia de cobertura do conformal é de amostra finita, não assintótica — exatamente o que faltava com 130–250 observações por horizonte.

**Ressalva metodológica que precisa ser tratada, não ignorada:** a garantia do conformal exige **permutabilidade**, que série temporal macro com mudança de regime viola. O tratamento honesto é triplo:
1. Split respeitando o tempo (treino → calibração → teste, todos purgados), nunca aleatório;
2. Reportar **cobertura condicional por regime**, não só marginal — a pergunta real é se o intervalo continua calibrado *durante o pivot*, que é justamente quando ele importa;
3. Avaliar **Adaptive Conformal Inference** (ajuste online de α em função da cobertura realizada), desenhado para deriva de distribuição.

**Avaliação** — regras de pontuação próprias, não MAE:
- **Pinball loss** por quantil e **CRPS** (aproximado pela média do pinball numa grade densa de quantis)
- **Cobertura empírica** vs. nominal, marginal e **por regime**
- **PIT** para calibração da densidade inteira
- **Sharpness** (largura média) — sujeita à calibração, mais estreito é melhor; largura sem cobertura não vale nada

## Etapa 6 — SHAP (`06_shap_interacoes.ipynb`)

Reenquadrado: em vez de explicar o erro assinado (que a Etapa 4 mostrou ser pouco previsível), explicar **a largura do intervalo** — ou seja, o que faz a incerteza explodir. Interações-alvo continuam as do plano original (`momento_ipca_3m × flag_pivot_copom`, `momento_selic_6m × desvio_padrao_focus`), agora sobre os modelos quantílicos. Manter o teto de ~25 features e subamostragem (custo `O(N·F²)`).

---

## Novos módulos em `src/focus/` (com testes, seguindo o padrão já estabelecido)

| Módulo | Conteúdo | Por que em `src/` |
|---|---|---|
| `calibracao.py` | Kupiec, Christoffersen, PIT | Fórmulas estatísticas fáceis de errar em silêncio |
| `scores.py` | Pinball, CRPS, cobertura, sharpness | Idem — e reusado por 3 notebooks |
| `conformal.py` | CQR + variante adaptativa | Núcleo da entrega; um erro aqui invalida a garantia de cobertura |
| `splits.py` | **estender** com variante por grupo de data | Corrige vazamento no pooling entre horizontes |
| `ingest_focus.py` | **generalizar** para `buscar_focus_mensal(indicador)` | Remove hardcode de IPCA/Selic |
| `features_ml.py` | **generalizar** lista de features por indicador | Hoje assume colunas do painel IPCA/Selic |

---

## Verificação

1. `python -m pytest testes/` — os 43 testes atuais continuam passando, mais os novos de `calibracao`, `scores`, `conformal` e o teste de agrupamento por data no `splits`.
2. **Teste de sinal na calibração:** um intervalo nominal de 95% construído a partir de quantis *históricos do próprio erro* deve cobrir ~95% em backtest. Se não cobrir, o pipeline de avaliação está errado — antes de concluir qualquer coisa sobre o Focus.
3. **Teste de cobertura do CQR:** por construção, o CQR deve entregar cobertura marginal ≥ nominal no conjunto de teste. Cobertura abaixo do nominal indica quebra de permutabilidade — a ser reportada, não escondida.
4. **Invariante de look-ahead** (`pontual.juntar_as_of`) continua valendo para os 5 indicadores, com as defasagens novas de IGP-M/Câmbio/PNADC.
5. Executar os 6 notebooks de ponta a ponta via `python -m nbconvert --execute`, conferindo zero erros e zero warnings (padrão mantido desde a Etapa 1).
6. Conferir contagens do painel: ~970 linhas por indicador mensal × 4 horizontes; Selic ~644.

## Riscos

- **Permutabilidade do conformal sob mudança de regime** — o risco metodológico central; mitigado por split temporal, cobertura condicional por regime e variante adaptativa, mas precisa ser declarado no relatório, não contornado.
- **Convenção do Câmbio** — resolver empiricamente antes de construir o painel (ver Etapa 1); errar aqui fabrica viés e o atribui ao Focus.
- **PNADC** — trimestre móvel com defasagem longa; a convenção de referência precisa ser confirmada contra a divulgação do IBGE.
- **Amostra ainda modesta mesmo com pooling** (~970 linhas/indicador) — é exatamente o motivo de o conformal ser a escolha certa, mas não elimina a necessidade de reportar incerteza sobre as próprias métricas.

## Ordem de execução

Etapa 1 (com a verificação do Câmbio primeiro) → Etapa 2 → `calibracao.py` + `scores.py` com testes → Etapa 3 (a calibração justifica quantitativamente o resto) → ajuste do `splits.py` → Etapa 4 → `conformal.py` com testes → Etapa 5 → Etapa 6.
