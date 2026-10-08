"""Painel de acompanhamento do Focus (HTML autonomo, PT e EN).

Le `dados/processado` e `dados/bruto`, calibra `k` (escala da dispersao do Focus) por
indicador x horizonte e escreve uma pagina com plotly.js embutido.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
import statsmodels.api as sm
from plotly.subplots import make_subplots

from .configuracao import DIR_BRUTO, DIR_PROCESSADO, HORIZONTES_MESES
from .probabilistico import TAUS_PADRAO, ajustar_fator_sigma

pio.templates.default = "plotly_white"

ORDEM = ["ipca", "cambio", "igpm", "selic"]
CORES_H = {1: "#1f77b4", 3: "#ff7f0e", 6: "#2ca02c", 12: "#d62728"}
Z90 = 1.6448536269514722  # norm.ppf(0.95), banda simetrica de 90%
Z70 = 1.0364333894937898  # norm.ppf(0.85), banda simetrica de 70%

# Dimensoes travadas com o usuario -- reduzidas p/ caber numa pagina sem scroll
LARGURA_GRAFICO_HIST = 1200
ALTURA_GRAFICO_HIST = 300
ALTURA_GRAFICO_FASE = 230
ALTURA_CARTAO = 100


def carregar_painel(nome: str) -> pd.DataFrame:
    df = pd.read_parquet(DIR_PROCESSADO / f"painel_{nome}.parquet")
    if "referencia_periodo" not in df.columns:
        df = df.rename(columns={"data_reuniao": "referencia_periodo"})
    return df


def calibrar_k(painel: pd.DataFrame) -> dict[int, float]:
    """k por horizonte, ajustado sobre TODA a amostra (calibração in-sample -- ver riscos do plano)."""
    ks = {}
    for h in HORIZONTES_MESES:
        sub = painel[painel["horizonte"] == h]
        try:
            ks[h] = ajustar_fator_sigma(
                sub["erro_focus"].to_numpy(), sub["desvio_padrao_focus"].to_numpy(), taus=TAUS_PADRAO
            )
        except ValueError:
            ks[h] = np.nan
    return ks


def erro_por_fase(painel: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    p = painel[painel["regime_fase_ciclo"].isin(["aperto", "afrouxamento"])].copy()
    p["D_aperto"] = (p["regime_fase_ciclo"] == "aperto").astype(float)
    for h in HORIZONTES_MESES:
        s = p[p["horizonte"] == h].sort_values("referencia_periodo")
        n_ap, n_af = int(s["D_aperto"].sum()), int((1 - s["D_aperto"]).sum())
        media_ap = s.loc[s["D_aperto"] == 1, "erro_focus"].mean()
        media_af = s.loc[s["D_aperto"] == 0, "erro_focus"].mean()
        if n_ap < 5 or n_af < 5:
            linhas.append(
                {"horizonte": h, "n_aperto": n_ap, "n_afrouxamento": n_af, "media_aperto": media_ap,
                 "media_afrouxamento": media_af, "delta": np.nan, "p_hac": np.nan,
                 "erros_aperto": s.loc[s["D_aperto"] == 1, "erro_focus"].to_numpy(),
                 "erros_afrouxamento": s.loc[s["D_aperto"] == 0, "erro_focus"].to_numpy()}
            )
            continue
        X = sm.add_constant(s["D_aperto"].to_numpy())
        m = sm.OLS(s["erro_focus"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        linhas.append(
            {
                "horizonte": h,
                "n_aperto": n_ap,
                "n_afrouxamento": n_af,
                "media_aperto": media_ap,
                "media_afrouxamento": media_af,
                "delta": m.params[1],
                "p_hac": m.pvalues[1],
                "erros_aperto": s.loc[s["D_aperto"] == 1, "erro_focus"].to_numpy(),
                "erros_afrouxamento": s.loc[s["D_aperto"] == 0, "erro_focus"].to_numpy(),
            }
        )
    return pd.DataFrame(linhas).set_index("horizonte")


def _mes_para_data(referencia: str) -> pd.Timestamp:
    mes, ano = referencia.split("/")
    return pd.Timestamp(year=int(ano), month=int(mes), day=1)


def projecao_vigente_mensal(nome: str, ks: dict[int, float]) -> tuple[pd.DataFrame, pd.Timestamp]:
    bruto = pd.read_parquet(DIR_BRUTO / f"focus_{nome}_mensal.parquet")
    ult = bruto["data_pesquisa"].max()
    viv = bruto[bruto["data_pesquisa"] == ult].copy()
    viv["data_alvo"] = viv["referencia"].apply(_mes_para_data)
    mes_pesquisa = pd.Timestamp(year=ult.year, month=ult.month, day=1)
    viv["horizonte_meses"] = (
        (viv["data_alvo"].dt.year - mes_pesquisa.year) * 12 + (viv["data_alvo"].dt.month - mes_pesquisa.month)
    )
    viv = viv[(viv["horizonte_meses"] >= 1) & (viv["horizonte_meses"] <= 24)].sort_values("data_alvo")

    def k_para(h):
        if h in ks and np.isfinite(ks[h]):
            return ks[h], False
        return ks.get(12, 1.0), True

    kk, extra = zip(*viv["horizonte_meses"].apply(lambda h: k_para(min(h, 12) if h <= 12 else 12)))
    viv["k"] = kk
    viv["extrapolado"] = viv["horizonte_meses"] > 12
    viv["q_inf_90"] = viv["mediana"] - Z90 * viv["k"] * viv["desvio_padrao"]
    viv["q_sup_90"] = viv["mediana"] + Z90 * viv["k"] * viv["desvio_padrao"]
    viv["q_inf_70"] = viv["mediana"] - Z70 * viv["k"] * viv["desvio_padrao"]
    viv["q_sup_70"] = viv["mediana"] + Z70 * viv["k"] * viv["desvio_padrao"]
    return viv, ult


def projecao_vigente_selic(painel: pd.DataFrame, ks: dict[int, float]) -> tuple[pd.DataFrame, pd.Timestamp]:
    bruto = pd.read_parquet(DIR_BRUTO / "focus_selic.parquet")
    ult = bruto["data_pesquisa"].max()
    viv = bruto[bruto["data_pesquisa"] == ult].copy()

    # Simplificacao: sem calendario Copom p/ reunioes futuras
    # (o historico oficial so cobre reunioes JA decididas). Ordena os rotulos
    # (ano, rank) e usa a ORDEM como proxy de "numero de reunioes a frente" --
    # suficiente para o proposito do painel.
    def rank(rot):
        n, a = rot.lstrip("R").split("/")
        return (int(a), int(n))

    viv["_rank"] = viv["reuniao"].apply(rank)
    ultima_reuniao_resolvida = painel["referencia_periodo"].max()
    viv = viv.sort_values("_rank").reset_index(drop=True)
    viv["reunioes_a_frente"] = np.arange(1, len(viv) + 1)
    viv = viv[viv["reunioes_a_frente"] <= 12]

    def k_para(n):
        h = n if n in HORIZONTES_MESES else min(HORIZONTES_MESES, key=lambda x: abs(x - n))
        if h in ks and np.isfinite(ks[h]):
            return ks[h]
        return ks.get(12, 1.0)

    viv["k"] = viv["reunioes_a_frente"].apply(k_para)
    viv["extrapolado"] = False
    viv["q_inf_90"] = viv["mediana"] - Z90 * viv["k"] * viv["desvio_padrao"]
    viv["q_sup_90"] = viv["mediana"] + Z90 * viv["k"] * viv["desvio_padrao"]
    viv["q_inf_70"] = viv["mediana"] - Z70 * viv["k"] * viv["desvio_padrao"]
    viv["q_sup_70"] = viv["mediana"] + Z70 * viv["k"] * viv["desvio_padrao"]
    # data_alvo aproximada: espaça 45 dias por reuniao a partir da ultima resolvida
    viv["data_alvo"] = ultima_reuniao_resolvida + pd.to_timedelta(viv["reunioes_a_frente"] * 45, unit="D")
    return viv, ult


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def fig_historico_projecao(nome: str, painel: pd.DataFrame, viv: pd.DataFrame) -> tuple[go.Figure, dict[int, int]]:
    """Grafico unico: TODO o historico realizado + Focus por horizonte (toggle externo) + projecao vigente."""
    hist = painel.drop_duplicates("referencia_periodo").sort_values("referencia_periodo")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["referencia_periodo"], y=hist["realizado"], name=T["realizado"],
                              mode="lines", line=dict(color="#333", width=1.8)))

    idx_horizonte: dict[int, int] = {}
    for h in HORIZONTES_MESES:
        sub = painel[painel["horizonte"] == h].sort_values("referencia_periodo")
        fig.add_trace(go.Scatter(x=sub["referencia_periodo"], y=sub["mediana_focus"], name=f"Focus h={h}",
                                  mode="lines", line=dict(color=CORES_H[h], dash="dot", width=1.2), visible=True))
        idx_horizonte[h] = len(fig.data) - 1

    normal = viv[~viv["extrapolado"]]
    extra = viv[viv["extrapolado"]]
    cor_faixa = "31,119,180"
    # Ordem de desenho: 90% (mais larga, mais clara) primeiro, 70% (mais estreita, mais opaca) por cima.
    for grupo, q_inf_col, q_sup_col, nome_legenda, opacidade, mostrar_legenda in [
        (normal, "q_inf_90", "q_sup_90", T["faixa90"], 0.15, True),
        (extra, "q_inf_90", "q_sup_90", T["extra90"], 0.08, True),
        (normal, "q_inf_70", "q_sup_70", T["faixa70"], 0.30, True),
        (extra, "q_inf_70", "q_sup_70", T["extra70"], 0.16, False),
    ]:
        if grupo.empty:
            continue
        fig.add_trace(go.Scatter(x=pd.concat([grupo["data_alvo"], grupo["data_alvo"][::-1]]),
                                  y=pd.concat([grupo[q_sup_col], grupo[q_inf_col][::-1]]),
                                  fill="toself", fillcolor=f"rgba({cor_faixa},{opacidade})",
                                  line=dict(color="rgba(0,0,0,0)"), name=nome_legenda,
                                  hoverinfo="skip", showlegend=mostrar_legenda))
    fig.add_trace(go.Scatter(x=viv["data_alvo"], y=viv["mediana"], name=T["focus_vigente"],
                              mode="lines+markers", line=dict(color="#1f77b4", dash="dash"), marker=dict(size=4)))

    fig.add_vline(x=hist["referencia_periodo"].max(), line_dash="dot", line_color="gray",
                  annotation_text=T["hoje"], annotation_position="top")

    fig.update_layout(
        title=dict(text=f"{ROTULOS[nome]} — {T['titulo_hist']} ({UNIDADES[nome]})", font=dict(size=13)),
        width=LARGURA_GRAFICO_HIST, height=ALTURA_GRAFICO_HIST,
        margin=dict(t=30, b=55, l=45, r=15),
        legend=dict(orientation="h", y=-0.30, font=dict(size=10)),
        font=dict(size=11),
    )
    return fig, idx_horizonte


def _titulo_subplot_fase(row: pd.Series, h: int) -> str:
    if not np.isfinite(row["p_hac"]):
        return f"h={h} ({T['amostra_insuf']})"
    sig = row["p_hac"] < 0.05
    marca = T["sig"] if sig else "n.s."
    return f"h={h} · {marca} (p={row['p_hac']:.3f})"


def fig_distribuicao_fase(nome: str, fase: pd.DataFrame) -> go.Figure:
    horizontes_validos = [h for h in HORIZONTES_MESES if h in fase.index]
    fig = make_subplots(
        rows=1, cols=len(horizontes_validos), shared_yaxes=True,
        subplot_titles=[_titulo_subplot_fase(fase.loc[h], h) for h in horizontes_validos],
    )
    for i, h in enumerate(horizontes_validos, start=1):
        row = fase.loc[h]
        fig.add_trace(go.Histogram(x=row["erros_aperto"], name=T["aperto"], marker_color="#d62728",
                                    opacity=0.6, legendgroup="aperto", showlegend=(i == 1), nbinsx=12),
                      row=1, col=i)
        fig.add_trace(go.Histogram(x=row["erros_afrouxamento"], name=T["afrouxamento"], marker_color="#1f77b4",
                                    opacity=0.6, legendgroup="afrouxamento", showlegend=(i == 1), nbinsx=12),
                      row=1, col=i)
    fig.update_layout(
        barmode="overlay", height=ALTURA_GRAFICO_FASE,
        margin=dict(t=30, b=35, l=35, r=15), legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
        font=dict(size=10),
    )
    fig.update_annotations(font_size=10)
    return fig


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------

def checkboxes_horizonte(nome: str) -> str:
    itens = "".join(
        f'<label class="chk-horizonte"><input type="checkbox" checked '
        f'onchange="alternarHorizonte(\'{nome}\', {h}, this.checked)"> h={h}</label>'
        for h in HORIZONTES_MESES
    )
    return f'<div class="seletor-horizonte"><span>{T["horizontes"]}</span>{itens}</div>'


def cartao_html(nome: str, painel: pd.DataFrame, viv: pd.DataFrame) -> str:
    hist = painel.drop_duplicates("referencia_periodo").sort_values("referencia_periodo")
    ultimo = hist.iloc[-1]
    prox = viv.sort_values("data_alvo").iloc[0]
    data_realizado = pd.Timestamp(ultimo["referencia_periodo"]).strftime("%m/%Y")
    data_prox = pd.Timestamp(prox["data_alvo"]).strftime("%m/%Y")

    # O que o Focus havia previsto (h=1) para este mesmo periodo, ja realizado.
    previsto = painel.loc[
        (painel["referencia_periodo"] == ultimo["referencia_periodo"]) & (painel["horizonte"] == 1),
        "mediana_focus",
    ]
    previsto_txt = f"{previsto.iloc[0]:.2f}" if len(previsto) else T["nd"]

    return f"""
    <div class="cartao">
      <div class="cartao-titulo">{ROTULOS[nome]} <span class="unidade">{UNIDADES[nome]}</span></div>
      <div class="cartao-corpo">
        {T["realizado"]} {data_realizado}: <b>{ultimo['realizado']:.2f}</b>
        <span class="cartao-prev">({T["previu"]}: {previsto_txt})</span><br>
        Focus {data_prox}: <b class="cartao-foco">{prox['mediana']:.2f}</b>
        <span class="cartao-faixa">[{prox['q_inf_90']:.2f}; {prox['q_sup_90']:.2f}]</span>
      </div>
    </div>"""


def gerar(lang: str, saida: Path) -> None:
    global T, ROTULOS, UNIDADES
    T, ROTULOS, UNIDADES = TEXTOS[lang], ROTULOS_POR_LANG[lang], UNIDADES_POR_LANG[lang]

    dados = {}
    for nome in ORDEM:
        painel = carregar_painel(nome)
        ks = calibrar_k(painel)
        fase = erro_por_fase(painel)
        if nome == "selic":
            viv, data_pesquisa = projecao_vigente_selic(painel, ks)
        else:
            viv, data_pesquisa = projecao_vigente_mensal(nome, ks)
        dados[nome] = dict(painel=painel, ks=ks, fase=fase, viv=viv, data_pesquisa=data_pesquisa)

    data_pesquisa_global = max(d["data_pesquisa"] for d in dados.values())

    blocos_por_indicador = {}
    mapa_idx_horizonte = {}
    for nome in ORDEM:
        d = dados[nome]
        fig_hist, idx_horizonte = fig_historico_projecao(nome, d["painel"], d["viv"])
        fig_fase = fig_distribuicao_fase(nome, d["fase"])
        mapa_idx_horizonte[nome] = idx_horizonte
        blocos_por_indicador[nome] = dict(
            cartao=cartao_html(nome, d["painel"], d["viv"]),
            fig_hist=pio.to_html(fig_hist, include_plotlyjs=False, full_html=False, div_id=f"fig-hist-{nome}"),
            checkboxes=checkboxes_horizonte(nome),
            fig_fase=pio.to_html(fig_fase, include_plotlyjs=False, full_html=False, div_id=f"fig-fase-{nome}"),
        )

    cartoes = "".join(blocos_por_indicador[n]["cartao"] for n in ORDEM)

    paineis_secao = []
    for i, nome in enumerate(ORDEM):
        b = blocos_por_indicador[nome]
        display = "block" if i == 0 else "none"
        paineis_secao.append(f"""
        <div class="painel-indicador" id="painel-{nome}" style="display:{display}">
          <div class="secao">
            <h2>{T["titulo_hist"]}</h2>
            <p class="linha-simples">{T["sub_hist"]}</p>
            {b['checkboxes']}
            {b['fig_hist']}
            <p class="nota-aviso">⚠️ {T["aviso"]}</p>
          </div>
          <div class="secao">
            <h2>{T["titulo_fase"]}</h2>
            <p class="linha-simples">{T["sub_fase"]}</p>
            {b['fig_fase']}
          </div>
        </div>""")

    seletor_botoes = "".join(
        f'<button class="botao-indicador{" ativo" if i == 0 else ""}" onclick="selecionarIndicador(\'{n}\')" id="botao-{n}">{ROTULOS[n]}</button>'
        for i, n in enumerate(ORDEM)
    )

    plotly_js_tag = f"<script>{pyo.get_plotlyjs()}</script>"
    idx_horizonte_json = json.dumps(mapa_idx_horizonte)

    html = f"""<!DOCTYPE html>
<html lang="{T["lang"]}">
<head>
<meta charset="utf-8">
<title>{T["titulo"]}</title>
{plotly_js_tag}
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 0; padding: 20px; background: #f7f7f9; color: #1a1a1a; }}
  .pagina {{ max-width: 1240px; margin: 0 auto; }}
  .barra-topo {{ background: #14213d; color: white; padding: 8px 18px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 15px; }}
  .barra-topo h1 {{ font-size: 16px; margin: 0; }}
  .meta-topo {{ font-size: 11px; color: #cbd3e8; }}
  .seletor {{ display: flex; gap: 6px; }}
  .botao-indicador {{ background: #2a3a63; color: #cbd3e8; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
  .botao-indicador.ativo {{ background: #fca311; color: #14213d; font-weight: 700; }}
  .cartoes {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px; }}
  .cartao {{ background: white; border-radius: 10px; padding: 8px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); max-height: {ALTURA_CARTAO}px; display: flex; flex-direction: column; justify-content: flex-start; }}
  .cartao-titulo {{ font-weight: 700; font-size: 13px; }}
  .unidade {{ font-weight: 400; color: #888; font-size: 10px; }}
  .cartao-corpo {{ font-size: 11px; color: #444; line-height: 1.5; margin-top: 4px; }}
  .cartao-foco {{ color: #1f77b4; }}
  .cartao-prev {{ color: #999; font-size: 10px; }}
  .cartao-faixa {{ color: #888; font-size: 10px; }}
  .secao {{ background: white; border-radius: 10px; padding: 10px 16px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .secao h2 {{ font-size: 14px; margin: 0 0 2px; }}
  .linha-simples {{ font-size: 11px; color: #555; margin: 0 0 4px; }}
  .seletor-horizonte {{ display: flex; gap: 14px; align-items: center; font-size: 11px; margin-bottom: 2px; color: #444; flex-wrap: wrap; }}
  .chk-horizonte {{ display: inline-flex; gap: 4px; align-items: center; cursor: pointer; }}
  .nota-aviso {{ font-size: 10px; color: #8a6d00; background: #fff8e1; padding: 4px 10px; border-radius: 6px; margin-top: 2px; }}
  .notas-detalhes {{ margin-top: 2px; }}
  .notas-detalhes summary {{ font-size: 11px; color: #555; cursor: pointer; padding: 4px 2px; }}
  .notas {{ font-size: 11px; color: #777; padding: 6px 2px 0; line-height: 1.6; }}
  .notas b {{ color: #444; }}
</style>
</head>
<body>
<div class="pagina">
<div class="barra-topo">
  <h1>{T["titulo"]}</h1>
  <div class="meta-topo">{T['pesquisa']}: {data_pesquisa_global.strftime(T['fmt_data'])} · {T['gerado']}: {pd.Timestamp.today().strftime(T['fmt_data'])}</div>
  <div class="seletor">{seletor_botoes}</div>
</div>
<div class="cartoes">{cartoes}</div>
{''.join(paineis_secao)}
<details class="notas-detalhes">
<summary>▸ {T["notas"]}</summary>
<div class="notas">
{T["texto_notas"]}
</div>
</details>
</div>
<script>
var idxHorizonte = {idx_horizonte_json};
function alternarHorizonte(nome, h, marcado) {{
  var idx = idxHorizonte[nome][h];
  Plotly.restyle(document.getElementById('fig-hist-' + nome), {{visible: marcado}}, [idx]);
}}
function selecionarIndicador(nome) {{
  document.querySelectorAll('.painel-indicador').forEach(function(el) {{ el.style.display = 'none'; }});
  document.getElementById('painel-' + nome).style.display = 'block';
  document.querySelectorAll('.botao-indicador').forEach(function(el) {{ el.classList.remove('ativo'); }});
  document.getElementById('botao-' + nome).classList.add('ativo');
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(html, encoding="utf-8")


ROTULOS_POR_LANG = {
    "pt": {"ipca": "IPCA", "cambio": "Câmbio", "igpm": "IGP-M", "selic": "Selic"},
    "en": {"ipca": "IPCA", "cambio": "FX Rate", "igpm": "IGP-M", "selic": "Selic"},
}
UNIDADES_POR_LANG = {
    "pt": {"ipca": "% a.m.", "cambio": "R$/US$", "igpm": "% a.m.", "selic": "% a.a."},
    "en": {"ipca": "% m/m", "cambio": "BRL/USD", "igpm": "% m/m", "selic": "% p.a."},
}
TEXTOS = {
    "pt": {
        'lang': 'pt-BR',
        'realizado': 'Realizado',
        'faixa90': 'faixa 90%',
        'extra90': 'extrapolado (h>12)',
        'faixa70': 'faixa 70%',
        'extra70': 'faixa 70% (extrapolado)',
        'focus_vigente': 'Focus (vigente)',
        'hoje': 'hoje',
        'titulo_hist': 'Histórico completo e projeção vigente',
        'amostra_insuf': 'amostra insuf.',
        'sig': '✱ sig.',
        'aperto': 'aperto',
        'afrouxamento': 'afrouxamento',
        'horizontes': 'Horizontes do Focus:',
        'nd': 'n/d',
        'previu': 'Focus h=1 previu',
        'sub_hist': 'Todo o histórico realizado, a projeção do Focus por horizonte (marque/desmarque abaixo) e a faixa recalibrada vigente.',
        'aviso': 'a faixa é um piso de incerteza, não uma garantia — em viradas de ciclo ela cobre ~60–80%, não 90%.',
        'titulo_fase': 'Distribuição do erro: aperto × afrouxamento',
        'sub_fase': 'O Focus erra de forma diferente quando a Selic está subindo (aperto) ou caindo (afrouxamento)? Só considerar diferenças marcadas como ✱ significativas.',
        'titulo': 'Focus — Acompanhamento',
        'pesquisa': 'Pesquisa',
        'gerado': 'Gerado',
        'fmt_data': '%d/%m/%Y',
        'notas': 'Notas e metodologia',
        "texto_notas": """
  <b>Nota.</b> Dados reais, `k` calculado in-sample sem persistência.
  Câmbio, IGP-M e IPCA: horizonte em meses. Selic: horizonte em reuniões do Copom (aproximado aqui
  — sem o calendário oficial de reuniões futuras). Desocupação fora do escopo (amostra insuficiente).
  <br><br>
  <b>Metodologia.</b> O erro é definido como <i>realizado − mediana Focus</i>, usando um painel
  <i>point-in-time</i> (só entra no cálculo o que já era publicamente conhecido na data de corte — sem
  vazamento de informação futura). Para corrigir o <b>ponto</b> da projeção (a média condicional do
  erro), foram testados <b>ElasticNet</b> e <b>XGBoost</b>, validados por validação cruzada purgada por
  data e pelo teste de <b>Diebold-Mariano</b> (com a correção de amostra pequena de
  Harvey-Leybourne-Newbold) contra o próprio Focus: nenhum superou o Focus com robustez estatística
  suficiente para ser adotado (ganho significativo só no IPCA em h=1; piora significativa na Selic em
  h=1,3,6). Para a <b>incerteza</b> — o que este painel mostra — a escada de modelos testada foi:
  <b>B0</b> (dispersão publicada pelo próprio Focus), <b>B1</b> (quantis históricos incondicionais do
  erro), <b>B2</b> (regressão quantílica condicional via XGBoost multi-quantil) e <b>B3</b>
  (Conformalized Quantile Regression sobre B2, com garantia de cobertura em amostra finita). Nenhuma
  variante acima de B0 justificou, em CRPS e cobertura, a complexidade adicional. O modelo
  <b>selecionado para este painel é B0′</b>: a dispersão publicada pelo Focus, reescalada por um fator
  <code>k</code> estimado por indicador × horizonte (minimizando a perda pinball média sobre uma grade
  de quantis — <code>ajustar_fator_sigma</code>), calibrado sobre o histórico disponível. A diferença de
  erro entre fases do ciclo (aperto × afrouxamento) é testada por regressão OLS com erros-padrão
  <b>HAC</b> (Newey-West, <code>maxlags = horizonte</code>), para corrigir a autocorrelação de meses
  contíguos dentro de cada fase. Fonte: BCB Focus (OData) e SGS.
""",
    },
    "en": {
        'lang': 'en',
        'realizado': 'Actual',
        'faixa90': '90% range',
        'extra90': 'extrapolated (h>12)',
        'faixa70': '70% range',
        'extra70': '70% range (extrapolated)',
        'focus_vigente': 'Focus (current)',
        'hoje': 'today',
        'titulo_hist': 'Full history and current projection',
        'amostra_insuf': 'insuff. sample',
        'sig': '✱ sig.',
        'aperto': 'tightening',
        'afrouxamento': 'easing',
        'horizontes': 'Focus horizons:',
        'nd': 'n/a',
        'previu': 'Focus h=1 forecasted',
        'sub_hist': "The complete actual history, Focus's projection by horizon (check/uncheck below), and the current recalibrated range.",
        'aviso': 'the range is a floor on uncertainty, not a guarantee — during cycle turning points it covers ~60–80%, not 90%.',
        'titulo_fase': 'Error distribution: tightening × easing',
        'sub_fase': 'Does Focus err differently when the Selic rate is rising (tightening) or falling (easing)? Only consider differences marked ✱ as significant.',
        'titulo': 'Focus — Tracking Dashboard',
        'pesquisa': 'Survey',
        'gerado': 'Generated',
        'fmt_data': '%m/%d/%Y',
        'notas': 'Notes and methodology',
        "texto_notas": """
  <b>Note.</b> Real data, `k` computed in-sample without persistence.
  FX, IGP-M and IPCA: horizon in months. Selic: horizon in Copom (monetary policy committee) meetings
  (approximated here — without the official calendar of future meetings). Unemployment rate
  excluded from scope (insufficient sample).
  <br><br>
  <b>Methodology.</b> The error is defined as <i>actual − Focus median</i>, using a
  <i>point-in-time</i> panel (only data that was already publicly known at the cutoff date enters the
  calculation — no leakage of future information). To correct the <b>point</b> forecast (the conditional
  mean of the error), <b>ElasticNet</b> and <b>XGBoost</b> were tested, validated via date-purged cross-validation
  and the <b>Diebold-Mariano</b> test (with the Harvey-Leybourne-Newbold small-sample correction) against
  Focus itself: neither beat Focus with enough statistical robustness to be adopted (significant gain only
  for IPCA at h=1; significant deterioration for Selic at h=1,3,6). For the <b>uncertainty</b> — what this
  dashboard shows — the model ladder tested was: <b>B0</b> (dispersion as published by Focus itself),
  <b>B1</b> (unconditional historical quantiles of the error), <b>B2</b> (conditional quantile regression via
  multi-quantile XGBoost), and <b>B3</b> (Conformalized Quantile Regression on top of B2, with finite-sample
  coverage guarantee). No variant above B0 justified, in CRPS and coverage terms, the added complexity. The
  model <b>selected for this dashboard is B0′</b>: Focus's published dispersion, rescaled by a factor
  <code>k</code> estimated per indicator × horizon (minimizing the average pinball loss over a quantile
  grid — <code>ajustar_fator_sigma</code>), calibrated over the available history. The difference in error
  between cycle phases (tightening × easing) is tested via OLS regression with <b>HAC</b> standard errors
  (Newey-West, <code>maxlags = horizon</code>), to correct for the autocorrelation of contiguous months
  within each phase. Source: BCB Focus (OData) and SGS.
""",
    },
}
T: dict = {}
ROTULOS: dict = {}
UNIDADES: dict = {}


def gerar_site(destino: Path) -> None:
    """Gera `destino/index.html` (PT) e `destino/en/index.html` (EN)."""
    gerar("pt", destino / "index.html")
    gerar("en", destino / "en" / "index.html")
