"""Valores unicos de `fault` e a assinatura de vibracao de cada um.

A secao 2 compara ate 4 rotulos. A ordem dos blocos e proposital: primeiro o
comportamento ao longo do tempo (todas as colunas empilhadas), depois o resumo
em medianas. Ver a serie antes da estatistica evita ler uma mediana sem saber
se ela descreve um patamar estavel ou a media de dois regimes distintos.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Analise de Falhas", "📊")

# Acima disso as colunas ficam estreitas demais para os graficos serem lidos.
MAX_ROTULOS = 4

st.title("📊 Analise de Falhas")
st.caption("Valores unicos da coluna `fault` e o comportamento medido de cada um.")

try:
    rotulos = D.r_rotulos()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

familias = D.r_familias()
# `perfil_rotulos` traz contagem e janela; `sugerir_familias` traz o agrupamento
# proposto. Juntar aqui evita repetir groupby em cada widget.
# So trazemos `familia_sugerida` do segundo: `e_problema` existe nos dois (e a
# mesma funcao por tras) e o merge criaria `e_problema_x` / `e_problema_y`.
tabela = rotulos.merge(familias[["fault", "familia_sugerida"]], on="fault", how="left")

# ==========================================================================
# 1. Panorama dos valores unicos
# ==========================================================================
st.header("1. Valores unicos de `fault`")

n_total = len(tabela)
n_problema = int(tabela["e_problema"].sum())
n_familias = tabela["familia_sugerida"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rotulos distintos", n_total)
c2.metric("Defeitos", n_problema)
c3.metric("Estados (nao defeito)", n_total - n_problema)
c4.metric("Familias sugeridas", n_familias)

st.markdown(
    f"""
Os {n_total} rotulos nao sao {n_total} defeitos. A coluna `familia_sugerida` agrupa
por radical — e **sugestao heuristica**, para tornar a lista navegavel. A decisao
final vive no `data/fault_map.yaml`, curado a mao na Parte 1.
"""
)

col_f, col_b = st.columns([2, 3])
with col_f:
    filtro_tipo = st.radio("Mostrar", ["Todos", "So defeitos", "So estados"], horizontal=True)
with col_b:
    busca = st.text_input("Filtrar por texto", placeholder="ex.: rolamento, cocked, normal")

vis = tabela
if filtro_tipo == "So defeitos":
    vis = vis[vis["e_problema"]]
elif filtro_tipo == "So estados":
    vis = vis[~vis["e_problema"]]
if busca:
    vis = vis[vis["fault"].str.contains(busca, case=False, na=False)]

st.caption(f"{len(vis)} de {n_total} rotulos.")
st.dataframe(
    vis[
        [
            "fault",
            "familia_sugerida",
            "e_problema",
            "n_leituras",
            "pct",
            "primeira",
            "ultima",
            "span_horas",
        ]
    ],
    hide_index=True,
    height=340,
    column_config={
        "fault": "rotulo",
        "familia_sugerida": "familia (sugerida)",
        "e_problema": st.column_config.CheckboxColumn("defeito?"),
        "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
        "pct": st.column_config.NumberColumn("% do total", format="%.2f%%"),
        "primeira": st.column_config.DatetimeColumn("1a leitura", format="DD/MM/YY HH:mm"),
        "ultima": st.column_config.DatetimeColumn("ultima leitura", format="DD/MM/YY HH:mm"),
        "span_horas": st.column_config.NumberColumn("span (h)", format="%.1f"),
    },
)

with st.expander("Leituras por familia sugerida"):
    por_fam = (
        tabela.groupby("familia_sugerida", as_index=False)
        .agg(rotulos=("fault", "count"), leituras=("n_leituras", "sum"))
        .sort_values("leituras", ascending=False)
    )
    st.altair_chart(
        alt.Chart(por_fam)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("leituras:Q"),
            y=alt.Y("familia_sugerida:N", sort="-x", title=None),
            tooltip=["familia_sugerida", "rotulos", "leituras"],
        )
        .properties(height=28 * len(por_fam)),
        width="stretch",
    )

st.divider()

# ==========================================================================
# 2. Caracteristicas dos rotulos selecionados
# ==========================================================================
st.header("2. Caracteristicas de um ou mais rotulos")

opcoes = tabela.sort_values("n_leituras", ascending=False)["fault"].tolist()
_leituras = dict(zip(tabela["fault"], tabela["n_leituras"]))

selecionados = st.multiselect(
    "Rotulos",
    opcoes,
    default=opcoes[:1],
    max_selections=MAX_ROTULOS,
    format_func=lambda r: f"{r}  ({_leituras[r]} leituras)",
    help=f"Ate {MAX_ROTULOS} rotulos. As series de todos aparecem no mesmo grafico, "
         "uma cor por rotulo.",
)

if not selecionados:
    st.info("Selecione ao menos um rotulo para ver a analise.")
    st.stop()

n_sel = len(selecionados)
# Cores fixas por posicao: a mesma cor identifica o rotulo em todos os blocos.
PALETA = ["#d1495b", "#4c78a8", "#2d6a4f", "#e2a03f"]
cor_de = {r: PALETA[i % len(PALETA)] for i, r in enumerate(selecionados)}

info_de = {r: tabela[tabela["fault"] == r].iloc[0] for r in selecionados}
ordenado_de = {r: D.r_ordenado(r) for r in selecionados}

numericas = D.r_numericas()
padrao = "z_rms_velocity_mm_s" if "z_rms_velocity_mm_s" in numericas else numericas[0]

# --- 2a. Cabecalho comparativo ---------------------------------------------
st.subheader("Panorama")

for coluna, rotulo in zip(st.columns(n_sel), selecionados):
    info = info_de[rotulo]
    ordenado = ordenado_de[rotulo]
    with coluna:
        st.markdown(
            f"<div style='border-left:5px solid {cor_de[rotulo]};padding-left:10px'>"
            f"<b>{rotulo}</b></div>",
            unsafe_allow_html=True,
        )
        st.metric("Leituras", f"{int(info['n_leituras']):,}".replace(",", "."))
        st.metric("% do dataset", f"{info['pct']:.2f}%")
        st.metric("Familia sugerida", info["familia_sugerida"])
        st.metric("Classificacao", "Defeito" if info["e_problema"] else "Estado")
        st.metric("Sessoes de coleta", int(ordenado["sessao"].nunique()))
        st.metric("Span coberto", f"{info['span_horas']:.1f} h")

estados = [r for r in selecionados if not info_de[r]["e_problema"]]
if estados:
    st.info(
        f"**{', '.join(estados)}** — classificado(s) como **estado**, nao defeito. "
        "No pipeline final o guardrail G2 encerra o fluxo prescritivo aqui: nao ha "
        "acao corretiva a sugerir para uma maquina normal, em teste ou desligada."
    )

st.divider()

# --- 2b. Serie temporal ----------------------------------------------------
st.subheader("Serie temporal")

st.caption(
    "Eixo X = **data e hora reais da coleta**, em UTC, como gravadas em `created_at`. "
    "Nada e normalizado nem deslocado. O arquivo bruto nao esta em ordem cronologica — "
    "a ordenacao e feita aqui. Intervalos maiores que "
    f"{int(D.config.GAP_NOVA_SESSAO_S)} s marcam fronteira de sessao, e a linha "
    "**quebra** nessas fronteiras: ligar o fim de uma sessao ao inicio da seguinte "
    "inventaria uma transicao que nunca existiu."
)

# Janela real de cada rotulo. Precisa ficar visivel porque os rotulos foram
# coletados em campanhas diferentes: no eixo de tempo real eles aparecem lado a
# lado, nao sobrepostos, e isso confunde quem espera curvas concorrentes.
janelas = pd.DataFrame(
    [
        {
            "rotulo": r,
            "inicio": ordenado_de[r]["created_at"].iloc[0],
            "fim": ordenado_de[r]["created_at"].iloc[-1],
            "sessoes": int(ordenado_de[r]["sessao"].nunique()),
            "leituras": len(ordenado_de[r]),
        }
        for r in selecionados
    ]
)

st.dataframe(
    janelas,
    hide_index=True,
    column_config={
        "inicio": st.column_config.DatetimeColumn("1a leitura (UTC)",
                                                  format="DD/MM/YYYY HH:mm:ss"),
        "fim": st.column_config.DatetimeColumn("ultima leitura (UTC)",
                                               format="DD/MM/YYYY HH:mm:ss"),
        "sessoes": st.column_config.NumberColumn("sessoes", format="%d"),
        "leituras": st.column_config.NumberColumn("leituras", format="%d"),
    },
)

if n_sel > 1:
    # Ha sobreposicao temporal entre algum par de rotulos?
    sobrepoe = any(
        (janelas.loc[i, "inicio"] <= janelas.loc[j, "fim"])
        and (janelas.loc[j, "inicio"] <= janelas.loc[i, "fim"])
        for i in range(n_sel)
        for j in range(i + 1, n_sel)
    )
    if not sobrepoe:
        st.warning(
            "**Os rotulos selecionados nao compartilham janela de coleta.** No eixo de "
            "tempo real eles aparecem em trechos separados, nao como curvas concorrentes. "
            "Isso e o dado, nao um defeito do grafico: cada condicao foi gravada numa "
            "campanha propria. Use o zoom (roda do mouse) para entrar em cada trecho."
        )
    else:
        st.info(
            "Os rotulos tem janelas de coleta que se cruzam — as curvas vao aparecer "
            "sobrepostas nos trechos em comum."
        )

col_todas, col_faixa = st.columns([1, 1])
with col_todas:
    todas_colunas = st.checkbox(
        "Todas as colunas numericas",
        value=True,
        help="Desmarque para escolher um subconjunto e ganhar resolucao — o orcamento "
             "de pontos e dividido entre os graficos.",
    )
with col_faixa:
    mostrar_faixa = st.checkbox(
        "Mostrar faixa min-max",
        value=False,
        help="A faixa preserva os picos que a reamostragem esconderia. Com muitas "
             "series ela polui a leitura, por isso vem desligada.",
    )

if todas_colunas:
    colunas_serie = numericas
else:
    colunas_serie = st.multiselect("Colunas para plotar", numericas, default=[padrao])

if not colunas_serie:
    st.caption("Selecione ao menos uma coluna.")
else:
    n_col = len(colunas_serie)

    # Orcamento global: sao ate n_sel x n_col series simultaneas na pagina, e
    # cada ponto vira um objeto JSON no navegador. Dividimos o teto da pagina
    # entre elas e respeitamos o piso, para a linha nao perder a forma.
    orcamento = int(
        min(
            D.config.MAX_PONTOS_SERIE,
            max(D.config.MIN_PONTOS_SERIE, D.config.MAX_PONTOS_PAGINA // (n_sel * n_col)),
        )
    )
    series = {
        r: D.r_serie_temporal(r, tuple(colunas_serie), orcamento) for r in selecionados
    }

    reamostrados = [r for r in selecionados if series[r]["reamostrado"]]
    if reamostrados:
        detalhe = ", ".join(
            f"`{r}` ({series[r]['n_original']:,} → {series[r]['n_pontos']:,} pontos, "
            f"blocos de {series[r]['fator']})".replace(",", ".")
            for r in reamostrados
        )
        st.caption(
            f"**Reamostrado** — {n_col} coluna(s) x {n_sel} rotulo(s) = "
            f"{n_col * n_sel} series, com orcamento de {orcamento} pontos cada: "
            f"{detalhe}. A linha e a mediana do bloco, e cada ponto carrega o "
            "`created_at` real da primeira leitura do bloco — nenhum instante e "
            "inventado. Para mais resolucao, desmarque *Todas as colunas*."
        )

    escala_cor = alt.Scale(domain=selecionados, range=[cor_de[r] for r in selecionados])
    # Com 23 graficos empilhados, 260px cada dariam 6 mil pixels de rolagem.
    altura = 260 if n_col <= 4 else 170

    for coluna_dado in colunas_serie:
        # Um DataFrame com todos os rotulos: e o que permite sobrepor as series
        # num grafico so, cada uma com sua cor.
        partes = []
        for r in selecionados:
            d = series[r]["dados"]
            d = d[d["coluna"] == coluna_dado]
            if not d.empty:
                partes.append(d.assign(rotulo=r))
        if not partes:
            continue
        junto = pd.concat(partes, ignore_index=True)

        st.markdown(f"**{coluna_dado}**")

        # O Vega agrupa as linhas pela combinacao dos canais discretos: `color`
        # (rotulo) e `detail` (sessao). Isso da uma linha por par
        # rotulo-sessao — cores distintas entre rotulos, e quebra nos gaps.
        base_ch = alt.Chart(junto).encode(
            x=alt.X("created_at:T", title="data / hora da coleta (UTC)"),
            color=alt.Color("rotulo:N", title=None, scale=escala_cor,
                            legend=alt.Legend(orient="bottom")),
            detail=alt.Detail("sessao:N"),
        )

        linha = base_ch.mark_line(strokeWidth=1.4).encode(
            y=alt.Y("valor:Q", title=coluna_dado, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("rotulo:N", title="rotulo"),
                alt.Tooltip("created_at:T", title="instante (UTC)",
                            format="%d/%m/%Y %H:%M:%S"),
                alt.Tooltip("sessao:Q", title="sessao"),
                alt.Tooltip("valor:Q", title="mediana", format=".4f"),
                alt.Tooltip("minimo:Q", title="min", format=".4f"),
                alt.Tooltip("maximo:Q", title="max", format=".4f"),
            ],
        )

        if mostrar_faixa and reamostrados:
            faixa = base_ch.mark_area(opacity=0.18).encode(
                y=alt.Y("minimo:Q", title=coluna_dado, scale=alt.Scale(zero=False)),
                y2="maximo:Q",
            )
            grafico = faixa + linha
        else:
            grafico = linha

        # `.interactive()` liga zoom e pan no eixo do tempo — necessario quando
        # os rotulos estao a semanas de distancia e cada sessao vira um risco.
        st.altair_chart(grafico.properties(height=altura).interactive(), width="stretch")

st.divider()

# --- 2c. Assinatura comparada ----------------------------------------------
st.subheader("Assinatura: o que distingue cada rotulo")

comparacoes = {r: D.r_comparacao(r) for r in selecionados}

# Tabela unica: uma linha por feature, uma coluna por rotulo. Comparar valores
# na horizontal e mais direto que alternar entre tabelas separadas.
primeiro = comparacoes[selecionados[0]]
comparada = primeiro[["feature", "mediana_global"]].copy()
for r in selecionados:
    c = comparacoes[r].set_index("feature")
    comparada[f"{r}"] = comparada["feature"].map(c["mediana_rotulo"])
    comparada[f"Δ% {r}"] = comparada["feature"].map(c["desvio_pct"])

# Ordem compartilhada por todos os blocos: features que mais separam ALGUM dos
# rotulos selecionados vem primeiro. Sem ordem comum, os graficos lado a lado
# ficariam com linhas trocadas e a comparacao visual nao funcionaria.
colunas_desvio = [f"Δ% {r}" for r in selecionados]
comparada["_max_abs"] = comparada[colunas_desvio].abs().max(axis=1)
comparada = comparada.sort_values("_max_abs", ascending=False).drop(columns="_max_abs")
ordem_features = comparada["feature"].tolist()

st.caption(
    "Mediana de cada rotulo contra a mediana do dataset inteiro. `Δ%` e o quanto a "
    "feature se afasta do comportamento geral. Ordenado pelo maior desvio entre os "
    "rotulos escolhidos — a mesma ordem vale para os graficos abaixo, para que as "
    "linhas correspondam entre as colunas."
)

st.dataframe(
    comparada,
    hide_index=True,
    height=min(560, 40 + 35 * len(comparada)),
    column_config={
        "feature": st.column_config.TextColumn("feature", width="medium"),
        "mediana_global": st.column_config.NumberColumn("mediana global", format="%.4f"),
        **{r: st.column_config.NumberColumn(r, format="%.4f") for r in selecionados},
        **{
            f"Δ% {r}": st.column_config.NumberColumn(f"Δ% {r}", format="%.1f%%")
            for r in selecionados
        },
    },
)

for coluna, rotulo in zip(st.columns(n_sel), selecionados):
    with coluna:
        st.caption(f"**{rotulo}** — desvio da mediana global")
        st.altair_chart(
            alt.Chart(comparacoes[rotulo])
            .mark_bar(color=cor_de[rotulo])
            .encode(
                x=alt.X("desvio_pct:Q", title="Δ% vs global"),
                y=alt.Y("feature:N", sort=ordem_features, title=None),
                tooltip=["feature", "mediana_rotulo", "mediana_global", "desvio_pct"],
            )
            .properties(height=26 * len(ordem_features)),
            width="stretch",
        )

with st.expander("Estatistica detalhada por rotulo (quartis, desvio, CV)"):
    st.caption(
        "`cv` e o coeficiente de variacao: quanto maior, mais dispersa a classe e "
        "mais o kNN vai confundi-la na Parte 3."
    )
    for coluna, rotulo in zip(st.columns(n_sel), selecionados):
        with coluna:
            st.caption(f"**{rotulo}**")
            st.dataframe(
                D.r_assinatura(rotulo),
                hide_index=True,
                height=400,
                column_config={
                    "cv": st.column_config.NumberColumn("cv", format="%.3f"),
                },
            )

st.divider()

# --- 2d. Distribuicao de uma feature ---------------------------------------
st.subheader("Distribuicao de uma feature")

feature = st.selectbox("Feature", numericas, index=numericas.index(padrao))

serie_global = D.dados()[feature].to_numpy()
series_rotulo = {r: D.r_serie(r, feature) for r in selecionados}

# Bordas COMPARTILHADAS entre todos os rotulos. Com bins proprios por rotulo os
# histogramas ficariam com larguras diferentes e a comparacao visual seria falsa.
lo = float(min(min(s.min() for s in series_rotulo.values()),
               np.quantile(serie_global, 0.001)))
hi = float(max(max(s.max() for s in series_rotulo.values()),
               np.quantile(serie_global, 0.999)))
if hi <= lo:
    hi = lo + 1e-6
bordas = np.linspace(lo, hi, 61)
centros = (bordas[:-1] + bordas[1:]) / 2
h_glo, _ = np.histogram(serie_global, bins=bordas)
dens_glo = h_glo / max(h_glo.sum(), 1)

st.caption(
    "Densidade, nao contagem — cada rotulo tem um numero diferente de leituras e em "
    "contagem bruta o menor sumiria. Os intervalos do histograma sao os mesmos nos "
    "graficos, entao as formas sao comparaveis."
)

for coluna, rotulo in zip(st.columns(n_sel), selecionados):
    h_rot, _ = np.histogram(series_rotulo[rotulo], bins=bordas)
    hist = pd.DataFrame(
        {
            "valor": np.concatenate([centros, centros]),
            "densidade": np.concatenate([h_rot / max(h_rot.sum(), 1), dens_glo]),
            "serie": [rotulo] * len(centros) + ["dataset inteiro"] * len(centros),
        }
    )
    with coluna:
        st.caption(f"**{rotulo}**")
        st.altair_chart(
            alt.Chart(hist)
            .mark_area(opacity=0.55, interpolate="step")
            .encode(
                x=alt.X("valor:Q", title=feature),
                y=alt.Y("densidade:Q", title="densidade", stack=None),
                color=alt.Color(
                    "serie:N",
                    title=None,
                    scale=alt.Scale(
                        domain=[rotulo, "dataset inteiro"],
                        range=[cor_de[rotulo], "#b0b0b0"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    "serie",
                    alt.Tooltip("valor:Q", format=".4f"),
                    alt.Tooltip("densidade:Q", format=".4f"),
                ],
            )
            .properties(height=260),
            width="stretch",
        )

st.divider()

# --- 2e. Outliers ----------------------------------------------------------
with st.expander("Outliers dentro de cada rotulo (criterio IQR, apenas reportado)"):
    st.caption(
        "Limites recalculados **dentro** de cada rotulo. Isso separa a variacao "
        "natural da classe de um pico que destoa da propria classe — globalmente, "
        "toda leitura de um defeito severo pareceria outlier, o que nao ajuda."
    )
    for coluna, rotulo in zip(st.columns(n_sel), selecionados):
        with coluna:
            st.caption(f"**{rotulo}**")
            st.dataframe(
                D.r_outliers_do_rotulo(rotulo)[
                    ["coluna", "mediana", "lim_inferior", "lim_superior",
                     "min", "max", "outliers", "pct_outliers"]
                ],
                hide_index=True,
                height=320,
            )

# --- 2f. Base ordenada -----------------------------------------------------
with st.expander("Base de cada rotulo em ordem cronologica"):
    st.caption(
        "Em abas, e nao lado a lado: sao 26 colunas por rotulo, e dividir a largura "
        "deixaria a tabela ilegivel. `sessao` identifica a campanha de coleta; "
        "`delta_s` e o intervalo desde a leitura anterior dentro da sessao."
    )
    for aba, rotulo in zip(st.tabs(selecionados), selecionados):
        with aba:
            ordenado = ordenado_de[rotulo]
            colunas_tabela = (
                ["created_at", "sessao", "delta_s", "id", "fault"]
                + [c for c in numericas if c in ordenado.columns]
            )
            st.dataframe(
                ordenado[colunas_tabela],
                hide_index=True,
                height=400,
                column_config={
                    "created_at": st.column_config.DatetimeColumn(
                        "created_at", format="DD/MM/YY HH:mm:ss.SSS"
                    ),
                    "delta_s": st.column_config.NumberColumn("delta (s)", format="%.2f"),
                },
            )
            st.download_button(
                "Baixar CSV ordenado",
                ordenado[colunas_tabela].to_csv(index=False).encode("utf-8"),
                file_name=f"{rotulo}_cronologico.csv",
                mime="text/csv",
                key=f"dl_{rotulo}",
            )

st.divider()

# ==========================================================================
# 3. Tabela de assinaturas — entrega da Parte 0
# ==========================================================================
st.header("3. Tabela de assinaturas por rotulo")
st.caption(
    "Entrega da Parte 0: e esta tabela que sera cruzada com o que os PDFs de "
    "procedimento descrevem. Divergencia entre o medido e o documentado e achado, "
    "nao erro."
)

minimo = st.slider(
    "Ignorar rotulos com menos de N leituras",
    0,
    500,
    100,
    step=50,
    help="Varios rotulos tem 2 leituras; uma mediana sobre isso nao significa nada.",
)
st.dataframe(D.r_assinaturas(minimo), hide_index=True, height=400)
