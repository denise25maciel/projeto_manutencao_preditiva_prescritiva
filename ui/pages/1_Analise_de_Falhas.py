"""Valores unicos de `fault` e a assinatura de vibracao de cada um."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Analise de Falhas", "📊")

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
# 2. Caracteristicas do rotulo selecionado
# ==========================================================================
st.header("2. Caracteristicas de um rotulo")

opcoes = tabela.sort_values("n_leituras", ascending=False)["fault"].tolist()
_leituras = dict(zip(tabela["fault"], tabela["n_leituras"]))
escolhido = st.selectbox("Rotulo", opcoes, format_func=lambda r: f"{r}  ({_leituras[r]} leituras)")

info = tabela[tabela["fault"] == escolhido].iloc[0]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Leituras", f"{int(info['n_leituras']):,}".replace(",", "."))
m2.metric("% do dataset", f"{info['pct']:.2f}%")
m3.metric("Familia sugerida", info["familia_sugerida"])
m4.metric("Classificacao", "Defeito" if info["e_problema"] else "Estado")
m5.metric("Span coberto", f"{info['span_horas']:.1f} h")

if not info["e_problema"]:
    st.info(
        "Rotulo classificado como **estado**, nao defeito. No pipeline final o "
        "guardrail G2 encerra o fluxo prescritivo aqui: nao ha acao corretiva a "
        "sugerir para uma maquina normal, em teste ou desligada."
    )

# --- 2a. Assinatura vs. resto do dataset ----------------------------------
st.subheader("O que distingue este rotulo")

comp = D.r_comparacao(escolhido)
st.caption(
    "Mediana do rotulo contra a mediana do dataset inteiro. Ordenado pelo desvio "
    "absoluto — as primeiras linhas sao as features que mais caracterizam o rotulo."
)

st.altair_chart(
    alt.Chart(comp.head(12))
    .mark_bar()
    .encode(
        x=alt.X("desvio_pct:Q", title="desvio da mediana global (%)"),
        y=alt.Y("feature:N", sort=alt.EncodingSortField("desvio_pct", op="min"), title=None),
        color=alt.condition(alt.datum.desvio_pct > 0, alt.value("#d1495b"), alt.value("#4c78a8")),
        tooltip=["feature", "mediana_rotulo", "mediana_global", "desvio_pct"],
    )
    .properties(height=330),
    width="stretch",
)

# --- 2b. Tabela de assinatura ---------------------------------------------
st.subheader("Assinatura detalhada")
st.caption(
    "Mediana como valor central (kurtosis e crest factor sao definidos sobre picos — "
    "um impacto isolado desloca a media, nao a mediana). `cv` e o coeficiente de "
    "variacao: quanto maior, mais dispersa a classe e mais o kNN vai confundi-la."
)

st.dataframe(
    D.r_assinatura(escolhido),
    hide_index=True,
    height=420,
    column_config={
        "cv": st.column_config.NumberColumn(
            "cv", format="%.3f", help="desvio / |media| — dispersao relativa"
        ),
    },
)

# --- 2c. Distribuicao de uma feature --------------------------------------
st.subheader("Distribuicao de uma feature")

numericas = D.r_numericas()
padrao = "z_rms_velocity_mm_s" if "z_rms_velocity_mm_s" in numericas else numericas[0]
feature = st.selectbox("Feature", numericas, index=numericas.index(padrao))

serie_rotulo = D.r_serie(escolhido, feature)
serie_global = D.dados()[feature].to_numpy()

# Histograma calculado no numpy e desenhado no Altair. Passar 166 mil pontos
# crus para o navegador travaria a pagina.
# O limite superior usa o percentil 99,9 do global para a cauda longa nao
# achatar todo o grafico num unico bin.
lo = float(min(serie_rotulo.min(), np.quantile(serie_global, 0.001)))
hi = float(max(serie_rotulo.max(), np.quantile(serie_global, 0.999)))
if hi <= lo:
    hi = lo + 1e-6
bordas = np.linspace(lo, hi, 61)
centros = (bordas[:-1] + bordas[1:]) / 2

h_rot, _ = np.histogram(serie_rotulo, bins=bordas)
h_glo, _ = np.histogram(serie_global, bins=bordas)

hist = pd.DataFrame(
    {
        "valor": np.concatenate([centros, centros]),
        # Densidade, nao contagem: o rotulo tem milhares de linhas e o dataset
        # tem 166 mil. Em contagem bruta a barra do rotulo sumiria.
        "densidade": np.concatenate(
            [h_rot / max(h_rot.sum(), 1), h_glo / max(h_glo.sum(), 1)]
        ),
        "serie": [escolhido] * len(centros) + ["dataset inteiro"] * len(centros),
    }
)

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
                domain=[escolhido, "dataset inteiro"], range=["#d1495b", "#b0b0b0"]
            ),
        ),
        tooltip=[
            "serie",
            alt.Tooltip("valor:Q", format=".4f"),
            alt.Tooltip("densidade:Q", format=".4f"),
        ],
    )
    .properties(height=300),
    width="stretch",
)

# --- 2d. Serie temporal ----------------------------------------------------
st.subheader("Serie temporal")

ordenado = D.r_ordenado(escolhido)
n_sessoes = int(ordenado["sessao"].nunique())

s1, s2, s3, s4 = st.columns(4)
s1.metric("Leituras", f"{len(ordenado):,}".replace(",", "."))
s2.metric("Sessoes de coleta", n_sessoes)
s3.metric("Inicio", f"{ordenado['created_at'].iloc[0]:%d/%m %H:%M}")
s4.metric("Fim", f"{ordenado['created_at'].iloc[-1]:%d/%m %H:%M}")

st.caption(
    "As leituras deste rotulo, ordenadas por `created_at`. O arquivo bruto **nao** "
    "esta em ordem cronologica — a ordenacao e feita aqui. Intervalos maiores que "
    f"{int(D.config.GAP_NOVA_SESSAO_S)} s marcam fronteira de sessao, e a linha "
    "**quebra** nessas fronteiras: ligar o fim de uma sessao ao inicio da seguinte "
    "inventaria uma transicao que nunca existiu."
)

if n_sessoes > 1:
    resumo_sessoes = (
        ordenado.groupby("sessao")
        .agg(leituras=("created_at", "size"),
             inicio=("created_at", "min"),
             fim=("created_at", "max"))
        .reset_index()
    )
    resumo_sessoes["duracao_min"] = (
        (resumo_sessoes["fim"] - resumo_sessoes["inicio"]).dt.total_seconds() / 60
    ).round(1)
    st.info(
        f"Este rotulo aparece em **{n_sessoes} sessoes** de coleta distintas, "
        f"cobrindo {info['span_horas']:.1f} h de ponta a ponta. Contar suas linhas "
        "como ocorrencias somaria sessoes independentes."
    )
    with st.expander("Sessoes de coleta"):
        st.dataframe(
            resumo_sessoes, hide_index=True,
            column_config={
                "leituras": st.column_config.NumberColumn("leituras", format="%d"),
                "inicio": st.column_config.DatetimeColumn("inicio", format="DD/MM/YY HH:mm:ss"),
                "fim": st.column_config.DatetimeColumn("fim", format="DD/MM/YY HH:mm:ss"),
                "duracao_min": st.column_config.NumberColumn("duracao (min)", format="%.1f"),
            },
        )

colunas_serie = st.multiselect(
    "Colunas para plotar",
    numericas,
    default=[padrao],
    help="Cada coluna vira um grafico proprio — escalas diferentes num eixo so "
         "esconderiam a variacao da menor.",
)

if not colunas_serie:
    st.caption("Selecione ao menos uma coluna.")
else:
    serie = D.r_serie_temporal(escolhido, tuple(colunas_serie))

    if serie["reamostrado"]:
        st.caption(
            f"**Reamostrado**: {serie['n_original']:,} leituras agrupadas em blocos de "
            f"{serie['fator']} ({serie['n_pontos']:,} pontos por grafico). A linha e a "
            "mediana do bloco; a faixa clara e o intervalo min-max, que preserva os "
            "picos — em vibracao o pico raro e o sinal, nao o ruido.".replace(",", ".")
        )

    for coluna in colunas_serie:
        d = serie["dados"][serie["dados"]["coluna"] == coluna]
        if d.empty:
            continue

        # `detail="sessao"` faz o Vega desenhar uma linha por sessao em vez de
        # uma so atravessando os gaps.
        base_ch = alt.Chart(d).encode(
            x=alt.X("created_at:T", title=None),
            detail=alt.Detail("sessao:N"),
        )

        faixa = base_ch.mark_area(opacity=0.25, color="#d1495b").encode(
            y=alt.Y("minimo:Q", title=coluna, scale=alt.Scale(zero=False)),
            y2="maximo:Q",
        )
        linha = base_ch.mark_line(strokeWidth=1.4, color="#d1495b").encode(
            y=alt.Y("valor:Q", title=coluna, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("created_at:T", title="instante", format="%d/%m/%y %H:%M:%S"),
                alt.Tooltip("sessao:Q", title="sessao"),
                alt.Tooltip("valor:Q", title="mediana", format=".4f"),
                alt.Tooltip("minimo:Q", format=".4f"),
                alt.Tooltip("maximo:Q", format=".4f"),
            ],
        )

        grafico = (faixa + linha) if serie["reamostrado"] else linha
        st.altair_chart(grafico.properties(height=220), width="stretch")

with st.expander("Base do rotulo em ordem cronologica"):
    st.caption(
        "`sessao` identifica a campanha de coleta; `delta_s` e o intervalo desde a "
        "leitura anterior dentro da sessao (vazio na primeira linha de cada sessao)."
    )
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
        file_name=f"{escolhido}_cronologico.csv",
        mime="text/csv",
    )

# --- 2e. Outliers dentro do rotulo ----------------------------------------
with st.expander("Outliers dentro deste rotulo (criterio IQR, apenas reportado)"):
    st.caption(
        "Limites recalculados **dentro** do rotulo. Isso separa a variacao natural "
        "da classe de um pico que destoa da propria classe — globalmente, toda "
        "leitura de um defeito severo pareceria outlier, o que nao ajuda."
    )
    st.dataframe(
        D.r_outliers_do_rotulo(escolhido)[
            [
                "coluna",
                "mediana",
                "q1",
                "q3",
                "lim_inferior",
                "lim_superior",
                "min",
                "max",
                "outliers",
                "pct_outliers",
            ]
        ],
        hide_index=True,
        height=320,
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
