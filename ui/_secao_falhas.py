"""Ato 3 da narrativa — **o que os dados dizem**.

Valores unicos de `fault` e o comportamento medido de cada um.

Ordem dos blocos: primeiro a serie ao longo do tempo, depois o resumo em
medianas. Ver a curva antes da estatistica evita ler uma mediana sem saber se
ela descreve um patamar estavel ou a media de dois regimes distintos.

Era uma pagina propria (`pages/1_Analise_de_Falhas.py`). Virou secao de `app.py`
pelo mesmo motivo do ato 2, e com a mesma conversao conferida.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D



def render() -> None:
    """Desenha esta secao dentro da narrativa unica."""

    # Acima disso as colunas ficam estreitas demais para os graficos serem lidos.
    MAX_ROTULOS = 4
    CHAVE_ROTULOS = "rotulos_selecionados"

    st.header("📊 Analise de Falhas", divider="gray")
    st.caption("Escolha um ou mais tipos de falha e veja como cada um se comporta.")

    try:
        rotulos = D.r_rotulos()
    except FileNotFoundError as e:
        D.aviso_csv_ausente(e)

    familias = D.r_familias()
    # `perfil_rotulos` traz contagem e janela; `sugerir_familias` traz o agrupamento
    # proposto. So trazemos `familia_sugerida` do segundo: `e_problema` existe nos
    # dois e o merge criaria `e_problema_x` / `e_problema_y`.
    tabela = rotulos.merge(familias[["fault", "familia_sugerida"]], on="fault", how="left")

    # ==========================================================================
    # 1. Panorama
    # ==========================================================================
    st.header("1. Os tipos de falha do arquivo")

    n_total = len(tabela)
    n_problema = int(tabela["e_problema"].sum())
    n_familias = tabela["familia_sugerida"].nunique()

    # Quantos nomes ha, quantos sao defeito e quantos sao estado ja foram ditos
    # no ato 1. O que este ato acrescenta e o **agrupamento** — e so ele fica.
    st.metric("Grupos (familias)", n_familias)

    st.markdown(
        f"""
Os {n_total} nomes do ato 1 nao sao {n_total} defeitos: o mesmo problema aparece
escrito de varias formas. Agrupando por radical, eles viram **{n_familias}
familias**.

Exemplo: `rolamento_inner`, `rolamento_inner_2`, `new_rolamento_inner_0` e
`rolamento_inner_carga` viram todos a familia `rolamento_inner`.

Esse agrupamento e **automatico e provisorio** — a versao definitiva e conferida a
mao no arquivo `data/fault_map.yaml`.
"""
    )

    col_f, col_b = st.columns([2, 3])
    with col_f:
        filtro_tipo = st.radio("Mostrar", ["Todos", "So defeitos", "So estados"], horizontal=True)
    with col_b:
        busca = st.text_input("Procurar por nome", placeholder="ex.: rolamento, cocked, normal")

    vis = tabela
    if filtro_tipo == "So defeitos":
        vis = vis[vis["e_problema"]]
    elif filtro_tipo == "So estados":
        vis = vis[~vis["e_problema"]]
    if busca:
        vis = vis[vis["fault"].str.contains(busca, case=False, na=False)]

    st.caption(f"Mostrando {len(vis)} de {n_total} nomes.")
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
            "fault": "nome em `fault`",
            "familia_sugerida": "familia",
            "e_problema": st.column_config.CheckboxColumn("e defeito?"),
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "pct": st.column_config.NumberColumn("% do arquivo", format="%.2f%%"),
            "primeira": st.column_config.DatetimeColumn("1a leitura", format="DD/MM/YY HH:mm"),
            "ultima": st.column_config.DatetimeColumn("ultima leitura", format="DD/MM/YY HH:mm"),
            "span_horas": st.column_config.NumberColumn(
                "horas cobertas", format="%.1f",
                help="Tempo entre a primeira e a ultima leitura desse nome."
            ),
        },
    )
    with st.expander("Ver o total de leituras por familia", expanded=True):
        por_fam = (
            tabela.groupby("familia_sugerida", as_index=False)
            .agg(rotulos=("fault", "count"), leituras=("n_leituras", "sum"))
            .sort_values("leituras", ascending=False)
        )
        st.altair_chart(
            alt.Chart(por_fam)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X("leituras:Q", title="leituras"),
                y=alt.Y("familia_sugerida:N", sort="-x", title=None),
                tooltip=["familia_sugerida", "rotulos", "leituras"],
            )
            .properties(height=28 * len(por_fam)),
            width="stretch",
        )

    st.divider()
