"""Pagina inicial da UI — Parte 0.

Visao geral do dataset e porta de entrada para as duas telas de analise.
Rodar com:  streamlit run ui/app.py
"""

from __future__ import annotations

import altair as alt
import streamlit as st

import _dados as D

D.configurar_pagina("Visao geral")

st.title("🔧 Manutencao Prescritiva — Analise Exploratoria")
st.caption("Parte 0: entender o dado bruto antes de transformar qualquer coisa.")

try:
    resumo = D.r_resumo()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

janela = D.r_janela()
amostragem = D.r_amostragem()
rotulos = D.r_rotulos()

# --------------------------------------------------------------------------
# Cabecalho numerico
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Leituras", f"{resumo['linhas']:,}".replace(",", "."))
c2.metric("Colunas", resumo["colunas"])
c3.metric("Rotulos distintos", resumo["rotulos_distintos"])
c4.metric("Celulas nulas", f"{resumo['celulas_nulas']:,}".replace(",", "."))
c5.metric("Sessoes estimadas", amostragem["sessoes_estimadas"])

st.caption(f"Fonte: `{D.caminho_do_csv()}` — arquivo fora do controle de versao.")

st.divider()

# --------------------------------------------------------------------------
# Leitura rapida: o que ja sabemos
# --------------------------------------------------------------------------
st.subheader("Leitura rapida")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        f"""
**Janela de coleta**

- De `{janela['inicio']:%d/%m/%Y %H:%M}` a `{janela['fim']:%d/%m/%Y %H:%M}` UTC
- {janela['duracao_dias']} dias corridos
- Intervalo mediano entre leituras: **{amostragem['intervalo_mediano_s']} s**
  (o esperado era ~2 s — confirmado)
- {amostragem['pct_na_cadencia']}% das leituras respeitam essa cadencia
"""
    )

with col_b:
    n_prob = int(rotulos["e_problema"].sum())
    st.markdown(
        f"""
**Rotulos**

- {resumo['rotulos_distintos']} valores distintos em `fault`
- {n_prob} classificados como **defeito**, {resumo['rotulos_distintos'] - n_prob}
  como **estado** (normal, teste, baseline, motor desligado...)
- O maior rotulo cobre {rotulos.iloc[0]['pct']}% das linhas
"""
    )

# Achados que contrariam a expectativa inicial do GUIA.md. Ficam em destaque
# porque mudam decisao de engenharia, nao sao curiosidade.
st.markdown("**Tres achados que contrariam a suposicao inicial:**")

st.warning(
    f"**`created_at` nao esta em ordem cronologica.** O arquivo tem "
    f"{amostragem['cortes']} cortes acima de 60 s e saltos negativos de dezenas de "
    "dias — sao sessoes gravadas em epocas diferentes e concatenadas fora de ordem. "
    "Qualquer janela deslizante (a mediana movel da Parte 3) precisa ordenar antes."
)
st.warning(
    "**`z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` nao sao constantes em "
    "61 Hz.** Tem 79 e 50 valores distintos. 61 Hz e a moda (60% e 49% das linhas), "
    "nao o valor unico — as colunas carregam informacao e nao devem ser descartadas."
)
st.warning(
    f"**{resumo['rotulos_distintos']} rotulos, nao ~10.** Ha erros de digitacao "
    "(`mortor_desligado_novo`, `normla_carga_3_3`, `cockecocked_adxl_0`) e sufixos de "
    "sessao (`_2`, `_pos_2`, `_carga`, `_adxl_0`, `new_*`) que multiplicam a mesma "
    "familia. Consolidar isso e a Parte 1."
)

st.divider()

# --------------------------------------------------------------------------
# Distribuicao dos rotulos
# --------------------------------------------------------------------------
st.subheader("Distribuicao das leituras por rotulo")

top = st.slider("Quantos rotulos mostrar", 5, 60, 25, step=5)
recorte = rotulos.head(top)

grafico = (
    alt.Chart(recorte)
    .mark_bar()
    .encode(
        x=alt.X("n_leituras:Q", title="leituras"),
        y=alt.Y("fault:N", sort="-x", title=None),
        color=alt.Color(
            "e_problema:N",
            title="e defeito?",
            scale=alt.Scale(domain=[True, False], range=["#d1495b", "#5c8a8a"]),
        ),
        tooltip=["fault", "n_leituras", "pct", "e_problema"],
    )
    .properties(height=max(240, 18 * len(recorte)))
)
st.altair_chart(grafico, width="stretch")

st.info(
    "A barra conta **linhas**, nao ocorrencias. Um rotulo com 13.000 leituras "
    "coletadas ao longo de 34 h e uma sessao continua, nao 13.000 falhas. "
    "Colapsar isso em episodios e o primeiro item da Parte 1."
)

st.divider()
st.markdown(
    """
### Para onde ir

- **Analise de Falhas** — valores unicos de `fault` e a assinatura de vibracao de cada um
- **Qualidade dos Dados** — como o dado chegou: nulos, redundancias, duplicatas e outliers

Use o menu na barra lateral.
"""
)
