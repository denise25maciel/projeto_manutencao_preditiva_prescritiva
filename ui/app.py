"""Pagina inicial da UI — Parte 0.

Visao geral do dataset e porta de entrada para as telas de analise.
Rodar com:  streamlit run ui/app.py
"""

from __future__ import annotations

import altair as alt
import streamlit as st

import _dados as D

D.configurar_pagina("Visao geral")

st.title("🔧 Manutencao Prescritiva — Analise dos Dados")
st.caption("Entender o dado antes de mudar qualquer coisa.")

try:
    resumo = D.r_resumo()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

janela = D.r_janela()
amostragem = D.r_amostragem()
rotulos = D.r_rotulos()

st.markdown(
    """
Esta e a etapa de **exploracao**. Aqui nada e corrigido, filtrado ou apagado:
so descrevemos o que veio no arquivo. Limpar os dados vem depois, na proxima etapa.
"""
)

# --------------------------------------------------------------------------
# Cabecalho numerico
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Leituras", f"{resumo['linhas']:,}".replace(",", "."))
c2.metric("Colunas", resumo["colunas"])
c3.metric("Tipos de falha", resumo["rotulos_distintos"])
c4.metric("Campos vazios", f"{resumo['celulas_nulas']:,}".replace(",", "."))
c5.metric("Sessoes de coleta", amostragem["sessoes_estimadas"])

st.caption(f"Arquivo: `{D.caminho_do_csv()}` — fica fora do Git, e dado da empresa.")

st.divider()

# --------------------------------------------------------------------------
# Leitura rapida
# --------------------------------------------------------------------------
st.subheader("O que temos")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        f"""
**Quando os dados foram coletados**

- De {janela['inicio']:%d/%m/%Y} a {janela['fim']:%d/%m/%Y}, ou seja
  {janela['duracao_dias']:.0f} dias
- O sensor gravava uma leitura a cada **{amostragem['intervalo_mediano_s']:.0f} segundos**
- {amostragem['pct_na_cadencia']:.0f}% das leituras seguem esse ritmo
"""
    )

with col_b:
    n_prob = int(rotulos["e_problema"].sum())
    st.markdown(
        f"""
**O que foi medido**

- A coluna `fault` tem {resumo['rotulos_distintos']} valores diferentes
- {n_prob} sao **defeitos**; {resumo['rotulos_distintos'] - n_prob} sao apenas
  **estados da maquina** (normal, teste, motor desligado...)
- O valor mais comum sozinho responde por {rotulos.iloc[0]['pct']:.0f}% das linhas
"""
    )

st.markdown("### Tres coisas que esperavamos e nao se confirmaram")

st.warning(
    f"""
**1. Os dados nao estao em ordem de data.**

Abrimos o arquivo esperando uma linha do tempo continua. Nao e. Ele junta
**{amostragem['sessoes_estimadas']} gravacoes curtas** feitas em dias diferentes, e
elas nao estao na ordem certa: uma linha pode ser de junho e a seguinte, de maio.

*Por que importa:* qualquer calculo que dependa de "a leitura anterior" precisa
ordenar por data antes, senao compara coisas sem relacao.
"""
)
st.warning(
    """
**2. As colunas de frequencia nao sao fixas em 61 Hz.**

Suspeitavamos que `z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` tivessem
sempre o mesmo valor — se fosse assim, seriam inuteis e poderiam ser descartadas.
Elas tem 79 e 50 valores diferentes. 61 Hz e apenas o valor mais comum.

*Por que importa:* as colunas tem informacao e ficam. A frequencia muda justamente
em alguns defeitos.
"""
)
st.warning(
    f"""
**3. Sao {resumo['rotulos_distintos']} nomes de falha, nao uns 10.**

A maior parte e o mesmo defeito escrito de formas diferentes. Ha erros de digitacao
(`mortor_desligado_novo`, `normla_carga_3_3`) e sufixos que so indicam a sessao de
coleta (`_2`, `_pos_2`, `_carga`, `new_`).

*Por que importa:* juntar esses nomes em grupos e o primeiro passo da proxima etapa.
"""
)

st.divider()

# --------------------------------------------------------------------------
# Distribuicao dos rotulos
# --------------------------------------------------------------------------
st.subheader("Quantas leituras cada tipo de falha tem")

top = st.slider("Quantos mostrar", 5, 60, 25, step=5)
recorte = rotulos.head(top)

st.altair_chart(
    alt.Chart(recorte)
    .mark_bar()
    .encode(
        x=alt.X("n_leituras:Q", title="numero de leituras"),
        y=alt.Y("fault:N", sort="-x", title=None),
        color=alt.Color(
            "e_problema:N",
            title="e defeito?",
            scale=alt.Scale(domain=[True, False], range=["#d1495b", "#5c8a8a"]),
        ),
        tooltip=["fault", "n_leituras", "pct", "e_problema"],
    )
    .properties(height=max(240, 18 * len(recorte))),
    width="stretch",
)

st.info(
    """
**Cuidado ao ler este grafico.** A barra conta **linhas do arquivo**, nao quantas
vezes o defeito aconteceu.

Exemplo: `rolamento_inner` tem 13 mil linhas. Mas elas foram gravadas ao longo de
34 horas seguidas — e **uma** falha sendo medida sem parar, nao 13 mil falhas.

Agrupar leituras seguidas num unico evento e tarefa da proxima etapa.
"""
)

st.divider()
st.markdown(
    """
### As outras telas

Use o menu a esquerda.

- **Analise de Falhas** — escolha um ou mais tipos de falha e veja como cada um se
  comporta ao longo do tempo e o que os diferencia
- **Qualidade dos Dados** — como o arquivo chegou: campos vazios, colunas repetidas,
  leituras duplicadas e valores fora do normal
- **Documentos** — os 6 manuais de procedimento e quais falhas cada um cobre
"""
)
