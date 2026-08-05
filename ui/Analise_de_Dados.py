"""A analise dos dados, contada como uma historia so — Parte 0.

Antes eram quatro telas separadas no menu: a visao geral, os Dados Brutos, a
Qualidade dos Dados e a Analise de Falhas. Separadas, cada uma respondia um
pedaco e cabia a quem lesse costurar a ordem — e a ordem e justamente o
argumento. Aqui elas viram quatro atos de uma sequencia unica:

    ato 1   o que chegou          quanto dado, de quando, com que rotulos
    ato 2   o arquivo cru         a serie sem ordenar, agrupar nem reamostrar
    ato 3   da para confiar?      o que veio torto, e quanto disso pesa
    ato 4   o que os dados dizem  o comportamento medido de cada falha

A ordem carrega o argumento. O ato 2 mostra o arquivo **antes** de qualquer
correcao — e a unica tela que nao ordena por data —, e por isso vem antes do
levantamento de qualidade: primeiro se ve o problema, depois se mede. E o ato 3
vem antes do 4 porque ler a assinatura de uma falha sem saber que o arquivo tem
leituras repetidas e horarios errados e ler numero sem saber a margem dele.

Os atos 2, 3 e 4 moram em `_secao_dados_brutos.py`, `_secao_qualidade.py` e
`_secao_falhas.py`. Sao os arquivos das antigas paginas, convertidos em
`render()` — modulos fora de `pages/` nao viram item de menu.

**O nome do arquivo e o rotulo do menu.** O Streamlit deriva o nome da pagina do
nome do script (`source_util.page_icon_and_name`), trocando `_` por espaco;
`set_page_config` nao muda isso. Por isso o entrypoint se chama
`Analise_de_Dados.py` e nao `app.py`.

Rodar com:  streamlit run ui/Analise_de_Dados.py
"""

from __future__ import annotations

import altair as alt
import streamlit as st

import _dados as D
import _secao_dados_brutos
import _secao_falhas
import _secao_qualidade

D.configurar_pagina("Analise de Dados")

st.title("🔧 Manutencao Prescritiva — A analise dos dados")
st.caption("Entender o dado antes de mudar qualquer coisa.")

try:
    resumo = D.r_resumo()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

janela = D.r_janela()
amostragem = D.r_amostragem()
rotulos = D.r_rotulos()

# --------------------------------------------------------------------------
# O shape, antes de qualquer outra coisa
# --------------------------------------------------------------------------
#
# A primeira pergunta que se faz a um dataset: quantas linhas, quantas colunas.
# Vem antes da narrativa porque e o que dimensiona tudo o que vem depois — 166
# mil linhas e um numero que muda o que faz sentido desenhar na tela.
st.markdown(
    f"### `dataset.shape` → **({resumo['linhas']:,}, {resumo['colunas']})**"
    .replace(",", ".")
)
st.caption(
    f"{resumo['linhas']:,} linhas × {resumo['colunas']} colunas".replace(",", ".")
    + " — uma linha por leitura do sensor, uma coluna por medida "
      "(mais `id`, `created_at` e o rotulo `fault`)."
)

st.markdown(
    """
Esta e a etapa de **exploracao**. Aqui nada e corrigido, filtrado ou apagado:
so descrevemos o que veio no arquivo. Limpar os dados vem depois, na proxima etapa.

A tela e longa de proposito — e uma sequencia, nao um painel. Quatro perguntas,
nesta ordem:
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.info("**Ato 1 — O que chegou?**\n\nQuanto dado, de quando, com que rotulos.")
c2.info("**Ato 2 — Como e o arquivo cru?**\n\nA serie sem nenhum tratamento.")
c3.info("**Ato 3 — Da para confiar?**\n\nO que veio torto, e quanto disso pesa.")
c4.info("**Ato 4 — O que os dados dizem?**\n\nO comportamento medido de cada falha.")

st.caption(
    "A ordem carrega o argumento: o ato 2 mostra o arquivo antes de qualquer "
    "correcao, e o ato 3 vem antes do 4 porque ler a assinatura de uma falha sem "
    "saber que o arquivo tem leituras repetidas e horarios errados e ler numero "
    "sem saber a margem dele."
)

st.divider()

# ==========================================================================
# ATO 1 — O que chegou
# ==========================================================================
st.header("Ato 1 — O que chegou", divider="gray")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Leituras", f"{resumo['linhas']:,}".replace(",", "."))
c2.metric("Colunas", resumo["colunas"])
c3.metric("Tipos de falha", resumo["rotulos_distintos"])
c4.metric("Campos vazios", f"{resumo['celulas_nulas']:,}".replace(",", "."))
c5.metric("Sessoes de coleta", amostragem["sessoes_estimadas"])

st.caption(f"Arquivo: `{D.caminho_do_csv()}` — fica fora do Git, e dado da empresa.")

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

# ==========================================================================
# ATO 2 — O arquivo cru
# ==========================================================================
st.header("Ato 2 — Como e o arquivo cru", divider="gray")

st.markdown(
    """
O ato 1 resumiu. Aqui o arquivo aparece **sem nenhum tratamento**: sem ordenar por
data, sem agrupar leituras vizinhas, sem reamostrar. O eixo horizontal e a posicao
da linha no arquivo — linha 0, linha 1, linha 2 — e nao o tempo.

E a unica secao que nao corrige nada, e por isso vem antes do levantamento de
qualidade: primeiro se ve o problema, depois se mede.
"""
)

_secao_dados_brutos.render()

st.divider()

# ==========================================================================
# ATO 3 — Da para confiar?
# ==========================================================================
st.header("Ato 3 — Da para confiar no que chegou?", divider="gray")

st.markdown(
    """
Os atos 1 e 2 mostraram **quanto** dado existe e **como ele se parece**. Antes de
tirar conclusao dele, a pergunta e se ele se sustenta: campos vazios, colunas que
repetem a mesma medida, leituras identicas a anterior, horarios impossiveis.

Nada disto e corrigido aqui — e levantamento. O que fica pendente de proposito
esta registrado como pendencia, com o motivo.
"""
)

_secao_qualidade.render()

st.divider()

# ==========================================================================
# ATO 3 — O que os dados dizem
# ==========================================================================
st.header("Ato 4 — O que os dados dizem", divider="gray")

st.markdown(
    """
Com o tamanho conhecido (ato 1) e as ressalvas na mesa (ato 3), da para olhar o
**comportamento medido** de cada falha: como os valores variam no tempo, o que
separa um defeito de outro e onde dois rotulos diferentes medem a mesma coisa.

E aqui que aparece a materia-prima do motor de similaridade — as assinaturas por
rotulo que a Parte 3 usa para dizer *"esse evento se parece com aqueles"*.
"""
)

_secao_falhas.render()

st.divider()

# ==========================================================================
# Fecho
# ==========================================================================
st.header("Depois desta historia", divider="gray")

st.markdown(
    """
Os quatro atos acima respondem *o que veio e o que ele diz*. As telas do menu a
esquerda seguem dali:

- **Documentos** — os 6 manuais de procedimento e quais falhas cada um cobre
- **Eventos** — leituras seguidas do mesmo defeito agrupadas em ocorrencias
- **Modelo de Linguagem** — escolha do provedor (local ou API) e teste de conexao
- **Diagnostico** — o fluxo completo: chega o JSON do sensor, o sistema diz qual e
  a falha e conversa sobre o procedimento, citando documento, secao e pagina
"""
)
