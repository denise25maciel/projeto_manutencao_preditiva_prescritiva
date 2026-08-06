"""Do arquivo bruto aos procedimentos, contado como uma historia so.

Antes eram varias telas separadas no menu, cada uma respondendo um pedaco e
cabendo a quem lesse costurar a ordem — e a ordem e justamente o argumento. Aqui
elas viram atos de uma sequencia unica:

    ato 1   o que chegou          quanto dado, de quando, com que rotulos
    ato 2   o arquivo cru         a serie sem ordenar, agrupar nem reamostrar
    ato 3   da para confiar?      o que veio torto, e quanto disso pesa
    ato 4   o que os dados dizem  o comportamento medido de cada falha
    ato 5   de leitura a evento   as linhas viram ocorrencias contaveis
    ato 6   os procedimentos      a outra fonte: os 6 manuais da empresa

A ordem carrega o argumento. O ato 2 mostra o arquivo **antes** de qualquer
correcao — e a unica secao que nao ordena por data —, e por isso vem antes do
levantamento de qualidade: primeiro se ve o problema, depois se mede. O ato 3
vem antes do 4 porque ler a assinatura de uma falha sem saber que o arquivo tem
leituras repetidas e horarios errados e ler numero sem saber a margem dele.

E os atos 5 e 6 fecham a preparacao das **duas fontes** que o resto do sistema
cruza: o ato 5 transforma linhas em ocorrencias contaveis, e o ato 6 traz os
manuais. Os dois so se encontram pela coluna `fault`, via `fault_map.yaml` —
numero nunca e comparado com texto.

Cada ato mora num `_secao_*.py`. Sao os arquivos das antigas paginas,
convertidos em `render()` — modulos fora de `pages/` nao viram item de menu.

**O nome do arquivo vira o rotulo do menu.** O Streamlit deriva o nome da pagina
do nome do script (`source_util.page_icon_and_name`), trocando `_` por espaco, e
`set_page_config` nao muda isso — ele so define o titulo da aba do navegador.
Como o entrypoint se chama `app.py`, o primeiro item do menu lateral aparece
como **"app"**, e nao como "Analise de Dados".

E o preco de ter o nome convencional de entrypoint, que o README e a estrutura
do projeto ja usavam. Para recuperar o rotulo sem renomear o arquivo, o caminho
e migrar a navegacao para `st.navigation` / `st.Page`, onde o titulo de cada
pagina e declarado no codigo em vez de deduzido do nome do arquivo.

Rodar com:  streamlit run ui/app.py
"""

from __future__ import annotations

import altair as alt
import streamlit as st

import _dados as D
import _secao_dados_brutos
import _secao_documentos
import _secao_eventos
import _secao_falhas
import _secao_qualidade

D.configurar_pagina("Analise de Dados")

st.title("🔧 Manutencao Prescritiva — Dos dados aos procedimentos")
st.caption("Entender as duas fontes antes de cruzar qualquer coisa.")

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

A tela e longa de proposito — e uma sequencia, nao um painel. Seis perguntas,
nesta ordem:
"""
)

c1, c2, c3 = st.columns(3)
c1.info("**Ato 1 — O que chegou?**\n\nQuanto dado, de quando, com que rotulos.")
c2.info("**Ato 2 — Como e o arquivo cru?**\n\nA serie sem nenhum tratamento.")
c3.info("**Ato 3 — Da para confiar?**\n\nO que veio torto, e quanto disso pesa.")

c4, c5, c6 = st.columns(3)
c4.info("**Ato 4 — O que os dados dizem?**\n\nO comportamento medido de cada falha.")
c5.info("**Ato 5 — Quantas vezes aconteceu?**\n\nLeituras seguidas viram um evento.")
c6.info("**Ato 6 — O que fazer a respeito?**\n\nOs 6 manuais e o que cada um cobre.")

st.caption(
    "A ordem carrega o argumento: o ato 2 mostra o arquivo antes de qualquer "
    "correcao, e o ato 3 vem antes do 4 porque ler a assinatura de uma falha sem "
    "saber que o arquivo tem leituras repetidas e horarios errados e ler numero "
    "sem saber a margem dele. Os atos 5 e 6 preparam as **duas fontes** que o "
    "sistema cruza — e elas so se encontram pela coluna `fault`, nunca por "
    "semelhanca entre numero e texto."
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
st.header("Ato 3 — Qualidade dos dados", divider="gray")


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
# ATO 5 — De leitura para evento
# ==========================================================================
st.header("Ato 5 — Quantas vezes isso aconteceu", divider="gray")

st.markdown(
    """
Ate aqui a unidade foi a **linha do arquivo**. Ela nao responde a pergunta que o
tecnico faz: `rolamento_inner` tem 13 mil linhas, e isso nao sao 13 mil falhas —
sao 34 horas medindo a mesma. Contar linha por ocorrencia daria "4.200 vezes"
para uma unica sessao de bancada.

Um **evento** e uma vez em que a maquina foi medida com o mesmo defeito, na mesma
rotacao. Ha duas ordens possiveis para monta-los, e elas nao dao o mesmo
resultado — entao rodam lado a lado, sobre o mesmo arquivo.
"""
)

_secao_eventos.render()

st.divider()

# ==========================================================================
# ATO 6 — Os procedimentos
# ==========================================================================
st.header("Ato 6 — O que fazer a respeito", divider="gray")

st.markdown(
    """
Os cinco atos anteriores trataram de **uma** fonte: o que o sensor mediu. Ela
diz o que esta acontecendo e nunca diz o que fazer.

A segunda fonte sao os 6 manuais da empresa. Eles chegam em PDF, viram texto com
as secoes numeradas preservadas, e e dai que sai a citacao — nao basta responder
"alinhe o motor", tem de ser *"conforme o Doc2, secao 9"*.

**As duas fontes nunca se comparam diretamente.** Numero nao e comparado com
texto: o evento resolve para um rotulo, o rotulo resolve para uma familia, e a
familia aponta o documento pelo `fault_map.yaml`, que e curado a mao e
versionado. E por isso que uma falha sem manual e recusada por um `SELECT`, e
nao por uma busca que sempre acharia "o trecho menos diferente".
"""
)

_secao_documentos.render()

st.divider()

# ==========================================================================
# Fecho
# ==========================================================================
st.header("Depois desta historia", divider="gray")

st.markdown(
    """
Os seis atos acima prepararam as duas fontes: o que a maquina mediu e o que os
manuais mandam fazer. Quem usa isso sao as duas telas do menu a esquerda.

**Diagnostico** — o fluxo completo, em duas abas:

- **Diagnostico e conversa** — o tecnico descreve o problema ou chega o JSON do
  sensor, o sistema acha o procedimento e conversa sobre ele, citando documento,
  secao e pagina
- **Modelo de linguagem** — escolha do provedor (local ou API), as regras do
  prompt e o teste de conexao. Fica ali dentro, e nao numa tela separada, porque
  trocar de modelo no meio de uma conversa nao pode custar sair dela

**Classificacao** — a pergunta anterior a todas as outras: *da para descobrir a
familia so pelos numeros do sensor, sem ninguem anotar o rotulo?* Em tres abas:

- **Preparacao dos dados** — como uma leitura vira um exemplo que se pode
  aprender, e por que uma leitura sozinha nao e um
- **O modelo e o que ele vale** — a floresta, e as duas maneiras de cortar
  treino e teste que dao 92% e 44% sobre exatamente os mesmos dados. So uma das
  duas responde *"vai funcionar numa maquina nova?"*
- **Executar e ver o resultado** — o pipeline rodando de verdade, etapa por
  etapa e cronometrado, terminando no laudo dos testes. Sem cache: o tempo que
  a tela mostra e o tempo que levou
"""
)
