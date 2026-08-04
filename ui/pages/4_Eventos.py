"""Agrupamento das leituras em eventos.

Primeira tela que TRANSFORMA o dado em vez de so descreve-lo. O que aparece aqui
e o resultado de `mp.ingestion.sensors`, nao um calculo da propria pagina.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Eventos", "🧩")

st.title("🧩 Eventos")
st.caption("Agrupar leituras soltas em ocorrencias contaveis.")

st.markdown(
    """
### O problema

O arquivo tem 166.796 linhas. Se alguem perguntar *"quantas vezes esse defeito
aconteceu?"*, contar linhas responderia **13.000** para `rolamento_inner`.

Errado. Foram algumas medicoes longas, nao 13 mil falhas.

### A solucao

Agrupar leituras seguidas do mesmo defeito num unico **evento**, com inicio, fim
e duracao. E o evento que responde "quantas vezes".

### A regra usada aqui

**Um evento novo comeca quando o nome da falha muda.** Nada mais.
"""
)

try:
    leituras, eventos = D.r_eventos()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

validacao = D.r_validacao_eventos()
resumo = D.r_resumo_eventos()
diagnostico = D.r_diagnostico_eventos()
bruto = D.dados()
ordenacao = D.r_ordenacao()

# ==========================================================================
# 0. Ordenacao — pre-requisito de tudo
# ==========================================================================
st.header("1. Antes de agrupar: ordenar por data")

if ordenacao["ja_ordenado"]:
    st.info("O arquivo ja vem em ordem de data. A ordenacao nao muda nada.")
else:
    st.error(
        f"""
### O arquivo NAO vem ordenado por data

Esta e a informacao mais importante desta tela.

O `banner.csv` chega com as linhas fora de ordem cronologica. Nao e uma pequena
bagunca: **{ordenacao['pct_fora_do_lugar']:.0f}% das linhas
({ordenacao['linhas_fora_do_lugar']:,}) mudam de lugar** quando ordenamos.

A coluna `id` tambem nao esta em ordem.

**Por isso ordenamos por `created_at` antes de qualquer agrupamento.**
""".replace(",", ".")
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas fora do lugar", f"{ordenacao['linhas_fora_do_lugar']:,}".replace(",", "."),
              f"{ordenacao['pct_fora_do_lugar']:.0f}% do arquivo", delta_color="inverse")
    c2.metric("Linhas que voltam no tempo", ordenacao["linhas_que_voltam_no_tempo"],
              help="Linhas cuja data e ANTERIOR a da linha de cima.")
    c3.metric("Maior salto para tras",
              f"{abs(ordenacao['maior_salto_para_tras_dias']):.0f} dias")

    exemplo = D.r_exemplo_desordem()
    if not exemplo.empty:
        st.caption(
            "Duas linhas vizinhas no arquivo, como ele veio. A de baixo e de "
            "**um mes antes** da de cima:"
        )
        st.dataframe(
            exemplo,
            hide_index=True,
            column_config={
                "posicao_no_arquivo": st.column_config.NumberColumn("linha n", format="%d"),
                "created_at": st.column_config.DatetimeColumn(
                    "data e hora", format="DD/MM/YYYY HH:mm:ss"
                ),
                "fault": "tipo de falha",
            },
        )

    st.warning(
        f"""
**O que aconteceria sem ordenar**

Agrupar linhas vizinhas de um arquivo desordenado junta leituras que nao tem
relacao entre si. Duas linhas coladas no arquivo podem estar a um mes de
distancia na realidade.

O resultado seriam **{ordenacao['eventos_sem_ordenar']} eventos** em vez de
{len(eventos)} — e cada um deles misturaria momentos diferentes, com inicio e fim
sem sentido.

A ordenacao acontece dentro de `construir_eventos`, sempre. Nao depende de o
arquivo chegar arrumado.
"""
    )

st.divider()

# ==========================================================================
# 1. Antes e depois
# ==========================================================================
st.header("2. O que mudou")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Linhas no arquivo", f"{len(bruto):,}".replace(",", "."))
c2.metric("Eventos", f"{len(eventos):,}".replace(",", "."))
c3.metric("Reducao", f"{len(bruto) / max(len(eventos), 1):.0f}x")
c4.metric("Leituras por evento", f"{eventos['n_leituras'].median():.0f}",
          help="Valor tipico.")

st.caption(
    "Nenhuma leitura foi apagada. Todas continuam la — agora sabendo a qual "
    "evento pertencem."
)

# ==========================================================================
# 2. Validacao
# ==========================================================================
st.header("3. O agrupamento esta correto?")

st.markdown(
    "Cinco checagens que ou passam ou falham. Se alguma falhar, o resultado "
    "abaixo nao vale."
)

todas_ok = bool(validacao["passou"].all())
if todas_ok:
    st.success(f"**As {len(validacao)} checagens passaram.**")
else:
    st.error("**Alguma checagem falhou.** Nao usar o resultado.")

st.dataframe(
    validacao,
    hide_index=True,
    column_config={
        "checagem": st.column_config.TextColumn("checagem", width="large"),
        "passou": st.column_config.CheckboxColumn("ok?"),
        "detalhe": st.column_config.TextColumn("detalhe", width="medium"),
    },
)

# ==========================================================================
# 3. Quantas vezes cada falha aconteceu
# ==========================================================================
st.header("4. Quantas vezes cada falha aconteceu")

st.markdown("Esta e a tabela que responde a pergunta do operador.")

st.dataframe(
    resumo[
        ["fault", "eventos", "leituras", "leituras_por_evento",
         "duracao_mediana_min", "duracao_total_h", "primeira", "ultima"]
    ],
    hide_index=True,
    height=420,
    column_config={
        "fault": "tipo de falha",
        "eventos": st.column_config.NumberColumn("ocorrencias", format="%d"),
        "leituras": st.column_config.NumberColumn("leituras", format="%d"),
        "leituras_por_evento": st.column_config.NumberColumn(
            "leituras por ocorrencia", format="%.0f"
        ),
        "duracao_mediana_min": st.column_config.NumberColumn(
            "duracao tipica (min)", format="%.0f"
        ),
        "duracao_total_h": st.column_config.NumberColumn("horas no total", format="%.1f"),
        "primeira": st.column_config.DatetimeColumn("primeira vez", format="DD/MM/YY HH:mm"),
        "ultima": st.column_config.DatetimeColumn("ultima vez", format="DD/MM/YY HH:mm"),
    },
)

st.caption("Comparacao entre contar linhas e contar ocorrencias:")

top = resumo.head(15)
comparativo = pd.concat(
    [
        top[["fault", "leituras"]].rename(columns={"leituras": "quantidade"}).assign(
            medida="contando linhas"
        ),
        top[["fault", "eventos"]].rename(columns={"eventos": "quantidade"}).assign(
            medida="contando ocorrencias"
        ),
    ]
)

st.altair_chart(
    alt.Chart(comparativo)
    .mark_bar()
    .encode(
        x=alt.X("quantidade:Q", title="quantidade", scale=alt.Scale(type="symlog")),
        y=alt.Y("fault:N", sort=list(top["fault"]), title=None),
        color=alt.Color(
            "medida:N", title=None,
            scale=alt.Scale(domain=["contando linhas", "contando ocorrencias"],
                            range=["#b0b0b0", "#d1495b"]),
            legend=alt.Legend(orient="bottom"),
        ),
        yOffset="medida:N",
        tooltip=["fault", "medida", "quantidade"],
    )
    .properties(height=36 * len(top)),
    width="stretch",
)

# ==========================================================================
# 4. A lista de eventos
# ==========================================================================
st.header("5. Todos os eventos")

filtro = st.multiselect(
    "Filtrar por tipo de falha",
    sorted(eventos["fault"].dropna().unique()),
    default=[],
    help="Deixe vazio para ver todos.",
)
vis = eventos[eventos["fault"].isin(filtro)] if filtro else eventos

st.caption(f"{len(vis)} de {len(eventos)} eventos.")
st.dataframe(
    vis[["evento", "fault", "n_leituras", "inicio", "fim", "duracao_min"]],
    hide_index=True,
    height=420,
    column_config={
        "evento": st.column_config.NumberColumn("n", format="%d"),
        "fault": "tipo de falha",
        "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
        "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm:ss"),
        "fim": st.column_config.DatetimeColumn("terminou", format="DD/MM/YY HH:mm:ss"),
        "duracao_min": st.column_config.NumberColumn("duracao (min)", format="%.1f"),
    },
)

st.download_button(
    "Baixar os eventos em CSV",
    eventos.to_csv(index=False).encode("utf-8"),
    file_name="eventos.csv",
    mime="text/csv",
)

# ==========================================================================
# 5. O custo de usar so o rotulo
# ==========================================================================
st.header("6. O que esta regra deixa passar")

_, eventos_10s = D.r_eventos(10.0)

st.markdown(
    f"""
A regra atual quebra o evento **so quando o nome da falha muda**. Ela nao percebe
quando a coleta simplesmente parou e recomecou depois, com o mesmo nome.

Quando isso acontece, duas medicoes distantes viram um evento so.
"""
)

d1, d2, d3 = st.columns(3)
d1.metric("Eventos com a regra atual", len(eventos))
d2.metric("Eventos incluindo pausas de 10 s", len(eventos_10s))
d3.metric(
    "Eventos com buraco interno", len(diagnostico),
    f"{len(diagnostico) / max(len(eventos), 1) * 100:.0f}% do total",
    delta_color="inverse",
)

if len(diagnostico):
    pior = diagnostico.iloc[0]
    st.warning(
        f"""
**O caso mais extremo:** o evento {int(pior['evento'])} (`{pior['fault']}`) aparece
com **{pior['duracao_min'] / 60:.0f} horas de duracao** — mas tem
**{pior['maior_buraco_h']:.0f} horas seguidas sem nenhuma leitura** por dentro.

Nao foi uma medicao de {pior['duracao_min'] / 60:.0f} horas. Foram duas medicoes
separadas por {pior['maior_buraco_h']:.0f} horas, que a regra juntou porque o nome
da falha nao mudou entre elas.
"""
    )

    st.caption(
        "Os eventos com interrupcao interna maior que 1 minuto. A coluna "
        "*maior pausa* mostra o tamanho do buraco."
    )
    st.dataframe(
        diagnostico[
            ["evento", "fault", "n_leituras", "inicio", "fim",
             "duracao_min", "maior_buraco_h"]
        ],
        hide_index=True,
        height=380,
        column_config={
            "evento": st.column_config.NumberColumn("n", format="%d"),
            "fault": "tipo de falha",
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
            "fim": st.column_config.DatetimeColumn("terminou", format="DD/MM/YY HH:mm"),
            "duracao_min": st.column_config.NumberColumn("duracao (min)", format="%.0f"),
            "maior_buraco_h": st.column_config.NumberColumn("maior pausa (h)", format="%.1f"),
        },
    )

st.info(
    """
**Isto esta aqui de proposito, nao e um defeito escondido.**

A decisao foi comecar so com o rotulo. O codigo ja aceita a regra de tempo — ela
esta pronta e desligada. Ligar e trocar um parametro.

A tela mostra o custo para a escolha continuar sendo informada.
"""
)

# ==========================================================================
# 6. Pendencia P1
# ==========================================================================
zero = eventos[eventos["duracao_s"] == 0]
if len(zero):
    st.header("7. Um evento impossivel")
    st.error(
        f"""
Existe **{len(zero)} evento com duracao zero** e
{int(zero.iloc[0]['n_leituras']):,} leituras dentro.

E o bloco de `{zero.iloc[0]['fault']}` em que todas as leituras receberam a mesma
data e hora — a pendencia **P1**, detalhada na tela *Qualidade dos Dados*.

Mil medicoes num unico instante e fisicamente impossivel. O erro esta no dado, nao
no agrupamento. Fica marcado, nao corrigido.
""".replace(",", ".")
    )
    st.dataframe(
        zero[["evento", "fault", "n_leituras", "inicio", "fim", "duracao_min"]],
        hide_index=True,
        column_config={
            "evento": st.column_config.NumberColumn("n", format="%d"),
            "fault": "tipo de falha",
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm:ss"),
            "fim": st.column_config.DatetimeColumn("terminou", format="DD/MM/YY HH:mm:ss"),
            "duracao_min": st.column_config.NumberColumn("duracao (min)", format="%.1f"),
        },
    )
