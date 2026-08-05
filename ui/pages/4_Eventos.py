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
        "Os eventos com interrupcao interna maior que 1 minuto. **Atencao as "
        "unidades:** a duracao esta em HORAS e a maior pausa tambem, para poderem "
        "ser comparadas na mesma linha."
    )
    tabela_diag = diagnostico.copy()
    tabela_diag["duracao_h"] = (tabela_diag["duracao_min"] / 60).round(1)

    st.dataframe(
        tabela_diag[
            ["evento", "fault", "n_leituras", "inicio", "fim",
             "duracao_h", "maior_buraco_h"]
        ],
        hide_index=True,
        height=380,
        column_config={
            "evento": st.column_config.NumberColumn("n", format="%d"),
            "fault": "tipo de falha",
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
            "fim": st.column_config.DatetimeColumn("terminou", format="DD/MM/YY HH:mm"),
            "duracao_h": st.column_config.NumberColumn(
                "duracao total (horas)", format="%.1f h",
                help="Do inicio ao fim do evento, incluindo as pausas de dentro."
            ),
            "maior_buraco_h": st.column_config.NumberColumn(
                "maior pausa (horas)", format="%.1f h",
                help="A maior interrupcao sem nenhuma leitura, dentro do evento."
            ),
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

# --------------------------------------------------------------------------
# 6.1 Como o corte por tempo agiria
# --------------------------------------------------------------------------
st.subheader("Se ligassemos o corte por tempo, onde ele cairia")

corte = D.r_analise_corte()
est = corte["estatisticas"]
vazio = corte["vazio"]

st.markdown(
    """
Aqui olhamos **so os intervalos que existem dentro dos eventos atuais** — o tempo
entre uma leitura e a seguinte, sem atravessar a fronteira de um evento para outro.

Se um dia decidirmos cortar tambem por tempo, e nestes intervalos que o corte agiria.
"""
)

e1, e2, e3, e4 = st.columns(4)
e1.metric("Menor intervalo", f"{est['minimo_s']:.1f} s")
e2.metric("Intervalo tipico", f"{est['mediana_s']:.1f} s")
e3.metric("Intervalo medio", f"{est['media_s']:.1f} s")
e4.metric("Maior intervalo", f"{est['maximo_s'] / 3600:.0f} h")

st.caption(
    f"Sobre {est['n']:,} intervalos.".replace(",", ".")
    + " O tipico e 2 segundos, mas a media e 16 — sinal de que ha dois grupos "
    "misturados."
)

faixas = corte["faixas"].copy()
faixas["faixa"] = [
    f"{a:.0f} a {b:.0f} s" if b != float("inf") else f"mais de {a:.0f} s"
    for a, b in zip(faixas["de_s"], faixas["ate_s"])
]

st.altair_chart(
    alt.Chart(faixas)
    .mark_bar()
    .encode(
        x=alt.X("faixa:N", sort=list(faixas["faixa"]), title="intervalo entre leituras",
                axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("intervalos:Q", title="quantas vezes", scale=alt.Scale(type="symlog")),
        color=alt.Color(
            "vazia:N", title=None,
            scale=alt.Scale(domain=[False, True], range=["#4c78a8", "#d1495b"]),
            legend=alt.Legend(labelExpr="datum.label == 'true' ? 'faixa vazia' : 'com dados'"),
        ),
        tooltip=["faixa", "intervalos"],
    )
    .properties(height=280),
    width="stretch",
)

if vazio.get("centro_s"):
    st.success(
        f"""
**Os dois grupos nao se tocam.**

- Coleta continua: ate **{vazio['maior_continuo_s']:.0f} segundos**
- Pausas de verdade: a partir de **{vazio['menor_pausa_s']:.0f} segundos**
- Entre os dois: **nenhuma ocorrencia**

Sao **{vazio['n_pausas']} pausas** escondidas dentro dos {corte['eventos_atuais']}
eventos atuais. Qualquer corte entre {vazio['maior_continuo_s']:.0f} e
{vazio['menor_pausa_s']:.0f} segundos pegaria exatamente essas — nem uma a mais,
nem uma a menos.
"""
    )

st.markdown("#### Deixando a estatistica escolher o numero")

criterios, saltos = D.r_criterios_limiar()

st.markdown(
    """
Ate aqui o limiar veio de olhar o grafico. Isso e fragil — outra pessoa olharia e
escolheria outro numero.

Abaixo, cinco criterios que calculam o limiar **sozinhos**, sem ninguem escolher.
Estao todos aqui, inclusive os que **nao funcionam** — mostrar so o que confirma a
conclusao seria escolher a dedo.
"""
)

tabela_crit = criterios.copy()
tabela_crit["eventos"] = tabela_crit["cortes_que_faria"] + corte["eventos_atuais"]

st.dataframe(
    tabela_crit[["criterio", "valor_s", "cortes_que_faria", "eventos", "observacao"]],
    hide_index=True,
    column_config={
        "criterio": st.column_config.TextColumn("criterio", width="medium"),
        "valor_s": st.column_config.NumberColumn("limiar que produz", format="%.2f s"),
        "cortes_que_faria": st.column_config.NumberColumn("cortes", format="%d"),
        "eventos": st.column_config.NumberColumn("eventos resultantes", format="%d"),
        "observacao": st.column_config.TextColumn("por que", width="large"),
    },
)

st.warning(
    """
**Tres dos cinco criterios desabam — e pelo mesmo motivo.**

Tukey e MAD medem "o quanto os valores se espalham". Mas aqui a esmagadora maioria
das leituras tem **exatamente o mesmo intervalo de 2 segundos**. O espalhamento e
praticamente nulo: o IQR vale 0,0003 s e o MAD, 0,0000 s.

Com espalhamento quase zero, os tres devolvem ~2 segundos. Repare na coluna
*cortes*: fariam mais de 10 mil cortes, partindo a cadencia normal em pedacos.

Nao e defeito dos criterios. Eles pressupoem uma distribuicao que se espalha, e
esta nao se espalha: sao dois blocos rigidos com um vazio no meio.
"""
)

st.caption(
    "O percentil 99,8 nao desaba, mas erra por outro lado: corta em 22,5 s e "
    "**deixa passar 31 pausas reais** (334 cortes em vez de 365). Ele cai depois "
    "do vazio, ja dentro do grupo das pausas."
)

if not saltos.empty:
    st.markdown("**O criterio que funciona: a maior descontinuidade**")
    st.markdown(
        """
Em vez de medir espalhamento, este procura o **maior salto** entre dois valores
consecutivos da lista ordenada. Onde a distribuicao mais se rompe, ali esta a
fronteira.
"""
    )
    st.dataframe(
        saltos,
        hide_index=True,
        column_config={
            "de_s": st.column_config.NumberColumn("de", format="%.3f s"),
            "para_s": st.column_config.NumberColumn("para", format="%.3f s"),
            "salto": st.column_config.NumberColumn("salto", format="%.1f x"),
            "ponto_medio_s": st.column_config.NumberColumn("meio do salto", format="%.2f s"),
        },
    )
    st.success(
        f"""
**O maior salto de toda a distribuicao e de {saltos.iloc[0]['de_s']:.0f} s para
{saltos.iloc[0]['para_s']:.1f} s — um pulo de {saltos.iloc[0]['salto']:.1f} vezes.**

Ele e maior que os saltos entre pausas de horas e de dias, que aparecem logo
abaixo na tabela. Ou seja: a separacao entre "gravando" e "parado" e a
descontinuidade mais forte que existe nestes dados.

O meio desse salto e **{saltos.iloc[0]['ponto_medio_s']:.1f} segundos** — que e
exatamente o valor que escolhemos olhando o grafico, agora obtido sem olhar nada.
"""
    )

st.markdown("**Simulacao: o que cada corte faria com os eventos de hoje**")

sim = corte["simulacao"]
st.dataframe(
    sim[["corte_s", "eventos", "eventos_partidos", "pct_eventos_partidos"]],
    hide_index=True,
    column_config={
        "corte_s": st.column_config.NumberColumn("corte (segundos)", format="%.1f s"),
        "eventos": st.column_config.NumberColumn("eventos resultantes", format="%d"),
        "eventos_partidos": st.column_config.NumberColumn(
            "eventos que se partiriam", format="%d",
            help=f"Dos {corte['eventos_atuais']} eventos atuais."
        ),
        "pct_eventos_partidos": st.column_config.NumberColumn(
            "% dos atuais", format="%.0f%%"
        ),
    },
)

st.altair_chart(
    alt.Chart(sim)
    .mark_line(point=True, strokeWidth=2, color="#d1495b")
    .encode(
        x=alt.X("corte_s:Q", title="corte usado (segundos)", scale=alt.Scale(type="log")),
        y=alt.Y("eventos:Q", title="eventos resultantes", scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("corte_s:Q", title="corte (s)"), "eventos",
                 "eventos_partidos"],
    )
    .properties(height=280),
    width="stretch",
)

st.info(
    f"""
### Como ler

**Corte de 2,5 ou 5 segundos → 11 mil eventos.** Errado. Parte das leituras vem a
cada 5,3 segundos, e um corte abaixo disso parte cada medicao em centenas de pedacos.

**Corte de 8, 10 ou 15 segundos → 570 eventos, sempre o mesmo.** Sao os
{corte['eventos_atuais']} de hoje mais as {vazio.get('n_pausas', 0)} pausas internas.
Tres valores diferentes, um unico resultado — porque todos caem na faixa vazia.

**Corte de 60 segundos → 366 eventos.** Ja deixa passar mais da metade das pausas.

### Se a decisao mudar

O numero a usar seria **10 segundos**, o centro da faixa vazia. E o que esta
documentado em `config.GAP_NOVO_EPISODIO_S`, pronto para quando quiser ligar.
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
