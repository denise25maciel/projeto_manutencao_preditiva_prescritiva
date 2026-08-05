"""Teste das duas ordens de operacao para montar eventos.

A pagina e um experimento com duas bases: a esquerda ordena por data antes de
separar por falha; a direita faz o contrario. Tudo o que nao serve para comparar
as duas foi para os expansores do fim.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Eventos", "🧩")

st.title("🧩 Eventos — teste de duas bases")

st.markdown(
    """
Um **evento** e uma vez em que a maquina foi medida com o mesmo defeito.
Agrupar as 166.796 linhas em eventos e o que permite responder *"quantas vezes
isso aconteceu?"*.

Ha duas ordens possiveis para montar esses eventos. Aqui elas rodam lado a lado,
sobre o mesmo arquivo.
"""
)

try:
    comp = D.r_comparar_abordagens()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

eventos_a = comp["eventos_a"]
eventos_b = comp["eventos_b"]
resumo = comp["resumo"]
linha_a = resumo.iloc[0]
linha_b = resumo.iloc[1]

st.divider()

# ==========================================================================
# As duas bases, lado a lado
# ==========================================================================
esquerda, direita = st.columns(2, gap="large")

with esquerda:
    st.subheader("🅐 Data → Falha")
    st.caption(
        "Ordena o arquivo inteiro por data. Depois percorre de cima a baixo e "
        "comeca um evento novo quando o nome da falha muda."
    )
    st.metric("Eventos", len(eventos_a))
    a1, a2 = st.columns(2)
    a1.metric("Maior duracao", f"{linha_a['duracao_maxima_h']:.0f} h")
    a2.metric("Duracao tipica", f"{linha_a['duracao_mediana_min']:.0f} min")

    st.dataframe(
        eventos_a[["evento", "fault", "n_leituras", "inicio", "duracao_min"]],
        hide_index=True,
        height=380,
        column_config={
            "evento": st.column_config.NumberColumn("n", format="%d"),
            "fault": "falha",
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
            "duracao_min": st.column_config.NumberColumn("min", format="%.0f"),
        },
    )
    st.download_button(
        "Baixar base 🅐",
        eventos_a.to_csv(index=False).encode("utf-8"),
        file_name="eventos_A_data_depois_falha.csv",
        mime="text/csv",
        key="dl_a",
    )

with direita:
    st.subheader("🅑 Falha → Data")
    st.caption(
        "Separa as leituras por falha. Depois ordena cada grupo por data. "
        "Como o nome nao muda dentro do grupo, cada falha vira um evento."
    )
    st.metric("Eventos", len(eventos_b))
    b1, b2 = st.columns(2)
    b1.metric("Maior duracao", f"{linha_b['duracao_maxima_h']:.0f} h")
    b2.metric("Duracao tipica", f"{linha_b['duracao_mediana_min']:.0f} min")

    st.dataframe(
        eventos_b[["evento", "fault", "n_leituras", "inicio", "duracao_min"]],
        hide_index=True,
        height=380,
        column_config={
            "evento": st.column_config.NumberColumn("n", format="%d"),
            "fault": "falha",
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
            "duracao_min": st.column_config.NumberColumn("min", format="%.0f"),
        },
    )
    st.download_button(
        "Baixar base 🅑",
        eventos_b.to_csv(index=False).encode("utf-8"),
        file_name="eventos_B_falha_depois_data.csv",
        mime="text/csv",
        key="dl_b",
    )

st.divider()

# ==========================================================================
# O resultado do teste
# ==========================================================================
st.header("As duas dao o mesmo resultado?")

if comp["resultado_igual"]:
    st.success("Sim. As duas produzem exatamente os mesmos eventos.")
else:
    st.error(
        f"""
**Nao.** {len(eventos_a)} eventos contra {len(eventos_b)}.

A diferenca aparece quando a **mesma falha foi medida em dias diferentes**:

- Na 🅐, entre as duas medicoes ha leituras de outras falhas. Elas cortam, e
  saem **dois** eventos.
- Na 🅑, as duas medicoes estao no mesmo grupo e nada as separa. Sai **um** evento
  so, cobrindo o periodo inteiro.
"""
    )

    pior = eventos_b.loc[eventos_b["duracao_s"].idxmax()]
    st.warning(
        f"""
**Exemplo concreto.** Na base 🅑, a falha `{pior['fault']}` vira **um evento**
que comeca em {pior['inicio']:%d/%m} e termina em {pior['fim']:%d/%m} —
**{pior['duracao_s'] / 86400:.0f} dias** de "duracao".

A maquina nao ficou {pior['duracao_s'] / 86400:.0f} dias sendo medida sem parar.
"""
    )

st.divider()

# ==========================================================================
# Similaridade dentro de cada evento
# ==========================================================================
st.header("As leituras de cada evento se parecem entre si?")

st.markdown(
    """
Um agrupamento so vale se o que ele junta for parecido. Se um evento reune
leituras muito diferentes, ele juntou o que nao deveria.

Medimos assim: todas as medidas de vibracao sao colocadas na mesma escala, e para
cada evento calculamos **o quanto suas leituras se afastam do proprio centro**.

- **Numero baixo** → leituras parecidas → agrupou bem
- **Numero alto** → leituras diferentes no mesmo evento → agrupou mal
"""
)

s1, s2 = st.columns(2)
s1.metric(
    "🅐 Data → Falha",
    f"{linha_a['dispersao_mediana']:.2f}",
    help="Dispersao tipica dentro dos eventos. Menor e melhor.",
)
s2.metric(
    "🅑 Falha → Data",
    f"{linha_b['dispersao_mediana']:.2f}",
    f"{(linha_b['dispersao_mediana'] / linha_a['dispersao_mediana'] - 1) * 100:+.0f}% pior",
    delta_color="inverse",
)

comparativo = pd.concat(
    [
        comp["coesao_a"][["dispersao"]].assign(base="🅐 Data → Falha"),
        comp["coesao_b"][["dispersao"]].assign(base="🅑 Falha → Data"),
    ]
)

st.altair_chart(
    alt.Chart(comparativo)
    .mark_boxplot(extent="min-max", size=40)
    .encode(
        x=alt.X("dispersao:Q", title="dispersao dentro do evento (menor = melhor)",
                scale=alt.Scale(type="symlog")),
        y=alt.Y("base:N", title=None),
        color=alt.Color(
            "base:N", legend=None,
            scale=alt.Scale(domain=["🅐 Data → Falha", "🅑 Falha → Data"],
                            range=["#2d6a4f", "#d1495b"]),
        ),
    )
    .properties(height=180),
    width="stretch",
)

st.info(
    f"""
A base 🅑 tem dispersao **{(linha_b['dispersao_mediana'] / linha_a['dispersao_mediana'] - 1) * 100:.0f}% maior**.

Faz sentido: ao juntar medicoes feitas com semanas de diferenca, ela mistura
leituras que nao se parecem — a maquina nao estava no mesmo estado.
"""
)

with st.expander("Os eventos com leituras mais diferentes entre si"):
    st.caption(
        "Valem atencao independentemente da base escolhida: sao eventos em que a "
        "vibracao variou muito durante a propria medicao."
    )
    st.dataframe(
        eventos_a.nlargest(15, "dispersao")[
            ["evento", "fault", "n_leituras", "inicio", "duracao_min", "dispersao"]
        ],
        hide_index=True,
        column_config={
            "evento": st.column_config.NumberColumn("n", format="%d"),
            "fault": "falha",
            "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
            "duracao_min": st.column_config.NumberColumn("duracao (min)", format="%.0f"),
            "dispersao": st.column_config.NumberColumn("dispersao", format="%.1f"),
        },
    )
    st.caption(
        "Repare no padrao: os tres primeiros terminam em `_pos_2` — o sensor "
        "estava em outra posicao. Nao e defeito do agrupamento, e a montagem que "
        "mudou durante a campanha."
    )

st.divider()

st.success(
    """
### Conclusao

**Usamos a base 🅐 (data → falha).** Ela conta ocorrencias; a 🅑 conta periodos.

Para responder *"quantas vezes esse defeito ja apareceu"*, so a 🅐 serve — e a
medida de similaridade confirma que ela tambem agrupa leituras mais parecidas.
"""
)

# ==========================================================================
# Detalhes, fora do caminho principal
# ==========================================================================
st.divider()
st.caption("Verificacoes e diagnosticos. Abra se quiser conferir.")

with st.expander("O arquivo nao vem ordenado por data"):
    ordenacao = D.r_ordenacao()
    st.markdown(
        f"""
O `banner.csv` chega fora de ordem cronologica: **{ordenacao['pct_fora_do_lugar']:.0f}%
das linhas** ({ordenacao['linhas_fora_do_lugar']:,}) mudam de lugar ao ordenar. A
coluna `id` tambem esta fora de ordem.

Por isso as duas abordagens ordenam por data — a diferenca entre elas e **quando**
isso acontece, nao **se** acontece.
""".replace(",", ".")
    )
    exemplo = D.r_exemplo_desordem()
    if not exemplo.empty:
        st.caption("Duas linhas vizinhas no arquivo. A de baixo e de um mes antes:")
        st.dataframe(
            exemplo,
            hide_index=True,
            column_config={
                "posicao_no_arquivo": st.column_config.NumberColumn("linha", format="%d"),
                "created_at": st.column_config.DatetimeColumn(
                    "data e hora", format="DD/MM/YYYY HH:mm:ss"
                ),
                "fault": "falha",
            },
        )

with st.expander("Verificacoes do agrupamento"):
    validacao = D.r_validacao_eventos()
    if bool(validacao["passou"].all()):
        st.success(f"As {len(validacao)} verificacoes passaram.")
    else:
        st.error("Alguma verificacao falhou.")
    st.dataframe(
        validacao,
        hide_index=True,
        column_config={
            "checagem": st.column_config.TextColumn("verificacao", width="large"),
            "passou": st.column_config.CheckboxColumn("ok?"),
            "detalhe": st.column_config.TextColumn("detalhe", width="medium"),
        },
    )

with st.expander("Em quais falhas as duas bases discordam"):
    st.dataframe(
        comp["por_rotulo"][comp["por_rotulo"]["diferenca"] != 0],
        hide_index=True,
        height=320,
        column_config={
            "fault": "falha",
            "eventos_A": st.column_config.NumberColumn("🅐", format="%d"),
            "eventos_B": st.column_config.NumberColumn("🅑", format="%d"),
            "diferenca": st.column_config.NumberColumn("diferenca", format="%d"),
        },
    )

with st.expander("Quantas vezes cada falha aconteceu (base 🅐)"):
    st.dataframe(
        D.r_resumo_eventos()[
            ["fault", "eventos", "leituras", "duracao_mediana_min", "primeira", "ultima"]
        ],
        hide_index=True,
        height=400,
        column_config={
            "fault": "falha",
            "eventos": st.column_config.NumberColumn("ocorrencias", format="%d"),
            "leituras": st.column_config.NumberColumn("leituras", format="%d"),
            "duracao_mediana_min": st.column_config.NumberColumn(
                "duracao tipica (min)", format="%.0f"
            ),
            "primeira": st.column_config.DatetimeColumn("primeira", format="DD/MM/YY"),
            "ultima": st.column_config.DatetimeColumn("ultima", format="DD/MM/YY"),
        },
    )

zero = eventos_a[eventos_a["duracao_s"] == 0]
if len(zero):
    with st.expander("Um evento com duracao zero (pendencia conhecida)"):
        st.markdown(
            f"""
Um evento de `{zero.iloc[0]['fault']}` tem
{int(zero.iloc[0]['n_leituras']):,} leituras e **duracao zero**: todas receberam a
mesma data e hora no arquivo original.

Mil medicoes num unico instante e impossivel. O erro esta no dado, nao no
agrupamento. Fica marcado, nao corrigido — detalhes na tela *Qualidade dos Dados*.
""".replace(",", ".")
        )
