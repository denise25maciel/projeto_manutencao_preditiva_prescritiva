"""Ato 5 da narrativa — **de leituras para ocorrencias**.

Um evento e uma vez em que a maquina foi medida com o mesmo defeito, na mesma
rotacao. Agrupar as 166.796 linhas em eventos e o que permite responder *"quantas
vezes isso aconteceu?"* — contar linhas daria "4.200 ocorrencias" para uma unica
sessao de bancada.

A secao e um experimento com duas bases: a esquerda ordena por data antes de
separar por falha; a direita faz o contrario.

Era uma pagina propria (`pages/4_Eventos.py`). Virou secao de `app.py` pelo mesmo
motivo das outras. O conteudo e o mesmo — a conversao foi mecanica.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D


def render() -> None:
    """Desenha esta secao dentro da narrativa unica."""

    st.header("🧩 Eventos — de leituras para ocorrencias", divider="gray")

    st.markdown(
        """
    Um **evento** e uma vez em que a maquina foi medida com o mesmo defeito, na mesma
    rotacao. Agrupar as 166.796 linhas em eventos e o que permite responder *"quantas
    vezes isso aconteceu?"*.

    Ha duas ordens possiveis para monta-los, e elas nao dao o mesmo resultado. As
    duas rodam aqui, mas a comparacao fica **recolhida** — ela sustenta a escolha,
    nao e o que se veio ver.
    """
    )

    with st.expander("Por que a rotacao encerra um evento"):
        st.markdown(
            """
    No comeco a regra quebrava so na troca de falha. O resultado eram **136 dos 205
    eventos misturando rotacoes** — 95% das leituras.

    A bancada rodava 500, 1000 e 2000 rpm em sequencia sem trocar o nome da falha,
    entao tres ensaios viravam um evento so.

    O caso extremo era o evento de `rolamento_combination_pos_2`:

    | rotacao | leituras | velocidade RMS |
    |---|---|---|
    | 500 rpm | 50 | 3,5 |
    | 1.000 rpm | 50 | 5,4 |
    | 2.000 rpm | 50 | **21,1** |

    Seis vezes maior dentro do "mesmo" evento. A mediana dele nao descrevia nenhum
    dos tres regimes.

    Incluindo a rotacao na regra: **205 → 526 eventos**, dispersao interna tipica de
    **2,40 → 1,31**, e zero eventos com regime misturado.
    """
        )

    try:
        comp = D.r_comparar_abordagens()
    except FileNotFoundError as e:
        D.aviso_csv_ausente(e)

    eventos_a_todos = comp["eventos_a"]
    eventos_b_todos = comp["eventos_b"]
    resumo = comp["resumo"]
    linha_a = resumo.iloc[0]
    linha_b = resumo.iloc[1]

    NOME_A = "🅐 Data → Falha"
    NOME_B = "🅑 Falha → Data"
    COR_A = "#2d6a4f"
    COR_B = "#d1495b"

    st.divider()

    # ==========================================================================
    # Filtro por familia — vale para tudo o que vem abaixo
    # ==========================================================================
    familias = sorted(
        set(eventos_a_todos["familia"].dropna()) | set(eventos_b_todos["familia"].dropna())
    )

    escolhidas = st.multiselect(
        "Familia de falhas",
        familias,
        default=familias,
        help="Filtra as duas bases e o grafico. Deixe vazio para ver tudo.",
        key="ev_familias",
    )
    filtro = escolhidas or familias

    eventos_a = eventos_a_todos[eventos_a_todos["familia"].isin(filtro)]
    eventos_b = eventos_b_todos[eventos_b_todos["familia"].isin(filtro)]

    if len(escolhidas) < len(familias):
        st.caption(
            f"Mostrando {len(escolhidas)} de {len(familias)} familias: "
            f"{len(eventos_a)} eventos na base 🅐 e {len(eventos_b)} na 🅑."
        )

    # ==========================================================================
    # As duas bases, lado a lado — recolhido
    # ==========================================================================
    #
    # Comeca fechado porque e **metodologia, nao resultado**. Quem abre a tela
    # quer saber quantas vezes cada falha aconteceu; qual das duas ordens de
    # operacao monta os eventos e a pergunta de quem for auditar a decisao.
    #
    # Continua aqui, e nao apagado, porque a comparacao e o que sustenta a
    # escolha — mas ocupava a primeira tela inteira do ato com duas tabelas de
    # 380 px antes de qualquer dado medido aparecer.
    with st.expander(
        "🔬 O teste das duas bases — como os eventos foram montados", expanded=False
    ):
        st.caption(
            "Ha duas ordens possiveis de operacao, e elas nao dao o mesmo "
            "resultado. As duas rodam aqui sobre o mesmo arquivo, para a escolha "
            "poder ser conferida em vez de aceita."
        )

        esquerda, direita = st.columns(2, gap="large")

        with esquerda:
            st.markdown(
                f"<h3 style='color:{COR_A};margin-bottom:0'>{NOME_A}</h3>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Ordena o arquivo inteiro por data. Depois percorre de cima a baixo e "
                "comeca um evento novo quando muda a falha ou a rotacao."
            )
            st.metric("Eventos", len(eventos_a))
            a1, a2 = st.columns(2)
            a1.metric(
                "Maior duracao",
                f"{eventos_a['duracao_s'].max() / 3600:.0f} h" if len(eventos_a) else "—",
            )
            a2.metric(
                "Duracao tipica",
                f"{eventos_a['duracao_min'].median():.0f} min" if len(eventos_a) else "—",
            )

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
            st.markdown(
                f"<h3 style='color:{COR_B};margin-bottom:0'>{NOME_B}</h3>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Separa as leituras por falha. Depois ordena cada grupo por data. "
                "Como nada muda dentro do grupo, cada combinacao vira um evento."
            )
            st.metric("Eventos", len(eventos_b))
            b1, b2 = st.columns(2)
            b1.metric(
                "Maior duracao",
                f"{eventos_b['duracao_s'].max() / 3600:.0f} h" if len(eventos_b) else "—",
            )
            b2.metric(
                "Duracao tipica",
                f"{eventos_b['duracao_min'].median():.0f} min" if len(eventos_b) else "—",
            )

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


    # --------------------------------------------------------------------------
    # A serie medida, com a divisao em eventos por cima
    # --------------------------------------------------------------------------
    st.subheader("A medida ao longo do tempo, dividida em eventos")

    st.markdown(
        """
    Aqui esta o **dado medido**, nao so quando o evento aconteceu. Cada ponto e uma
    leitura do sensor; a linha muda de tom a cada evento novo.

    E onde a diferenca entre as duas bases fica evidente: a mesma curva, cortada de
    formas diferentes.
    """
    )

    numericas = D.r_numericas()
    padrao_col = (
        "z_rms_velocity_mm_s" if "z_rms_velocity_mm_s" in numericas else numericas[0]
    )
    coluna_medida = st.selectbox(
        "Medida", numericas, index=numericas.index(padrao_col),
        key="ev_medida",
    )

    TONS = {
        NOME_A: ["#2d6a4f", "#95d5b2"],   # verde escuro / verde claro
        NOME_B: ["#d1495b", "#f4a3ac"],   # vermelho escuro / vermelho claro
    }
    chave_familias = tuple(sorted(filtro))

    # As duas bases, sempre. Havia um seletor "Bases no grafico" que permitia
    # esconder uma delas; sem ele, mostrar as duas e o comportamento certo —
    # a secao inteira existe para compara-las, e uma sozinha nao compara nada.
    for nome, versao in ((NOME_A, "A"), (NOME_B, "B")):
        serie = D.r_serie_com_eventos(versao, coluna_medida, chave_familias)
        if serie.empty:
            continue

        n_eventos_serie = serie["evento"].nunique()
        st.markdown(f"**{nome}** — {n_eventos_serie} eventos nesta selecao")

        st.altair_chart(
            alt.Chart(serie)
            .mark_line(strokeWidth=1.6)
            .encode(
                x=alt.X("created_at:T", title="data e hora da coleta (UTC)"),
                y=alt.Y("valor:Q", title=coluna_medida, scale=alt.Scale(zero=False)),
                # `detail` quebra a linha entre eventos; `color` alterna o tom para
                # a fronteira ficar visivel mesmo quando os eventos sao colados.
                detail=alt.Detail("evento:N"),
                color=alt.Color(
                    "paridade:N",
                    legend=None,
                    scale=alt.Scale(domain=[0, 1], range=TONS[nome]),
                ),
                tooltip=[
                    alt.Tooltip("fault:N", title="falha"),
                    alt.Tooltip("evento:Q", title="evento n"),
                    alt.Tooltip("created_at:T", title="quando",
                                format="%d/%m/%Y %H:%M:%S"),
                    alt.Tooltip("valor:Q", title=coluna_medida, format=".4f"),
                ],
            )
            .properties(height=280)
            .interactive(),
            width="stretch",
        )

    st.caption(
        "Os tons alternam a cada evento — onde a cor muda, um evento terminou e outro "
        "comecou. A linha tambem se interrompe entre eventos. Zoom com a roda do mouse."
    )

    st.divider()

    # --------------------------------------------------------------------------
    # Um evento por grafico, para comparar formas
    # --------------------------------------------------------------------------
    st.subheader("Um evento abaixo do outro, com todas as medidas")

    st.markdown(
        """
    Aqui cada evento ganha o proprio grafico, com **todas as medidas juntas**. Serve
    para responder olhando: *estes eventos se parecem?*

    Duas coisas tornam a comparacao possivel:

    - **Todas as medidas na mesma escala.** Cada uma vira "quantos desvios acima ou
      abaixo da media do arquivo". Sem isso, `rpm` (0 a 3000) e `z_kurtosis` (2 a 65)
      nao caberiam no mesmo eixo.
    - **Todos os eventos comecam no zero.** O eixo horizontal e o tempo decorrido
      desde o inicio de cada evento, nao a data. Em data absoluta, cada um apareceria
      num canto da tela e as formas nao poderiam ser comparadas. A data real de inicio
      esta no titulo de cada grafico.

    **Eventos parecidos tem desenhos parecidos.**
    """
    )

    versao_perfil = st.radio(
        "Base",
        [NOME_A, NOME_B],
        horizontal=True,
        key="ev_versao_perfil",
    )
    eventos_da_base = eventos_a if versao_perfil == NOME_A else eventos_b
    letra = "A" if versao_perfil == NOME_A else "B"

    if eventos_da_base.empty:
        st.info("Nenhum evento nesta selecao.")
    else:
        rotulos_evento = {
            int(r["evento"]): (
                f"{int(r['evento'])} · {r['fault']} · {r['inicio']:%d/%m %H:%M} · "
                f"{int(r['n_leituras'])} leituras"
            )
            for _, r in eventos_da_base.iterrows()
        }
        opcoes_evento = list(rotulos_evento)

        escolhidos = st.multiselect(
            "Eventos para comparar",
            opcoes_evento,
            default=opcoes_evento[:4],
            max_selections=8,
            format_func=lambda e: rotulos_evento[e],
            help="Ate 8. Cada um vira um grafico, um abaixo do outro.",
            key="ev_rotulos_perfil",
        )

        medidas = st.multiselect(
            "Medidas",
            numericas,
            default=[c for c in D.config.COLUNAS_ASSINATURA if c in numericas],
            help="Por padrao, as medidas que compoem a assinatura de vibracao.",
            key="ev_medidas_perfil",
        )

        if not escolhidos:
            st.info("Escolha ao menos um evento.")
        elif not medidas:
            st.info("Escolha ao menos uma medida.")
        else:
            # Orcamento por grafico: n_medidas x pontos precisa ficar abaixo do
            # limite de 5000 linhas do Vega.
            pontos = max(40, min(200, 4000 // max(len(medidas), 1)))
            perfis = D.r_series_por_evento(
                letra, tuple(escolhidos), tuple(medidas), pontos
            )

            if perfis.empty:
                st.info("Sem dados para estes eventos.")
            else:
                # Escala de Y compartilhada: sem isso, cada grafico se ajustaria ao
                # proprio maximo e formas diferentes pareceriam iguais.
                lim = float(perfis["valor"].abs().quantile(0.995))
                dominio = [-lim, lim]

                for ev in escolhidos:
                    d = perfis[perfis["evento"] == ev]
                    if d.empty:
                        continue
                    info_ev = eventos_da_base[eventos_da_base["evento"] == ev].iloc[0]

                    st.markdown(
                        f"**Evento {ev} · `{info_ev['fault']}`** — "
                        f"{info_ev['inicio']:%d/%m/%Y %H:%M} · "
                        f"{int(info_ev['n_leituras'])} leituras · "
                        f"{info_ev['duracao_min']:.0f} min · "
                        f"dispersao {info_ev['dispersao']:.2f}"
                    )

                    st.altair_chart(
                        alt.Chart(d)
                        .mark_line(strokeWidth=1.2, opacity=0.85)
                        .encode(
                            x=alt.X("minuto:Q", title="minutos desde o inicio do evento"),
                            y=alt.Y(
                                "valor:Q",
                                title="desvios da media do arquivo",
                                scale=alt.Scale(domain=dominio, clamp=True),
                            ),
                            color=alt.Color("coluna:N", title="medida",
                                            legend=alt.Legend(orient="right", columns=1)),
                            tooltip=[
                                alt.Tooltip("coluna:N", title="medida"),
                                alt.Tooltip("minuto:Q", title="minuto", format=".1f"),
                                alt.Tooltip("valor:Q", title="desvios", format=".2f"),
                            ],
                        )
                        .properties(height=240),
                        width="stretch",
                    )

                st.caption(
                    f"Eixo Y igual em todos os graficos (de {dominio[0]:.1f} a "
                    f"{dominio[1]:.1f} desvios), senao cada um se ajustaria ao proprio "
                    "maximo e formas diferentes pareceriam iguais. A linha em zero e a "
                    "media do arquivo inteiro."
                )

    st.divider()
    