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

    st.header("🧩 Eventos — teste de duas bases", divider="gray")

    st.markdown(
        """
    Um **evento** e uma vez em que a maquina foi medida com o mesmo defeito, na mesma
    rotacao. Agrupar as 166.796 linhas em eventos e o que permite responder *"quantas
    vezes isso aconteceu?"*.

    Ha duas ordens possiveis para montar esses eventos. Aqui elas rodam lado a lado,
    sobre o mesmo arquivo.
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
    # As duas bases, lado a lado
    # ==========================================================================
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

    # ==========================================================================
    # Linha do tempo das duas bases
    # ==========================================================================
    st.header("Os eventos na linha do tempo")

    bases_visiveis = st.multiselect(
        "Bases no grafico",
        [NOME_A, NOME_B],
        default=[NOME_A, NOME_B],
        help="Desmarque uma para ver a outra sozinha.",
        key="ev_bases",
    )

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

    for nome, versao in ((NOME_A, "A"), (NOME_B, "B")):
        if nome not in bases_visiveis:
            continue

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

    # --------------------------------------------------------------------------
    # Quando cada evento aconteceu
    # --------------------------------------------------------------------------
    st.subheader("Quando cada evento aconteceu")

    st.markdown(
        "A mesma informacao resumida: cada barra e um evento, do inicio ao fim. "
        "Onde a barra vermelha e longa e a verde e curta, as duas bases discordam."
    )

    partes = []
    if NOME_A in bases_visiveis and len(eventos_a):
        partes.append(eventos_a.assign(base=NOME_A))
    if NOME_B in bases_visiveis and len(eventos_b):
        partes.append(eventos_b.assign(base=NOME_B))

    if not partes:
        st.info("Selecione ao menos uma base para ver o grafico.")
    else:
        linha_tempo = pd.concat(partes, ignore_index=True)

        # Com muitas falhas na tela, o eixo Y vira uma lista ilegivel. Acima de 30
        # agrupamos por familia; abaixo disso mostramos cada falha.
        n_falhas = linha_tempo["fault"].nunique()
        eixo_y = "fault" if n_falhas <= 30 else "familia"
        titulo_y = "falha" if eixo_y == "fault" else "familia"

        if eixo_y == "familia":
            st.caption(
                f"{n_falhas} falhas selecionadas — o eixo agrupa por familia para "
                "continuar legivel. Escolha menos familias para ver falha a falha."
            )

        escala = alt.Scale(domain=[NOME_A, NOME_B], range=[COR_A, COR_B])
        n_linhas = linha_tempo[eixo_y].nunique()

        st.altair_chart(
            alt.Chart(linha_tempo)
            .mark_bar(height=9, cornerRadius=2)
            .encode(
                x=alt.X("inicio:T", title="data e hora"),
                x2="fim:T",
                y=alt.Y(f"{eixo_y}:N", title=titulo_y),
                yOffset=alt.YOffset("base:N"),
                color=alt.Color("base:N", title=None, scale=escala,
                                legend=alt.Legend(orient="top")),
                tooltip=[
                    alt.Tooltip("base:N", title="base"),
                    alt.Tooltip("fault:N", title="falha"),
                    alt.Tooltip("inicio:T", title="comecou", format="%d/%m/%Y %H:%M"),
                    alt.Tooltip("fim:T", title="terminou", format="%d/%m/%Y %H:%M"),
                    alt.Tooltip("duracao_min:Q", title="duracao (min)", format=".0f"),
                    alt.Tooltip("n_leituras:Q", title="leituras"),
                ],
            )
            .properties(height=max(220, 26 * n_linhas))
            .interactive(),
            width="stretch",
        )

        st.caption(
            "Use a roda do mouse para dar zoom no tempo. Eventos muito curtos "
            "aparecem como riscos finos."
        )

    st.divider()

    # ==========================================================================
    # O resultado do teste
    # ==========================================================================
    st.header("As duas dao o mesmo resultado?")

    if len(eventos_a) == len(eventos_b) and comp["resultado_igual"]:
        st.success("Sim. As duas produzem exatamente os mesmos eventos.")
    elif len(eventos_a) == len(eventos_b):
        st.warning(
            f"Nesta selecao as duas dao o mesmo numero de eventos ({len(eventos_a)}), "
            "mas no arquivo inteiro elas divergem — "
            f"{len(eventos_a_todos)} contra {len(eventos_b_todos)}."
        )
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

        if len(eventos_b):
            pior = eventos_b.loc[eventos_b["duracao_s"].idxmax()]
            st.warning(
                f"""
    **Exemplo concreto.** Na base 🅑, a falha `{pior['fault']}` vira **um evento**
    que comeca em {pior['inicio']:%d/%m} e termina em {pior['fim']:%d/%m} —
    **{pior['duracao_s'] / 86400:.1f} dias** de "duracao".

    A maquina nao ficou todo esse tempo sendo medida sem parar.
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

    # Calculado sobre o recorte escolhido, nao sobre o arquivo inteiro: assim o
    # numero acompanha a familia que esta na tela.
    disp_a = float(eventos_a["dispersao"].median()) if len(eventos_a) else float("nan")
    disp_b = float(eventos_b["dispersao"].median()) if len(eventos_b) else float("nan")

    s1, s2 = st.columns(2)
    s1.metric(
        NOME_A,
        f"{disp_a:.2f}" if pd.notna(disp_a) else "—",
        help="Dispersao tipica dentro dos eventos. Menor e melhor.",
    )
    s2.metric(
        NOME_B,
        f"{disp_b:.2f}" if pd.notna(disp_b) else "—",
        f"{(disp_b / disp_a - 1) * 100:+.0f}% pior" if pd.notna(disp_a) and disp_a else None,
        delta_color="inverse",
    )

    comparativo = pd.concat(
        [
            eventos_a[["dispersao"]].assign(base=NOME_A),
            eventos_b[["dispersao"]].assign(base=NOME_B),
        ]
    )

    if not comparativo.empty:
        st.altair_chart(
            alt.Chart(comparativo)
            .mark_boxplot(extent="min-max", size=40)
            .encode(
                x=alt.X("dispersao:Q",
                        title="dispersao dentro do evento (menor = melhor)",
                        scale=alt.Scale(type="symlog")),
                y=alt.Y("base:N", title=None),
                color=alt.Color("base:N", legend=None,
                                scale=alt.Scale(domain=[NOME_A, NOME_B],
                                                range=[COR_A, COR_B])),
            )
            .properties(height=180),
            width="stretch",
        )

    if pd.notna(disp_a) and pd.notna(disp_b) and disp_a:
        diferenca = (disp_b / disp_a - 1) * 100
        if diferenca > 1:
            st.info(
                f"""
    A base 🅑 tem dispersao **{diferenca:.0f}% maior**.

    Faz sentido: ao juntar medicoes feitas com semanas de diferenca, ela mistura
    leituras que nao se parecem — a maquina nao estava no mesmo estado.
    """
            )
        else:
            st.info(
                "Nesta selecao as duas bases tem dispersao parecida — sao familias em "
                "que cada falha foi medida uma vez so, entao nao ha o que juntar."
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
