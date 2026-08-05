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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nomes diferentes", n_total)
    c2.metric("Sao defeitos", n_problema)
    c3.metric("Sao so estados", n_total - n_problema)
    c4.metric("Grupos (familias)", n_familias)

    st.markdown(
        f"""
A coluna `fault` tem **{n_total} nomes diferentes**, mas nao sao {n_total} defeitos.
O mesmo problema aparece escrito de varias formas.

Para organizar isso, agrupamos os nomes por radical em **{n_familias} familias**.
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

    with st.expander("Ver o total de leituras por familia"):
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

    # ==========================================================================
    # 2. Analise de um ou mais rotulos
    # ==========================================================================
    st.header("2. Analisar um ou mais tipos de falha")

    st.markdown(
        "Escolha primeiro a **familia** para reduzir a lista, depois os **nomes** que "
    "quer comparar. Ate 4 por vez — cada um ganha uma cor e aparece junto dos outros "
    "nos mesmos graficos."
    )

    # --- filtro por familia ----------------------------------------------------
    familias_disponiveis = sorted(tabela["familia_sugerida"].dropna().unique())
    familias_escolhidas = st.multiselect(
        "1) Familia (opcional)",
        familias_disponiveis,
        default=[],
        help="Deixe vazio para ver todos os nomes.",
        format_func=lambda f: (
            f"{f}  ({int(tabela.loc[tabela.familia_sugerida == f, 'n_leituras'].sum())} leituras)"
        ),
    )

    if familias_escolhidas:
        disponivel = tabela[tabela["familia_sugerida"].isin(familias_escolhidas)]
    else:
        disponivel = tabela

    opcoes = disponivel.sort_values("n_leituras", ascending=False)["fault"].tolist()
    _leituras = dict(zip(tabela["fault"], tabela["n_leituras"]))

    # Ao trocar a familia, a selecao anterior pode conter nomes que saem da lista.
    # O Streamlit reclama se o valor guardado nao existe entre as opcoes, entao
    # limpamos antes de desenhar o widget.
    if CHAVE_ROTULOS not in st.session_state:
        st.session_state[CHAVE_ROTULOS] = opcoes[:1]
    else:
        validos = [r for r in st.session_state[CHAVE_ROTULOS] if r in opcoes]
        st.session_state[CHAVE_ROTULOS] = validos or opcoes[:1]

    if familias_escolhidas:
        st.caption(
            f"{len(opcoes)} nome(s) na(s) familia(s) escolhida(s), de {n_total} no total."
        )

    selecionados = st.multiselect(
        "2) Nomes para comparar",
        opcoes,
        key=CHAVE_ROTULOS,
        max_selections=MAX_ROTULOS,
        format_func=lambda r: f"{r}  ({_leituras[r]} leituras)",
    )

    if not selecionados:
        st.info("Escolha ao menos um nome para ver a analise.")
        return

    n_sel = len(selecionados)
    # Cores fixas por posicao: a mesma cor identifica o rotulo em todos os blocos.
    PALETA = ["#d1495b", "#4c78a8", "#2d6a4f", "#e2a03f"]
    cor_de = {r: PALETA[i % len(PALETA)] for i, r in enumerate(selecionados)}

    info_de = {r: tabela[tabela["fault"] == r].iloc[0] for r in selecionados}
    ordenado_de = {r: D.r_ordenado(r) for r in selecionados}

    numericas = D.r_numericas()
    padrao = "z_rms_velocity_mm_s" if "z_rms_velocity_mm_s" in numericas else numericas[0]

    # --- 2a. Panorama ----------------------------------------------------------
    st.subheader("Resumo de cada um")

    for coluna, rotulo in zip(st.columns(n_sel), selecionados):
        info = info_de[rotulo]
        ordenado = ordenado_de[rotulo]
        with coluna:
            st.markdown(
                f"<div style='border-left:5px solid {cor_de[rotulo]};padding-left:10px'>"
            f"<b>{rotulo}</b></div>",
                unsafe_allow_html=True,
            )
            st.metric("Leituras", f"{int(info['n_leituras']):,}".replace(",", "."))
            st.metric("% do arquivo", f"{info['pct']:.2f}%")
            st.metric("Familia", info["familia_sugerida"])
            st.metric("E defeito?", "Sim" if info["e_problema"] else "Nao, e um estado")
            st.metric("Sessoes de coleta", int(ordenado["sessao"].nunique()))
            st.metric("Horas cobertas", f"{info['span_horas']:.1f} h")

    estados = [r for r in selecionados if not info_de[r]["e_problema"]]
    if estados:
        st.info(
            f"**{', '.join(estados)}** nao e defeito, e um estado da maquina "
        "(operando normal, em teste ou desligada).\n\n"
        "No sistema final, isso encerra o atendimento: nao ha o que corrigir numa "
        "maquina que esta funcionando bem."
        )

    st.divider()

    # --- 2b. Serie temporal ----------------------------------------------------
    st.subheader("Como os valores variaram no tempo")

    st.markdown(
        """
Cada grafico abaixo e uma coluna de medida. O eixo horizontal e a **data e hora
reais** em que a leitura foi gravada (em UTC). Nada foi deslocado nem aproximado.

**A linha corta em alguns pontos de proposito.** Quando passam mais de 60 segundos
entre uma leitura e a seguinte, entendemos que uma gravacao terminou e outra
comecou depois. Ligar as duas desenharia uma transicao que nunca aconteceu.
"""
    )

    # Janela real de cada rotulo. Precisa ficar visivel porque os rotulos foram
    # coletados em campanhas diferentes: no eixo de tempo real eles aparecem lado a
    # lado, nao sobrepostos, e isso confunde quem espera curvas concorrentes.
    janelas = pd.DataFrame(
        [
            {
                "rotulo": r,
                "inicio": ordenado_de[r]["created_at"].iloc[0],
                "fim": ordenado_de[r]["created_at"].iloc[-1],
                "sessoes": int(ordenado_de[r]["sessao"].nunique()),
                "leituras": len(ordenado_de[r]),
            }
            for r in selecionados
        ]
    )

    st.caption("Quando cada um foi gravado:")
    st.dataframe(
        janelas,
        hide_index=True,
        column_config={
            "inicio": st.column_config.DatetimeColumn("primeira leitura",
                                                      format="DD/MM/YYYY HH:mm:ss"),
            "fim": st.column_config.DatetimeColumn("ultima leitura",
                                                   format="DD/MM/YYYY HH:mm:ss"),
            "sessoes": st.column_config.NumberColumn("gravacoes", format="%d"),
            "leituras": st.column_config.NumberColumn("leituras", format="%d"),
        },
    )

    if n_sel > 1:
        sobrepoe = any(
            (janelas.loc[i, "inicio"] <= janelas.loc[j, "fim"])
            and (janelas.loc[j, "inicio"] <= janelas.loc[i, "fim"])
            for i in range(n_sel)
            for j in range(i + 1, n_sel)
        )
        if not sobrepoe:
            st.warning(
                "**Estes tipos de falha foram gravados em dias diferentes.** Por isso as "
            "linhas aparecem em trechos separados do grafico, e nao uma em cima da "
            "outra. Isso e o dado, nao um erro do grafico.\n\n"
            "Para olhar de perto, use a roda do mouse sobre o grafico para dar zoom."
            )
        else:
            st.info(
                "Estes tipos de falha foram gravados em periodos que se cruzam — as linhas "
            "vao aparecer sobrepostas nos trechos em comum."
            )

    col_todas, col_faixa, col_altura = st.columns([1, 1, 2])
    with col_todas:
        todas_colunas = st.checkbox(
            "Todas as colunas",
            value=True,
            help="Desmarque para escolher poucas colunas e ver mais detalhe em cada uma.",
        )
    with col_faixa:
        mostrar_faixa = st.checkbox(
            "Mostrar minimo e maximo",
            value=False,
            help="Desenha uma faixa clara entre o menor e o maior valor de cada trecho.",
        )
    with col_altura:
        altura = st.slider(
            "Altura dos graficos (px)", 200, 700, 360, step=20,
            help="Graficos mais altos separam melhor variacoes pequenas.",
        )

    if todas_colunas:
        colunas_serie = numericas
    else:
        colunas_serie = st.multiselect("Colunas para mostrar", numericas, default=[padrao])

    if not colunas_serie:
        st.caption("Escolha ao menos uma coluna.")
    else:
        n_col = len(colunas_serie)

        # Orcamento global: sao ate n_sel x n_col series simultaneas na pagina, e
        # cada ponto vira um objeto JSON no navegador. Dividimos o teto da pagina
        # entre elas e respeitamos o piso, para a linha nao perder a forma.
        orcamento = int(
            min(
                D.config.MAX_PONTOS_SERIE,
                max(D.config.MIN_PONTOS_SERIE, D.config.MAX_PONTOS_PAGINA // (n_sel * n_col)),
            )
        )
        series = {
            r: D.r_serie_temporal(r, tuple(colunas_serie), orcamento) for r in selecionados
        }

        reamostrados = [r for r in selecionados if series[r]["reamostrado"]]
        if reamostrados:
            with st.expander("Por que os graficos mostram menos pontos que o total de leituras"):
                detalhe = "\n".join(
                    f"- `{r}`: {series[r]['n_original']:,} leituras viraram "
                f"{series[r]['n_pontos']:,} pontos (grupos de {series[r]['fator']})".replace(",", ".")
                    for r in reamostrados
                )
                st.markdown(
                    f"""
Um navegador nao aguenta desenhar centenas de milhares de pontos. Como aqui ha
**{n_col} coluna(s) x {n_sel} tipo(s) de falha = {n_col * n_sel} linhas** ao mesmo
tempo, agrupamos leituras vizinhas e mostramos o **valor do meio** de cada grupo.

{detalhe}

Duas garantias:

1. **Nenhuma data e inventada.** Cada ponto usa a hora real da primeira leitura do
   seu grupo.
2. **Os picos nao somem.** Marque *Mostrar minimo e maximo* para ver a faixa
   completa de cada grupo. Isso importa: em vibracao, o pico raro costuma ser o
   sinal do defeito, nao ruido.

Para ver mais detalhe, desmarque *Todas as colunas* e escolha poucas.
"""
                )

        escala_cor = alt.Scale(domain=selecionados, range=[cor_de[r] for r in selecionados])

        for coluna_dado in colunas_serie:
            # Um DataFrame com todos os rotulos: e o que permite sobrepor as series
            # num grafico so, cada uma com sua cor.
            partes = []
            for r in selecionados:
                d = series[r]["dados"]
                d = d[d["coluna"] == coluna_dado]
                if not d.empty:
                    partes.append(d.assign(rotulo=r))
            if not partes:
                continue
            junto = pd.concat(partes, ignore_index=True)

            st.markdown(f"**{coluna_dado}**")

            # O Vega agrupa as linhas pela combinacao dos canais discretos: `color`
            # (rotulo) e `detail` (sessao). Isso da uma linha por par
            # rotulo-sessao — cores distintas entre rotulos, e quebra nos gaps.
            base_ch = alt.Chart(junto).encode(
                x=alt.X("created_at:T", title="data e hora da coleta (UTC)"),
                color=alt.Color("rotulo:N", title=None, scale=escala_cor,
                                legend=alt.Legend(orient="bottom")),
                detail=alt.Detail("sessao:N"),
            )

            linha = base_ch.mark_line(strokeWidth=1.4).encode(
                y=alt.Y("valor:Q", title=coluna_dado, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("rotulo:N", title="tipo de falha"),
                    alt.Tooltip("created_at:T", title="quando",
                                format="%d/%m/%Y %H:%M:%S"),
                    alt.Tooltip("sessao:Q", title="gravacao no"),
                    alt.Tooltip("valor:Q", title="valor", format=".4f"),
                    alt.Tooltip("minimo:Q", title="minimo do grupo", format=".4f"),
                    alt.Tooltip("maximo:Q", title="maximo do grupo", format=".4f"),
                ],
            )

            if mostrar_faixa and reamostrados:
                faixa = base_ch.mark_area(opacity=0.18).encode(
                    y=alt.Y("minimo:Q", title=coluna_dado, scale=alt.Scale(zero=False)),
                    y2="maximo:Q",
                )
                grafico = faixa + linha
            else:
                grafico = linha

            # `.interactive()` liga zoom e pan no eixo do tempo — necessario quando
            # os rotulos estao a semanas de distancia e cada sessao vira um risco.
            st.altair_chart(grafico.properties(height=altura).interactive(), width="stretch")

    st.divider()

    # --- 2c. Assinatura --------------------------------------------------------
    st.subheader("O que diferencia cada tipo de falha")

    comparacoes = {r: D.r_comparacao(r) for r in selecionados}

    primeiro = comparacoes[selecionados[0]]
    comparada = primeiro[["feature", "mediana_global"]].copy()
    for r in selecionados:
        c = comparacoes[r].set_index("feature")
        comparada[f"{r}"] = comparada["feature"].map(c["mediana_rotulo"])
        comparada[f"Δ% {r}"] = comparada["feature"].map(c["desvio_pct"])

    # Ordem compartilhada por todos os graficos: sem ela, cada grafico ordenaria pelo
    # proprio desvio e as linhas nao corresponderiam entre as colunas.
    colunas_desvio = [f"Δ% {r}" for r in selecionados]
    comparada["_max_abs"] = comparada[colunas_desvio].abs().max(axis=1)
    comparada = comparada.sort_values("_max_abs", ascending=False).drop(columns="_max_abs")
    ordem_features = comparada["feature"].tolist()

    st.markdown(
        """
Aqui comparamos o **valor tipico** de cada tipo de falha com o valor tipico do
arquivo inteiro.

- A coluna **`mediana global`** e o valor do meio considerando todas as 166 mil leituras
- A coluna com o **nome da falha** e o valor do meio so daquele tipo
- A coluna **`Δ%`** e a diferenca entre os dois, em porcentagem

Um `Δ%` de +15% significa: nessa falha, essa medida fica 15% acima do normal.
Quanto maior o `Δ%`, mais aquela medida serve para reconhecer a falha.

As linhas estao ordenadas da maior para a menor diferenca.
"""
    )

    st.dataframe(
        comparada,
        hide_index=True,
        height=min(560, 40 + 35 * len(comparada)),
        column_config={
            "feature": st.column_config.TextColumn("medida", width="medium"),
            "mediana_global": st.column_config.NumberColumn("mediana global", format="%.4f"),
            **{r: st.column_config.NumberColumn(r, format="%.4f") for r in selecionados},
            **{
                f"Δ% {r}": st.column_config.NumberColumn(f"Δ% {r}", format="%.1f%%")
                for r in selecionados
            },
        },
    )

    st.caption(
        "Os mesmos numeros em grafico. Barra para a direita = acima do normal; "
    "para a esquerda = abaixo. As medidas aparecem na mesma ordem nos dois "
    "graficos, para poder comparar linha a linha."
    )

    for coluna, rotulo in zip(st.columns(n_sel), selecionados):
        with coluna:
            st.caption(f"**{rotulo}**")
            st.altair_chart(
                alt.Chart(comparacoes[rotulo])
                .mark_bar(color=cor_de[rotulo])
                .encode(
                    x=alt.X("desvio_pct:Q", title="diferenca do normal (%)"),
                    y=alt.Y("feature:N", sort=ordem_features, title=None),
                    tooltip=["feature", "mediana_rotulo", "mediana_global", "desvio_pct"],
                )
                .properties(height=26 * len(ordem_features)),
                width="stretch",
            )

    with st.expander("Ver estatistica completa de cada tipo (quartis, desvio, variacao)"):
        st.markdown(
            """
`cv` significa **coeficiente de variacao**: o quanto os valores se espalham em
relacao a media.

- `cv` baixo — as leituras desse tipo sao parecidas entre si
- `cv` alto — as leituras variam muito, mesmo sendo a mesma falha

Isso importa para a proxima etapa: tipos com `cv` alto sao os que o sistema mais
vai confundir com outros.
"""
        )
        for coluna, rotulo in zip(st.columns(n_sel), selecionados):
            with coluna:
                st.caption(f"**{rotulo}**")
                st.dataframe(
                    D.r_assinatura(rotulo),
                    hide_index=True,
                    height=400,
                    column_config={
                        "cv": st.column_config.NumberColumn("cv", format="%.3f"),
                    },
                )

    st.divider()

    # --- 2d. Distribuicao ------------------------------------------------------
    st.subheader("Como os valores se distribuem")

    feature = st.selectbox("Medida", numericas, index=numericas.index(padrao))

    serie_global = D.dados()[feature].to_numpy()
    series_rotulo = {r: D.r_serie(r, feature) for r in selecionados}

    # Bordas COMPARTILHADAS entre todos os rotulos. Com bins proprios por rotulo os
    # histogramas ficariam com larguras diferentes e a comparacao visual seria falsa.
    lo = float(min(min(s.min() for s in series_rotulo.values()),
                   np.quantile(serie_global, 0.001)))
    hi = float(max(max(s.max() for s in series_rotulo.values()),
                   np.quantile(serie_global, 0.999)))
    if hi <= lo:
        hi = lo + 1e-6
    bordas = np.linspace(lo, hi, 61)
    centros = (bordas[:-1] + bordas[1:]) / 2
    h_glo, _ = np.histogram(serie_global, bins=bordas)
    dens_glo = h_glo / max(h_glo.sum(), 1)

    st.markdown(
        """
Cada grafico mostra **onde os valores dessa medida se concentram**. A area colorida
e o tipo de falha escolhido; a cinza e o arquivo inteiro, para comparar.

Se a area colorida estiver deslocada em relacao a cinza, essa medida distingue bem
a falha. Se estiverem em cima uma da outra, essa medida nao ajuda a reconhece-la.

O eixo vertical mostra **proporcao**, nao contagem — assim um tipo com poucas
leituras nao fica invisivel ao lado de um com muitas.
"""
    )

    for coluna, rotulo in zip(st.columns(n_sel), selecionados):
        h_rot, _ = np.histogram(series_rotulo[rotulo], bins=bordas)
        hist = pd.DataFrame(
            {
                "valor": np.concatenate([centros, centros]),
                "densidade": np.concatenate([h_rot / max(h_rot.sum(), 1), dens_glo]),
                "serie": [rotulo] * len(centros) + ["arquivo inteiro"] * len(centros),
            }
        )
        with coluna:
            st.caption(f"**{rotulo}**")
            st.altair_chart(
                alt.Chart(hist)
                .mark_area(opacity=0.55, interpolate="step")
                .encode(
                    x=alt.X("valor:Q", title=feature),
                    y=alt.Y("densidade:Q", title="proporcao", stack=None),
                    color=alt.Color(
                        "serie:N",
                        title=None,
                        scale=alt.Scale(
                            domain=[rotulo, "arquivo inteiro"],
                            range=[cor_de[rotulo], "#b0b0b0"],
                        ),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=[
                        "serie",
                        alt.Tooltip("valor:Q", format=".4f"),
                        alt.Tooltip("densidade:Q", title="proporcao", format=".4f"),
                    ],
                )
                .properties(height=260),
                width="stretch",
            )

    st.divider()

    # --- 2e. Outliers ----------------------------------------------------------
    with st.expander("Valores fora do normal dentro de cada tipo"):
        st.markdown(
            """
Um valor e considerado **fora do normal** quando se afasta muito do meio da propria
classe. O calculo aqui e feito **dentro de cada tipo de falha**, nao no arquivo todo.

A diferenca importa: se comparassemos com o arquivo todo, quase toda leitura de uma
falha grave pareceria anormal — o que e esperado, e nao ajuda em nada. Assim vemos
o que destoa **dentro** da propria falha.

Nada e removido. Aqui so apontamos.
"""
        )
        for coluna, rotulo in zip(st.columns(n_sel), selecionados):
            with coluna:
                st.caption(f"**{rotulo}**")
                st.dataframe(
                    D.r_outliers_do_rotulo(rotulo)[
                        ["coluna", "mediana", "lim_inferior", "lim_superior",
                         "min", "max", "outliers", "pct_outliers"]
                    ],
                    hide_index=True,
                    height=320,
                    column_config={
                        "coluna": "medida",
                        "mediana": st.column_config.NumberColumn("valor do meio", format="%.4f"),
                        "lim_inferior": st.column_config.NumberColumn("limite baixo", format="%.4f"),
                        "lim_superior": st.column_config.NumberColumn("limite alto", format="%.4f"),
                        "outliers": st.column_config.NumberColumn("fora", format="%d"),
                        "pct_outliers": st.column_config.NumberColumn("% fora", format="%.2f%%"),
                    },
                )

    # --- 2f. Base ordenada -----------------------------------------------------
    with st.expander("Ver as leituras em ordem de data"):
        st.markdown(
            """
As leituras de cada tipo, da mais antiga para a mais recente.

- **`gravacao`** — numero da sessao de coleta. Muda quando ha uma pausa longa.
- **`intervalo (s)`** — segundos desde a leitura anterior. Fica vazio na primeira
  leitura de cada gravacao.

Cada tipo esta numa aba porque a tabela tem 26 colunas — lado a lado ficaria ilegivel.
"""
        )
        for aba, rotulo in zip(st.tabs(selecionados), selecionados):
            with aba:
                ordenado = ordenado_de[rotulo]
                colunas_tabela = (
                    ["created_at", "sessao", "delta_s", "id", "fault"]
                    + [c for c in numericas if c in ordenado.columns]
                )
                st.dataframe(
                    ordenado[colunas_tabela],
                    hide_index=True,
                    height=400,
                    column_config={
                        "created_at": st.column_config.DatetimeColumn(
                            "quando", format="DD/MM/YY HH:mm:ss.SSS"
                        ),
                        "sessao": "gravacao",
                        "delta_s": st.column_config.NumberColumn("intervalo (s)", format="%.2f"),
                    },
                )
                st.download_button(
                    "Baixar em CSV",
                    ordenado[colunas_tabela].to_csv(index=False).encode("utf-8"),
                    file_name=f"{rotulo}_ordenado_por_data.csv",
                    mime="text/csv",
                    key=f"dl_{rotulo}",
                )

    st.divider()

    # ==========================================================================
    # 3. Tabela de assinaturas
    # ==========================================================================
    st.header("3. Tabela de assinaturas")

    st.markdown(
        """
### O que e uma assinatura

Cada tipo de falha faz a maquina vibrar de um jeito diferente. Desalinhamento
sacode mais num sentido; rolamento quebrado gera pancadinhas rapidas; peca
desbalanceada balanca de forma constante.

A **assinatura** e o retrato desse jeito de vibrar, em numeros: para cada tipo de
falha, o valor tipico de cada medida do sensor.

### Como ler a tabela

Cada **linha** e um nome da coluna `fault`. Cada **coluna** e uma medida do sensor.
O numero na celula e o **valor do meio** (mediana) daquela medida, naquele tipo.

Exemplo: se `rolamento_inner` tem `z_kurtosis = 2,44`, esse e o valor tipico de
kurtosis quando essa falha esta presente.

### Por que "valor do meio" e nao media

`kurtosis` e `crest_factor` medem picos. Um unico impacto forte, mesmo que dure
um segundo, puxa a **media** do tipo inteiro para cima e da uma impressao errada.
O **valor do meio** ignora esse exagero e descreve o comportamento comum.

### Para que serve

Esta tabela e a ponte entre os **sensores** e os **manuais**.

Os manuais de procedimento descrevem sintomas em palavras — *"vibracao radial
elevada"*, *"vibracao em altas frequencias"*. Esta tabela diz esses mesmos sintomas
em numeros. Cruzar as duas coisas responde: **o que o sensor mede bate com o que o
manual descreve?**

Quando nao bater, e um achado para investigar — nao um erro de calculo.
"""
    )

    minimo = st.slider(
        "Ignorar tipos com menos de N leituras",
        0,
        500,
        100,
        step=50,
        help="Alguns nomes tem so 2 leituras. Um valor tipico calculado sobre 2 leituras "
         "nao significa nada.",
    )
    st.dataframe(D.r_assinaturas(minimo), hide_index=True, height=400)

    st.caption(
        "As colunas comecadas por `z_` e `x_` sao os dois eixos do sensor. "
    "`rms` = valor medio da vibracao; `peak` = valor de pico; `kurtosis` e "
    "`crest_factor` medem o quanto ha de impacto; `high_freq_rms_accel_g` e a "
    "vibracao em alta frequencia, tipica de rolamento com defeito."
    )
