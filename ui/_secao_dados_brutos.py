"""Ato 2 da narrativa — **o arquivo cru**, sem ordenar, agrupar ou reamostrar.

Todas as outras telas corrigem alguma coisa antes de desenhar: ordenam por data,
separam sessoes, agregam pontos vizinhos. Esta nao corrige nada. O eixo x e a
posicao da linha no arquivo, e cada ponto e uma leitura.

Era uma pagina propria (`pages/0_Dados_Brutos.py`). Virou secao da narrativa
unica pelo mesmo motivo dos outros atos, e com a mesma conversao conferida
literal por literal.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D



def render() -> None:
    """Desenha esta secao dentro da narrativa unica."""

    st.header("📄 O arquivo cru", divider="gray")
    st.caption("O arquivo na ordem em que foi lido. Nada tratado.")

    try:
        saltos = D.r_saltos()
    except FileNotFoundError as e:
        D.aviso_csv_ausente(e)


    def _mil(n) -> str:
        """Milhar com ponto. Aplicado so ao numero — nunca ao texto em volta, senao
    as virgulas da propria frase virariam pontos junto."""
        return f"{int(n):,}".replace(",", ".")

    st.markdown(
        """
Nas outras telas o dado ja chega arrumado — ordenado por data, separado por sessao,
com pontos vizinhos agrupados para caber no navegador. Aqui **nada disso acontece**.

O eixo horizontal e a **posicao da linha no arquivo**: linha 0, linha 1, linha 2.
Nao e o tempo. Cada ponto do grafico e uma leitura real, sem media nem resumo.
"""
    )

    # --------------------------------------------------------------------------
    # 1. Por que esta tela existe
    # --------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas no arquivo", _mil(saltos["linhas"]))
    c2.metric("Esta em ordem de data?", "sim" if saltos["em_ordem"] else "nao")
    c3.metric("Emendas fora de ordem", _mil(saltos["saltos_para_tras"]),
              delta=f"{saltos['pct_para_tras']}% das linhas", delta_color="off")
    c4.metric("Maior recuo", f"{saltos['maior_recuo_dias']:.0f} dias")

    st.info(
        f"""
**Ler o arquivo de cima para baixo nao e andar no tempo.**

Em {_mil(saltos['saltos_para_tras'])} pontos do arquivo, a linha seguinte e **mais
antiga** que a anterior — num deles, {saltos['maior_recuo_dias']:.0f} dias mais
antiga. Nas outras {100 - saltos['pct_para_tras']:.2f}% das linhas o tempo anda para
a frente {saltos['avanco_mediano_s']:.0f} s, que e a cadencia do sensor.

Ou seja: o arquivo e feito de blocos internamente em ordem, colados uns nos outros
fora de ordem. Sao gravacoes curtas de dias diferentes. E por isso que toda conta
do projeto ordena por data antes de qualquer coisa.
"""
    )

    st.divider()

    # --------------------------------------------------------------------------
    # 2. Onde olhar
    # --------------------------------------------------------------------------
    st.subheader("1. Escolher o trecho do arquivo")

    total_arquivo = saltos["linhas"]

    # --- quanto tem de cada falha, ANTES de escolher --------------------------
    #
    # Um `value_counts` da coluna `fault`, desenhado. Escolher no escuro e o que faz
    # alguem filtrar por um rotulo de 40 leituras e concluir que "nao tem dado".
    tabela_rotulos = D.r_rotulos()
    ordem_arquivo = D.r_rotulos_do_arquivo()

    st.markdown("**Quantas leituras tem cada tipo de falha**")

    col_ord, col_alt = st.columns([2, 1])
    with col_ord:
        # O padrao e a ordem do arquivo, coerente com o resto da tela. Ordenar por
        # quantidade e util para achar os rotulos raros, entao fica como opcao — e
        # como escolha explicita, nao como default silencioso.
        por_quantidade = st.toggle(
            "Ordenar por quantidade", value=False,
            help="Desligado, a ordem e a de aparicao no arquivo — a mesma que o "
             "filtro e o grafico usam.",
        )
    with col_alt:
        st.caption(f"{len(ordem_arquivo)} valores distintos em `fault`.")

    st.altair_chart(
        alt.Chart(tabela_rotulos)
        .mark_bar()
        .encode(
            x=alt.X("n_leituras:Q", title="leituras no arquivo"),
            y=alt.Y(f"{D.config.COLUNA_ROTULO}:N", title=None,
                    sort="-x" if por_quantidade else ordem_arquivo),
            color=alt.Color(
                "e_problema:N", title="e defeito?",
                scale=alt.Scale(domain=[True, False], range=["#d1495b", "#5c8a8a"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip(f"{D.config.COLUNA_ROTULO}:N", title="fault"),
                alt.Tooltip("n_leituras:Q", title="leituras", format=","),
                alt.Tooltip("pct:Q", title="% do arquivo", format=".2f"),
                alt.Tooltip("e_problema:N", title="e defeito?"),
            ],
        )
        .properties(height=max(240, 16 * len(tabela_rotulos))),
        width="stretch",
    )

    st.caption(
        "A barra conta **linhas**, nao ocorrencias: 13 mil linhas de "
    "`rolamento_inner` sao uma falha medida por 34 horas seguidas, nao 13 mil "
    "falhas. Agrupar em eventos e assunto da tela **Eventos**."
    )

    # --- filtro por tipo de falha ---------------------------------------------
    #
    # A lista de opcoes segue a ordem em que cada rotulo APARECE no arquivo — nao
    # por frequencia nem alfabetica. E a mesma ordem natural que a tela inteira
    # respeita; ordenar aqui seria a primeira correcao silenciosa da pagina.
    #
    # Filtrar nao reordena nada: seleciona linhas e mantem a numeracao do arquivo,
    # entao a leitura da linha 90.000 continua desenhada em 90.000.
    contagem = tabela_rotulos.set_index("fault")["n_leituras"].to_dict()

    filtro = st.multiselect(
        "Tipo de falha",
        ordem_arquivo,
        default=[],
        format_func=lambda r: f"{r} — {_mil(contagem.get(r, 0))} leituras",
        help="Vazio mostra o arquivo inteiro. Escolher um ou mais rotulos esconde "
         "as outras linhas, sem mexer na ordem nem renumerar: o eixo continua "
         "sendo a posicao no arquivo.",
    )

    total = sum(contagem.get(r, 0) for r in filtro) if filtro else total_arquivo

    col_a, col_b = st.columns([3, 1])
    with col_b:
        quantidade = st.select_slider(
            "Linhas por vez", options=[200, 500, 1000, 2000, 5000], value=1000,
            help="Sem agrupar nenhum ponto, so cabe um trecho por vez. "
             f"Sao {_mil(total)} linhas no recorte atual.",
        )
    with col_a:
        inicio = st.slider(
            "Comecar na linha", 0, max(0, total - quantidade), 0, step=max(1, quantidade // 4),
            help="Arraste para percorrer o recorte do inicio ao fim. Com filtro "
             "ligado, conta as linhas que sobraram — nao as do arquivo.",
        )

    colunas_numericas = D.r_numericas()
    padrao = [c for c in ("rms_velocity_z_mm_s", "kurtosis_z", "rpm") if c in colunas_numericas]

    col_todas, col_altura = st.columns([1, 3])
    with col_todas:
        todas = st.checkbox("Todas as colunas", value=False)
    with col_altura:
        altura = st.slider("Altura do grafico (px)", 140, 700, 340, step=20)

    if todas:
        colunas = colunas_numericas
        st.caption(f"As {len(colunas)} colunas numericas. Desmarque para escolher a mao.")
    else:
        colunas = st.multiselect(
            "Colunas para desenhar", colunas_numericas,
            default=padrao or colunas_numericas[:3],
            help="Escolha quais medidas aparecem. Os valores sao os do arquivo.",
        )

    if not colunas:
        st.caption("Escolha ao menos uma coluna.")
        return

    bruto = D.r_serie_bruta(tuple(colunas), inicio, quantidade, tuple(filtro))

    pontos = bruto["n_linhas"] * len(colunas)
    st.caption(
        f"Mostrando as linhas **{_mil(bruto['linha_inicio'])} a "
    f"{_mil(bruto['linha_fim'])}** do arquivo — {_mil(bruto['n_linhas'])} "
    f"leituras, uma por ponto, sem agrupar nem resumir. "
    f"Total no grafico: {_mil(pontos)} pontos."
    )

    if bruto["filtrado"]:
        vao = bruto["linha_fim"] - bruto["linha_inicio"] + 1
        st.warning(
            f"**Filtro ligado: {', '.join(bruto['rotulos_filtro'])}.** As "
        f"{_mil(bruto['n_linhas'])} leituras deste recorte estao espalhadas por "
        f"{_mil(vao)} linhas do arquivo, em **{_mil(bruto['n_blocos'])} trechos "
        "separados**. A linha do grafico se interrompe entre um trecho e outro — "
        "o que sumiu no meio sao as outras falhas, nao um buraco na coleta."
        )
    if pontos > 60_000:
        st.warning(
            f"**{_mil(pontos)} pontos e muito para o navegador desenhar com folga.** "
        "Cada ponto vira um objeto no HTML. Reduza as linhas por vez ou o numero "
        "de colunas se a pagina ficar lenta — aqui nada e reamostrado de proposito."
        )

    st.divider()

    # --------------------------------------------------------------------------
    # 3. A serie crua, com a faixa de rotulo embaixo
    # --------------------------------------------------------------------------
    st.subheader("2. A serie como esta no arquivo")

    dados = bruto["dados"]
    faixas = bruto["faixas"]
    rot = D.config.COLUNA_ROTULO

    # Uma cor por rotulo presente no trecho, reaproveitada pela faixa de baixo e
    # pelas linhas verticais — assim os tres graficos falam da mesma coisa.
    presentes = list(dict.fromkeys(bruto["rotulos"]))
    PALETA = [
        "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2",
        "#ff9da6", "#9d755d", "#eeca3b", "#bab0ac", "#1f77b4", "#d62728",
    ]
    escala_rot = alt.Scale(domain=presentes,
                           range=[PALETA[i % len(PALETA)] for i in range(len(presentes))])

    tem_troca = not bruto["trocas"].empty

    # Uma cor por coluna, para o grafico unico. Independente da paleta dos rotulos.
    escala_col = alt.Scale(domain=colunas,
                           range=[PALETA[i % len(PALETA)] for i in range(len(colunas))])

    col_e, col_d = st.columns([1, 1])
    with col_e:
        juntas = st.toggle(
            "Todas as colunas no mesmo grafico", value=True,
            help="Desligado, cada coluna ganha um grafico proprio, na escala real dela.",
        )
    with col_d:
        marcar_troca = st.checkbox(
            "Marcar e nomear onde o rotulo muda", value=True, disabled=not tem_troca,
            help="Linha vertical no ponto em que `fault` mudou, com a etiqueta "
             "'vinha de → passou a' escrita no grafico."
            if tem_troca else "Neste trecho o rotulo nao muda.",
        )

    _dicas_troca = [
        alt.Tooltip("linha:Q", title="linha do arquivo"),
        alt.Tooltip("de:N", title="vinha de"),
        alt.Tooltip("para:N", title="passou a"),
        alt.Tooltip("salto_s:Q", title="salto no tempo (s)", format=".1f"),
    ]

    # --------------------------------------------------------------------------
    # A barra de rolagem
    # --------------------------------------------------------------------------
    #
    # Primeiro tentei deixar o grafico mais largo que a tela e ligar `overflow-x`
    # no container com CSS. Nao funcionou: o Streamlit envolve o grafico em divs
    # proprias e a barra nao aparecia. CSS que depende da estrutura interna de
    # outro projeto e fragil por natureza.
    #
    # A solucao aqui nao usa CSS nenhum. A barra e **parte do grafico**: um mapa
    # reduzido da serie inteira, com uma janela colorida que se arrasta. Ela e
    # desenhada pelo Vega a partir do dado, entao aparece sempre, em qualquer
    # navegador, sem depender de como o Streamlit monta o HTML em volta.
    #
    # De quebra, faz o que uma barra comum nao faria: mostra a forma da serie
    # inteira, para voce saber para onde esta arrastando.

    JANELA = alt.selection_interval(
        encodings=["x"],
        # Vermelho forte e alca visivel — o pedido era "cor destacada".
        mark=alt.BrushConfig(fill="#e45756", fillOpacity=0.20,
                             stroke="#c9302c", strokeWidth=2),
    )

    eixo_x = alt.X("linha:Q", title=None,
                   scale=alt.Scale(domain=JANELA, nice=False))

    # **Todas as camadas de um mesmo grafico usam `eixo_x`** — nao e enfeite, e o
    # que faz a janela funcionar.
    #
    # Num grafico em camadas o Vega-Lite funde a escala x das camadas numa so. Se
    # uma camada declara o dominio preso a selecao e a irma declara um x simples,
    # a fusao descarta o vinculo **em silencio**: o pincel continua desenhando no
    # mapa e o grafico de cima nao se mexe. Como "marcar as trocas" ja vem ligado,
    # a barra nunca funcionava no estado padrao da tela.
    #
    # Conferido no spec compilado: com o x simples nas regras, a escala da serie
    # sai com `domainRaw: null`; com `eixo_x` nas tres camadas, sai com
    # `domainRaw: {"signal": ...}`, que e o vinculo com o pincel.
    #
    # E a segunda armadilha silenciosa do mesmo mecanismo — a outra esta anotada
    # junto ao `add_params` do mapa, mais abaixo.

    regras = (
        alt.Chart(bruto["trocas"])
        .mark_rule(strokeDash=[4, 3], strokeWidth=1, color="#999")
        .encode(x=eixo_x, tooltip=_dicas_troca)
    )

    # A etiqueta "vinha de → passou a", escrita no grafico e nao no tooltip. Duas
    # filas alternadas para trocas proximas nao se cobrirem.
    etiquetas = (
        alt.Chart(bruto["trocas"])
        .mark_text(align="left", dx=3, fontSize=10, color="#444", baseline="top")
        .encode(
            x=eixo_x,
            y=alt.Y("fila:Q", axis=None, scale=alt.Scale(domain=[-0.4, 6])),
            text=alt.Text("etiqueta:N"),
            tooltip=_dicas_troca,
        )
    )


    def _com_marcas(base, altura_px: int):
        """Sobrepoe as linhas de troca e as etiquetas fixas ao grafico da serie.

    `resolve_scale(y="independent")` e obrigatorio: a etiqueta usa uma escala
    propria (a fila 0/1) que nada tem a ver com o eixo dos valores. Sem isso o
    Vega tentaria unir as duas e o grafico ficaria achatado.
    """
        grafico = base
        if marcar_troca and tem_troca:
            grafico = alt.layer(base, regras, etiquetas).resolve_scale(y="independent")
        return grafico.properties(height=altura_px)


    def _dica() -> list:
        return [
            alt.Tooltip("coluna:N", title="coluna"),
            alt.Tooltip("linha:Q", title="linha do arquivo"),
            alt.Tooltip(f"{rot}:N", title="fault"),
            alt.Tooltip("created_at:T", title="quando", format="%d/%m/%Y %H:%M:%S"),
            alt.Tooltip("valor:Q", title="valor lido", format=".4f"),
            alt.Tooltip("valor_z:Q", title="desvios da media da coluna", format="+.2f"),
        ]


    # --- a faixa de rotulo, que acompanha a mesma janela ----------------------

    _dicas_faixa = [
        alt.Tooltip(f"{rot}:N", title="fault"),
        alt.Tooltip("linha_inicio:Q", title="da linha"),
        alt.Tooltip("linha_fim:Q", title="ate a linha"),
        alt.Tooltip("n_linhas:Q", title="leituras"),
        alt.Tooltip("inicio:T", title="primeira leitura", format="%d/%m/%Y %H:%M:%S"),
        alt.Tooltip("fim:T", title="ultima leitura", format="%d/%m/%Y %H:%M:%S"),
    ]

    cor_rotulo = alt.Color(f"{rot}:N", title=None, scale=escala_rot,
                           legend=alt.Legend(orient="bottom", columns=4))

    blocos = (
        alt.Chart(faixas)
        .mark_rect()
        .encode(
            x=alt.X("linha_inicio:Q", title="posicao da linha no arquivo",
                    scale=alt.Scale(domain=JANELA, nice=False)),
            x2="ate:Q", color=cor_rotulo, tooltip=_dicas_faixa,
        )
    )

    # O nome do rotulo escrito dentro do proprio bloco: quem olha a faixa le o
    # rotulo sem passar o mouse e sem procurar a cor na legenda.
    nomes = (
        alt.Chart(faixas)
        .mark_text(fontSize=10, color="white", fontWeight="bold", baseline="middle")
        .encode(
            x=alt.X("centro:Q", scale=alt.Scale(domain=JANELA, nice=False)),
            text=alt.Text(f"{rot}:N"),
            tooltip=_dicas_faixa,
            # Bloco estreito nao cabe o texto; escondemos em vez de deixar vazar.
            opacity=alt.condition(
                alt.datum.n_linhas > max(20, bruto["n_linhas"] / 40),
                alt.value(1), alt.value(0),
            ),
        )
    )

    faixa_rotulo = alt.layer(blocos, nomes).properties(height=64)

    # --- a barra: o mapa da serie inteira, com a janela arrastavel ------------

    campo_mapa = "valor_z:Q" if len(colunas) > 1 else "valor:Q"

    mapa = alt.layer(
        # Fundo: onde cada rotulo comeca e termina, para orientar o arrasto.
        alt.Chart(faixas).mark_rect(opacity=0.35).encode(
            # O dominio da barra e o vao REAL ocupado no arquivo, nao o tamanho do
            # recorte: com filtro os dois deixam de coincidir.
            x=alt.X("linha_inicio:Q", title=None,
                    scale=alt.Scale(domain=[bruto["linha_inicio"], bruto["linha_fim"]],
                                    nice=False)),
            x2="ate:Q", color=alt.Color(f"{rot}:N", legend=None, scale=escala_rot),
        ),
        # `add_params` vai na camada UNITARIA, nunca no `layer`. O Vega-Lite so
        # aceita selecao em spec de vista unica e o Altair a descarta **em
        # silencio** quando posta no layer — o grafico continua desenhando, so que
        # sem a janela. Foi exatamente o que aconteceu na primeira tentativa.
        alt.Chart(dados).mark_line(strokeWidth=0.6, color="#555", opacity=0.8).encode(
            x=alt.X("linha:Q", title=None),
            y=alt.Y(campo_mapa, title=None, axis=None, scale=alt.Scale(zero=False)),
            # Uma linha por coluna E por trecho contiguo — mesma razao do grafico
            # de cima: nao emendar o que o filtro separou.
            detail=["coluna:N", "bloco:N"],
        ).add_params(JANELA),
    ).properties(height=70)

    # --- montagem -------------------------------------------------------------
    #
    # Tudo num spec so. E o que faz a janela do mapa comandar os graficos de cima:
    # uma selecao do Vega so alcanca os graficos que estao no MESMO spec — em
    # chamadas separadas de `st.altair_chart` cada grafico vira um Vega isolado e
    # o arrasto nao chega a lugar nenhum.

    partes = []

    if juntas:
        # As unidades nao sao comparaveis (rpm na casa dos milhares, kurtosis perto
        # de 2,5), entao o eixo comum e em desvios-padrao. O valor lido continua no
        # tooltip.
        campo_y, titulo_y = "valor_z:Q", "desvios-padrao da media da coluna"
        if len(colunas) == 1:
            # Uma coluna so nao precisa de escala comum: mostra o numero de verdade.
            campo_y, titulo_y = "valor:Q", colunas[0]

        linha = (
            alt.Chart(dados)
            .mark_line(strokeWidth=1.2)
            .encode(
                x=eixo_x,
                y=alt.Y(campo_y, title=titulo_y, scale=alt.Scale(zero=False)),
                # Legenda em CIMA de proposito: embaixo ela se enfiaria entre a
                # serie e a barra de rolagem, que e justamente o que se quis colar.
                color=alt.Color("coluna:N", title=None, scale=escala_col,
                                legend=alt.Legend(orient="top", columns=3)),
                # Quebra a linha entre trechos nao vizinhos no arquivo. Sem isso o
                # filtro desenharia um segmento ligando a linha 900 a 90.000 como se
                # fosse uma leitura seguida da outra.
                detail=alt.Detail("bloco:N"),
                tooltip=_dica(),
            )
        )
        partes.append(_com_marcas(linha, max(altura, 320)))
    else:
        for coluna in colunas:
            d = dados[dados["coluna"] == coluna]
            if d.empty:
                continue
            linha = (
                alt.Chart(d)
                .mark_line(strokeWidth=1, color="#333")
                .encode(
                    x=eixo_x,
                    y=alt.Y("valor:Q", title=coluna, scale=alt.Scale(zero=False)),
                    detail=alt.Detail("bloco:N"),
                    tooltip=_dica(),
                )
            )
            partes.append(_com_marcas(linha, altura))

    # A barra vem **logo depois dos graficos da serie**, antes da faixa de rotulo:
    # quem arrasta olha a serie e o pincel de uma vez, sem varrer a tela. A faixa de
    # rotulo desce para o rodape — ela acompanha a mesma janela e continua alinhada
    # na vertical com a serie, so que agora com a barra entre as duas.
    st.altair_chart(
        alt.vconcat(*partes, mapa, faixa_rotulo)
        .resolve_scale(color="independent")
        .configure_view(stroke=None),
        width="stretch",
    )

    st.info(
        "**↔️ A barra de rolagem e o grafico estreito logo abaixo da serie.** Arraste "
    "sobre ele para escolher o trecho — a **janela vermelha** e o que os graficos "
    "de cima mostram. Arraste a janela pelo meio para percorrer o arquivo; puxe as "
    "bordas para abrir ou fechar o zoom; **clique fora dela para ver tudo de "
    "novo**.\n\n"
    "O fundo colorido da barra e a coluna `fault`, para voce saber para onde "
    "esta arrastando."
    )

    if len(colunas) > 1 and juntas:
        st.caption(
            "**Eixo vertical em desvios-padrao.** Sem isso, `rpm` (milhares) achataria "
        "`kurtosis` (perto de 2,5) numa reta. Zero = a media daquela coluna no "
        "arquivo inteiro; 2 = duas vezes o desvio acima dela. O valor original "
        "aparece ao passar o mouse."
        )

    st.caption(
        "A faixa colorida do rodape e a coluna `fault`: cada bloco e um trecho seguido de "
    "linhas com o mesmo valor, com o nome escrito dentro. Ela acompanha a janela, "
    "entao da para ler na vertical qual rotulo valia em cada pedaco da serie. Bloco "
    "curto demais fica sem o nome escrito; o mouse mostra."
    )

    st.divider()

    # --------------------------------------------------------------------------
    # 4. As trocas, uma a uma
    # --------------------------------------------------------------------------
    st.subheader("3. Onde o rotulo mudou")

    trocas = bruto["trocas"]

    if trocas.empty:
        st.info(
            f"Neste trecho o rotulo nao muda: as {bruto['n_linhas']} linhas sao todas "
        f"`{presentes[0] if presentes else '?'}`. Arraste o inicio para achar uma troca."
        )
    else:
        st.markdown(
            f"""
Foram **{len(trocas)} trocas** neste trecho. A coluna `salto_s` e a diferenca de tempo
entre a linha da troca e a anterior:

- valor perto de **2 s** — a troca e real, o operador mudou a anotacao durante a coleta
- valor **muito grande** — mudou de gravacao; sao ensaios de dias diferentes colados
- valor **negativo** — a linha seguinte e **mais antiga**: o arquivo voltou no tempo
"""
        )

        tabela = trocas.copy()
        tabela["salto"] = tabela["salto_s"].map(
            lambda s: "—" if pd.isna(s)
            else (f"{s:.1f} s" if abs(s) < 120 else f"{s / 3600:.1f} h")
        )
        tabela["voltou no tempo"] = tabela["salto_s"] < 0
        tabela = tabela.rename(columns={
            "linha": "linha do arquivo", "de": "vinha de", "para": "passou a",
            "created_at": "quando", "salto_s": "salto (s)",
        })

        st.dataframe(
            tabela[["linha do arquivo", "vinha de", "passou a", "quando",
                    "salto", "voltou no tempo"]],
            width="stretch", hide_index=True,
        )

        n_volta = int((trocas["salto_s"] < 0).sum())
        if n_volta:
            st.warning(
                f"**{n_volta} destas trocas voltam no tempo.** Nao e erro de leitura: e a "
            "emenda entre duas gravacoes que foram salvas no arquivo fora de ordem."
            )

    st.divider()

    # --------------------------------------------------------------------------
    # 5. A tabela mesmo
    # --------------------------------------------------------------------------
    st.subheader("4. As linhas, como estao")

    st.markdown(
        "Sem nenhuma coluna escondida — inclusive `id` e `created_at`, que a analise "
    "depois deixa de usar como medida."
    )

    linhas = D.r_linhas_cruas(bruto["inicio"], min(bruto["n_linhas"], 500),
                              tuple(filtro))
    st.caption(
        f"Primeiras {len(linhas)} linhas do trecho. A coluna `linha do arquivo` e a "
    "posicao original — com filtro ligado, ela pula."
    )
    st.dataframe(linhas, width="stretch")
