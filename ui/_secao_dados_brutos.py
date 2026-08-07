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

    # O comeco do arquivo, antes de qualquer controle desta tela.
    #
    # **Sem filtro e sem deslocamento, de proposito.** A tabela vinha depois dos
    # seletores e acompanhava o recorte deles; aqui em cima nao ha recorte ainda
    # — ninguem escolheu falha nem posicao. Amarra-la ao `bruto`/`filtro` que so
    # existem la embaixo era o que quebrava a tela.
    #
    # E a posicao certa para ela: a primeira coisa da secao passa a ser o
    # arquivo, e nao um texto sobre o arquivo.
    LINHAS_DE_ABERTURA = 200
    st.dataframe(D.r_linhas_cruas(0, LINHAS_DE_ABERTURA), width="stretch")
    st.caption(
        f"As primeiras {LINHAS_DE_ABERTURA} linhas, na ordem em que estao no "
        "arquivo. A coluna `linha do arquivo` e a posicao original. Mais abaixo "
        "da para filtrar por falha e percorrer o resto."
    )

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
    # O total de linhas nao entra aqui: e o primeiro numero do ato 1, e repeti-lo
    # gasta uma coluna com uma informacao que o leitor acabou de ver. As tres que
    # ficam sao as que **so** este ato mede.
    c1, c2, c3 = st.columns(3)
    c1.metric("Esta em ordem de data?", "sim" if saltos["em_ordem"] else "nao")
    c2.metric("Emendas fora de ordem", _mil(saltos["saltos_para_tras"]),
              delta=f"{saltos['pct_para_tras']}% das linhas", delta_color="off")
    c3.metric("Maior recuo", f"{saltos['maior_recuo_dias']:.0f} dias")

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

    # Quanto tem de cada falha continua sendo carregado, mas nao e mais
    # desenhado: o panorama por rotulo e o **ato 1**, e repeti-lo aqui atrasava a
    # unica coisa que esta secao tem de proprio, que e a serie sem tratamento.
    #
    # O que o grafico protegia — filtrar por um rotulo de 40 leituras e concluir
    # que "nao tem dado" — continua protegido: o `format_func` do filtro abaixo
    # mostra a contagem ao lado de cada nome.
    tabela_rotulos = D.r_rotulos()
    ordem_arquivo = D.r_rotulos_do_arquivo()

    st.caption(f"{len(ordem_arquivo)} valores distintos em `fault`.")

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

