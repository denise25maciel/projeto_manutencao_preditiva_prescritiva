"""Aba 3 da tela de classificacao — **ver o modelo rodar, e o laudo do teste**.

As abas 1 e 2 argumentam: explicam o que e um exemplo e o que a acuracia
significa. Esta nao argumenta — ela **executa**. Clica-se em um botao, as cinco
etapas rodam na ordem, cada uma diz quanto levou e o que produziu, e no fim sai
o relatorio dos testes.

**Aqui nada e cacheado, e essa e a escolha central da aba.** Todas as outras
telas passam por `_dados.py`, onde `@st.cache_data` guarda o resultado — o que e
certo la e seria mentira aqui: na segunda execucao os tempos apareceriam
proximos de zero, e um painel de execucao que nao mede execucao e enfeite.
Executar de novo executa de novo, e leva o mesmo tempo.

O que a aba mostra por etapa nao e barra de progresso: e o **dado mudando de
forma**. A previa da etapa 1 tem uma linha por leitura do sensor; a da etapa 3
tem uma linha por janela, com 80 colunas de numero. Da para ver a transformacao
acontecer, em vez de acreditar que aconteceu.

A sequencia das etapas e o que cada uma produz sao decisao de dominio e moram
em `mp.classificacao.execucao`. Este arquivo so desenha o que aquele gerador
entrega — e o mesmo relatorio sai no terminal com
`python -m mp.classificacao.execucao`.
"""

from __future__ import annotations

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D
from mp import config
from mp.classificacao import acerto_por_familia, matriz_de_confusao
from mp.classificacao.execucao import executar_pipeline, relatorio_texto

CHAVE_ETAPAS = "clf_exec_etapas"
CHAVE_CONFIG = "clf_exec_config"


def render() -> None:
    """Desenha o painel de execucao."""

    st.markdown(
        """
As duas abas anteriores explicam. Esta **roda**: as cinco etapas em ordem, cada
uma cronometrada, mostrando o que entregou — e no fim, o laudo dos testes.

Nada aqui vem de cache. Nas outras telas o cache e o certo; aqui seria mentira,
porque a segunda execucao mostraria tempos proximos de zero e o painel deixaria
de medir o que se propoe a medir. **Executar de novo executa de novo.**
        """
    )

    # ----------------------------------------------------------------------
    # Os parametros da corrida
    # ----------------------------------------------------------------------
    with st.expander("Parametros desta execucao", expanded=False):
        st.caption(
            "Os padroes sao os do `config.py`, cada um com a medicao que o "
            "justifica no comentario. Mexer aqui muda so esta execucao — o "
            "arquivo continua como esta."
        )
        c1, c2, c3, c4 = st.columns(4)
        tamanho = c1.number_input(
            "Janela (leituras)", min_value=10, max_value=500,
            value=config.CLF_JANELA_TAMANHO, step=10,
            help="Acima de 50 o descarte de eventos salta de 3% para 71%.",
        )
        n_arvores = c2.number_input(
            "Arvores", min_value=10, max_value=1000,
            value=config.CLF_N_ARVORES, step=50,
        )
        n_folds = c3.number_input(
            "Folds", min_value=2, max_value=10, value=config.CLF_N_FOLDS,
        )
        incluir_regime = c4.checkbox(
            "Incluir rpm e temperatura", value=False,
            help="Regime de operacao. Fora por padrao — ver a secao 7 da aba 2.",
        )

    c1, c2 = st.columns([1, 3])
    executar = c1.button("▶ Executar o pipeline", type="primary", width="stretch")
    c2.caption(
        f"Cinco etapas, {2 * int(n_folds) + 1} florestas de {int(n_arvores)} "
        "arvores no total. Leva de 1 a 2 minutos — e o tempo real, sem atalho."
    )

    if executar:
        _rodar(int(tamanho), bool(incluir_regime), int(n_arvores), int(n_folds))

    etapas = st.session_state.get(CHAVE_ETAPAS)
    if not etapas:
        st.info(
            "Nenhuma execucao ainda. O botao acima roda o pipeline inteiro a "
            "partir do CSV bruto — nada fica pre-calculado."
        )
        return

    if not executar:
        # Redesenho apos um rerun: as etapas ja rodaram, so voltam para a tela.
        cfg = st.session_state.get(CHAVE_CONFIG, {})
        st.caption(
            f"Ultima execucao: {cfg.get('quando', '?')} — janela "
            f"{cfg.get('tamanho')}, {cfg.get('n_arvores')} arvores, "
            f"{cfg.get('n_folds')} folds, regime "
            f"{'incluido' if cfg.get('incluir_regime') else 'de fora'}."
        )
        _linha_do_tempo(etapas)
        for etapa in etapas:
            _desenhar_etapa(etapa)

    st.divider()
    _relatorio(etapas)


# --------------------------------------------------------------------------
# Execucao
# --------------------------------------------------------------------------


def _rodar(tamanho: int, incluir_regime: bool, n_arvores: int, n_folds: int) -> None:
    """Consome o gerador, desenhando cada etapa assim que ela termina.

    O gerador existe para isto: entregar a etapa concluida enquanto a seguinte
    ainda roda. Uma lista pronta no fim daria a mesma informacao um minuto
    depois, com a tela parada no meio e ninguem sabendo se travou.
    """
    st.session_state[CHAVE_ETAPAS] = []
    st.session_state[CHAVE_CONFIG] = {
        "quando": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "tamanho": tamanho,
        "n_arvores": n_arvores,
        "n_folds": n_folds,
        "incluir_regime": incluir_regime,
    }

    barra = st.progress(0.0, text="Iniciando...")
    etapas = []

    try:
        for etapa in executar_pipeline(
            tamanho=tamanho,
            incluir_regime=incluir_regime,
            n_arvores=n_arvores,
            n_folds=n_folds,
        ):
            etapas.append(etapa)
            barra.progress(
                etapa.numero / etapa.total,
                text=f"{etapa.rotulo} — {etapa.segundos:.1f}s",
            )
            _desenhar_etapa(etapa, expandida=True)
    except FileNotFoundError as erro:
        barra.empty()
        D.aviso_csv_ausente(erro)
    except Exception as erro:  # a tela nao pode morrer com um traceback cru
        barra.empty()
        st.error(f"**A execucao parou na etapa {len(etapas) + 1}.**\n\n`{erro}`")
        st.session_state[CHAVE_ETAPAS] = etapas
        return

    barra.progress(1.0, text=f"Concluido em {sum(e.segundos for e in etapas):.1f}s")
    st.session_state[CHAVE_ETAPAS] = etapas


# --------------------------------------------------------------------------
# Desenho
# --------------------------------------------------------------------------


def _desenhar_etapa(etapa, expandida: bool = False) -> None:
    """Uma etapa concluida: tempo, o que fez, os numeros e a previa do dado."""
    with st.expander(
        f"**{etapa.rotulo}** · {etapa.segundos:.1f}s", expanded=expandida
    ):
        st.markdown(etapa.o_que_faz)
        st.caption(f"Roda em `{etapa.onde}`")

        if etapa.resumo:
            colunas = st.columns(len(etapa.resumo))
            for coluna, (chave, valor) in zip(colunas, etapa.resumo.items()):
                coluna.metric(chave, valor)

        if etapa.amostra is not None and not etapa.amostra.empty:
            st.caption("O que saiu desta etapa:")
            st.dataframe(etapa.amostra, width="stretch", hide_index=True)


def _linha_do_tempo(etapas: list) -> None:
    """Onde o tempo foi gasto. Uma barra por etapa, na ordem."""
    tabela = pd.DataFrame(
        [
            {"etapa": f"{e.numero}. {e.nome}", "segundos": e.segundos,
             "ordem": e.numero}
            for e in etapas
        ]
    )
    total = tabela["segundos"].sum()
    tabela["fatia"] = tabela["segundos"] / total

    st.altair_chart(
        alt.Chart(tabela)
        .mark_bar()
        .encode(
            x=alt.X("segundos:Q", title="segundos"),
            y=alt.Y("etapa:N", sort=alt.SortField("ordem"), title=None),
            tooltip=["etapa", alt.Tooltip("segundos:Q", format=".1f"),
                     alt.Tooltip("fatia:Q", format=".0%")],
        )
        .properties(height=28 * len(tabela) + 40),
        width="stretch",
    )
    st.caption(
        f"Tempo total: **{total:.1f}s**. A validacao domina porque treina "
        "florestas novas a cada fold — e a unica forma de a nota ser honesta."
    )


# --------------------------------------------------------------------------
# O laudo
# --------------------------------------------------------------------------


def _relatorio(etapas: list) -> None:
    """O resultado dos testes, com o que foi medido e o que ele significa."""
    resultado = next((e.resultado for e in reversed(etapas) if e.resultado), None)
    if resultado is None:
        st.warning(
            "A execucao nao chegou a etapa de validacao, entao nao ha "
            "resultado de teste a mostrar."
        )
        return

    st.header("Resultado dos testes", divider="gray")

    aleatoria = resultado["estrategias"]["aleatoria"]
    por_evento = resultado["estrategias"]["por_evento"]
    base = resultado["linha_de_base"]

    st.caption(
        f"janela {resultado['janela']} leituras · {resultado['n_arvores']} "
        f"arvores · {resultado['n_folds']} folds · "
        f"{resultado['n_amostras']:,} amostras · {resultado['n_eventos']} "
        f"eventos · {resultado['n_familias']} familias"
        .replace(",", ".")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acuracia honesta", f"{por_evento['acuracia']:.1%}",
              delta=f"±{por_evento['desvio']:.1%}", delta_color="off")
    c2.metric("Acuracia inflada", f"{aleatoria['acuracia']:.1%}",
              delta=f"{aleatoria['eventos_vazados']} eventos vazados",
              delta_color="inverse")
    c3.metric("Inflacao", f"{resultado['inflacao']:.1%}",
              help="A distancia entre as duas. E a medida do quanto o modelo cola.")
    c4.metric("Linha de base", f"{base['maioria']:.1%}",
              help=f"Responder sempre `{base['familia_mais_comum']}`, sem olhar "
                   "dado nenhum.")

    veredito_ok = por_evento["eventos_vazados"] == 0
    c1, c2 = st.columns(2)
    with c1:
        st.error(
            f"**Sorteando amostras — {aleatoria['acuracia']:.1%}**\n\n"
            f"{aleatoria['eventos_vazados']} eventos apareceram no treino e no "
            "teste ao mesmo tempo. Este numero nao pode ser reportado."
        )
    with c2:
        st.success(
            f"**Sorteando eventos — {por_evento['acuracia']:.1%}**\n\n"
            f"{por_evento['eventos_vazados']} eventos nos dois lados. "
            f"Acerta {por_evento['acuracia'] / base['maioria']:.1f}x a linha de "
            "base — ha sinal real, e menos do que o outro numero sugeria."
        )

    if not veredito_ok:
        st.warning(
            "A estrategia por evento registrou vazamento, o que nao deveria "
            "acontecer por construcao. Vale investigar antes de usar o numero."
        )

    st.markdown("**Cada fold, nas duas estrategias**")
    folds = _tabela_de_folds(resultado)
    st.dataframe(
        folds,
        width="stretch",
        hide_index=True,
        column_config={
            "acuracia": st.column_config.ProgressColumn(
                "acuracia", format="%.1f%%", min_value=0, max_value=1),
        },
    )

    st.altair_chart(
        alt.Chart(folds)
        .mark_line(point=True)
        .encode(
            x=alt.X("fold:O", title="fold"),
            y=alt.Y("acuracia:Q", title="acuracia",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "estrategia:N", title=None,
                scale=alt.Scale(
                    domain=["sorteando amostras", "sorteando eventos"],
                    range=["#d1495b", "#2a9d8f"]),
            ),
            tooltip=["estrategia", "fold", alt.Tooltip("acuracia:Q", format=".1%")],
        )
        .properties(height=260),
        width="stretch",
    )
    st.caption(
        "As duas linhas nunca se aproximam, e a de baixo balanca muito mais. As "
        "duas coisas tem a mesma causa: cada fold honesto testa em poucos "
        "eventos, e eventos diferentes sao difíceis de formas diferentes."
    )

    if resultado["classes_raras"]:
        st.info(
            "**Fora da validacao:** "
            f"`{'`, `'.join(resultado['classes_raras'])}` — menos de "
            f"{resultado['n_folds']} eventos cada, e nao da para reparti-las em "
            f"{resultado['n_folds']} partes. E resultado, nao detalhe: a bancada "
            "mediu esses casos poucas vezes."
        )

    st.markdown("**Onde o acerto se concentra, e com quem cada familia e confundida**")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(
            acerto_por_familia(por_evento),
            width="stretch",
            hide_index=True,
            column_config={
                "acuracia": st.column_config.ProgressColumn(
                    "acuracia", format="%.1f%%", min_value=0, max_value=1),
            },
        )
    with c2:
        st.altair_chart(
            heatmap_confusao(matriz_de_confusao(por_evento)),
            width="stretch",
        )

    st.divider()

    texto = relatorio_texto(etapas)
    c1, c2 = st.columns([1, 3])
    c1.download_button(
        "Baixar o relatorio (.txt)",
        data=texto,
        file_name=f"relatorio_classificacao_{datetime.now():%Y%m%d_%H%M}.txt",
        mime="text/plain",
        width="stretch",
    )
    c2.caption(
        "O arquivo e exatamente o que `python -m mp.classificacao.execucao` "
        "imprime — uma formatacao so para os dois, para nao existir versao da "
        "tela divergindo da versao do terminal."
    )

    with st.expander("Ver o relatorio em texto"):
        st.code(texto, language="text")


def _tabela_de_folds(resultado: dict) -> pd.DataFrame:
    nomes = {"aleatoria": "sorteando amostras", "por_evento": "sorteando eventos"}
    linhas = []
    for chave, dados in resultado["estrategias"].items():
        for f in dados["folds"]:
            linhas.append(
                {
                    "estrategia": nomes[chave],
                    "fold": f["fold"],
                    "amostras treino": f["n_treino"],
                    "amostras teste": f["n_teste"],
                    "eventos teste": f["eventos_teste"],
                    "eventos nos dois lados": f["eventos_vazados"],
                    "acuracia": f["acuracia"],
                }
            )
    return pd.DataFrame(linhas)


def heatmap_confusao(confusao: pd.DataFrame) -> alt.Chart:
    """Matriz de confusao normalizada por linha: o que era x o que o modelo disse.

    Cada linha soma 100%, entao a diagonal e o acerto daquela familia. Somar os
    folds e legitimo porque cada amostra e testada exatamente uma vez.
    """
    longo = (
        confusao.reset_index(names="verdadeira")
        .melt(id_vars="verdadeira", var_name="prevista", value_name="fracao")
    )
    ordem = list(confusao.index)

    base = alt.Chart(longo).encode(
        x=alt.X("prevista:N", sort=ordem, title="o modelo disse",
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("verdadeira:N", sort=ordem, title="era"),
    )

    celulas = base.mark_rect().encode(
        color=alt.Color("fracao:Q", title="fracao",
                        scale=alt.Scale(scheme="blues", domain=[0, 1])),
        tooltip=["verdadeira", "prevista", alt.Tooltip("fracao:Q", format=".1%")],
    )

    # So o valor relevante vira texto: a matriz tem 169 celulas e a maioria e
    # zero, que escrito por extenso vira ruido visual.
    rotulos = base.mark_text(fontSize=9).encode(
        text=alt.condition(alt.datum.fracao >= 0.05,
                           alt.Text("fracao:Q", format=".0%"), alt.value("")),
        color=alt.condition(alt.datum.fracao > 0.5,
                            alt.value("white"), alt.value("#333")),
    )

    return (celulas + rotulos).properties(height=380)
