"""Ato 6 da narrativa — **os manuais de procedimento**.

Os 6 PDFs da empresa, convertidos para texto e ligados aos tipos de falha. E a
segunda fonte do projeto: o `banner.csv` diz o que a maquina mediu, e estes
documentos dizem o que fazer a respeito. A ponte entre os dois e a coluna
`fault`, via `data/fault_map.yaml`.

Era uma pagina propria (`pages/3_Documentos.py`). Virou secao de `app.py` pelo
mesmo motivo das outras: a analise se le como uma historia so, em vez de telas
soltas que o avaliador precisa costurar de cabeca. O conteudo e o mesmo — a
conversao foi mecanica, e o `st.stop()` virou `return` porque parar aqui nao
pode mais levar as secoes seguintes junto.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D


def render() -> None:
    """Desenha esta secao dentro da narrativa unica."""

    st.header("📄 Manuais de Procedimento", divider="gray")
    st.caption("Os 6 PDFs da empresa, convertidos para texto e ligados aos tipos de falha.")

    st.markdown(
        """
    Os manuais chegaram em PDF e foram convertidos para markdown para melhores resultados. 
    Um dos arquivos precisou primeiro ser convertido para txt. 
    A saida normal seria OCR (leitura automatica de imagem), 
    mas isso exige instalar um programa a parte, o que quebraria a promessa de "baixar o projeto e rodar". Entao o conteudo foi transcrito e guardado ao lado do PDF. A origem fica registrada em cada arquivo gerado, para nao confundir com transcricao com extracao automatica.
    """
    )

    # ==========================================================================
    # 0. Conversao
    # ==========================================================================
    docs = D.r_docs()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if docs:
            st.caption(f"{len(docs)} manuais convertidos, em `{D.config.DOCS_MD_DIR}`")
        else:
            st.warning(
                "Nenhum manual convertido ainda. Clique em **Converter PDFs** para gerar "
                "os arquivos de texto a partir de `data/raw/*.pdf`."
            )
    with col_b:
        if st.button("Converter PDFs", type="primary", width="stretch", key="doc_converter"):
            with st.spinner("Convertendo..."):
                try:
                    resultado = D.converter_pdfs()
                except FileNotFoundError as e:
                    st.error(str(e))
                    return
            st.session_state["conversao"] = resultado
            st.rerun()

    if (res := st.session_state.get("conversao")) is not None:
        with st.expander("Resultado da ultima conversao"):
            st.dataframe(
                res[["documento", "origem", "titulo", "secoes", "ok", "aviso"]],
                hide_index=True,
            )

    if not docs:
        return

    # ==========================================================================
    # 1. Titulos
    # ==========================================================================
    st.header("1. Quais manuais temos")

    tabela_docs = pd.DataFrame(
        [
            {
                "arquivo": d["arquivo"].name,
                "titulo": d["titulo"],
                "secoes": d["n_secoes"],
                "campos": len(d["campos"]),
                "origem_texto": d["origem_texto"],
                "caracteres": d["caracteres"],
            }
            for d in docs
        ]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manuais", len(docs))
    c2.metric("Secoes no total", int(tabela_docs["secoes"].sum()))
    c3.metric("Itens esperados", len(D.config.CAMPOS_CANONICOS))
    c4.metric("Precisaram de transcricao", int((tabela_docs["origem_texto"] == "sidecar").sum()))

    st.dataframe(
        tabela_docs,
        hide_index=True,
        column_config={
            "titulo": st.column_config.TextColumn("titulo", width="large"),
            "secoes": st.column_config.NumberColumn("secoes", format="%d"),
            "campos": st.column_config.NumberColumn(
                "itens presentes", format="%d",
                help=f"De {len(D.config.CAMPOS_CANONICOS)} itens esperados num procedimento.",
            ),
            "origem_texto": st.column_config.TextColumn(
                "de onde veio o texto",
                help="pdf = o proprio arquivo tinha texto. sidecar = o PDF era imagem "
                     "e o conteudo foi transcrito a mao.",
            ),
            "caracteres": st.column_config.NumberColumn("tamanho", format="%d"),
        },
    )

    if (tabela_docs["origem_texto"] == "sidecar").any():
        nomes = tabela_docs.loc[tabela_docs["origem_texto"] == "sidecar", "arquivo"].tolist()
        st.info(
            f"""
    **{', '.join(nomes)} veio escaneado.**
    """
        )

    # ==========================================================================
    # 2. Matriz
    # ==========================================================================
    st.header("2. Quais itens estao em quais manuais")

    matriz = D.r_matriz_campos()
    colunas_doc = [d["documento"] for d in docs]

    st.markdown(
        """
    A tabela cruza os itens esperados (linhas) com os manuais (colunas).

    **A celula mostra o numero da secao, nao um "X".** Isso e proposital: o numero e o
    endereco que o sistema tera de citar na resposta — *"Doc2, secao 9"*. Guardar o
    numero desde agora permite conferir depois se a citacao existe mesmo.
    """
    )

    # Verde = tem, vermelho = falta — as MESMAS cores do mapa logo abaixo. Antes a
    # tabela vinha sem cor nenhuma e o mapa colorido vinha depois, entao o olho
    # tinha de casar duas leituras diferentes da mesma informacao. Agora as duas
    # falam a mesma lingua e a tabela ja se le sozinha.
    VERDE, VERMELHO = "#2d6a4f", "#d1495b"

    def _cor_da_celula(valor) -> str:
        """Fundo da celula conforme o item exista ou nao naquele manual."""
        tem = bool(str(valor).strip())
        return f"background-color: {VERDE if tem else VERMELHO}; color: white;"

    st.dataframe(
        matriz[["campo"] + colunas_doc + ["documentos_com", "pendente_em"]]
        .style.map(_cor_da_celula, subset=colunas_doc),
        hide_index=True,
        height=560,
        column_config={
            "campo": st.column_config.TextColumn("item esperado", width="medium"),
            "documentos_com": st.column_config.NumberColumn(
                "esta em", format=f"%d/{len(docs)}"
            ),
            "pendente_em": st.column_config.NumberColumn("falta em", format="%d"),
        },
    )

    st.caption(
        "Verde = tem, vermelho = falta. A celula mostra o numero da secao, nao um "
        "'X' — o numero e o endereco que a resposta tera de citar."
    )

    longo = matriz.melt(
        id_vars=["campo", "chave"], value_vars=colunas_doc,
        var_name="documento", value_name="secoes",
    )
    longo["presente"] = longo["secoes"] != ""
    ordem_campos = matriz["campo"].tolist()

    # ==========================================================================
    # 3. Ligacao com a coluna fault
    # ==========================================================================
    st.header("3. Qual manual atende qual falha")

    cobertura = D.r_cobertura()
    familias = D.r_familias_banner()

    base = cobertura.merge(familias, on="familia", how="left")
    base["n_leituras"] = base["n_leituras"].fillna(0).astype(int)
    base["n_rotulos"] = base["n_rotulos"].fillna(0).astype(int)

    n_doc = int((base["cobertura"] == "documentado").sum())
    n_par = int((base["cobertura"] == "parcial").sum())
    n_sem = int((base["cobertura"] == "sem_documento").sum())
    sem_doc_problema = base[(base["cobertura"] == "sem_documento") & (base["e_problema"])]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Falhas com manual", n_doc)
    c2.metric("Cobertura parcial", n_par)
    c3.metric("Sem manual", n_sem)
    c4.metric("Defeitos sem manual", len(sem_doc_problema), delta_color="inverse")

    st.markdown(
        """
    O desenho abaixo liga cada **manual** (esquerda) as **falhas que ele atende**
    (direita). O tamanho do circulo e proporcional ao numero de leituras daquela falha
    no arquivo de sensores.
    """
    )

    # --- diagrama --------------------------------------------------------------
    # Altair nao tem layout de grafo: calculamos as coordenadas na mao, documentos
    # numa coluna a esquerda e familias a direita, com uma linha por par.
    CORES = {"documentado": "#2d6a4f", "parcial": "#e2a03f", "sem_documento": "#d1495b"}

    esq = (
        base[base["documento"].notna()][["documento", "titulo_documento"]]
        .drop_duplicates()
        .sort_values("documento")
        .reset_index(drop=True)
    )
    esq["y"] = range(len(esq))
    # Espalha os documentos na mesma altura total ocupada pelas familias, senao a
    # coluna curta fica espremida no topo.
    esq["y"] = (esq["y"] + 0.5) * (len(base) / max(len(esq), 1))
    esq["x"] = 0.0

    dir_ = base.sort_values(["cobertura", "n_leituras"], ascending=[True, False]).reset_index(
        drop=True
    )
    dir_["y"] = [float(i) for i in range(len(dir_))]
    dir_["x"] = 1.0

    arestas = dir_[dir_["documento"].notna()].merge(
        esq[["documento", "x", "y"]].rename(columns={"x": "x_doc", "y": "y_doc"}),
        on="documento",
        how="left",
    )

    linhas = (
        alt.Chart(arestas)
        .mark_rule(strokeWidth=2, opacity=0.55)
        .encode(
            x=alt.X("x_doc:Q", scale=alt.Scale(domain=[-0.35, 1.55]), axis=None),
            y=alt.Y("y_doc:Q", scale=alt.Scale(domain=[-1, len(dir_)]), axis=None),
            x2="x:Q",
            y2="y:Q",
            color=alt.Color("cobertura:N",
                            scale=alt.Scale(domain=list(CORES), range=list(CORES.values())),
                            legend=None),
            strokeDash=alt.condition(
                alt.datum.cobertura == "parcial", alt.value([5, 4]), alt.value([1, 0])
            ),
        )
    )

    pontos_doc = (
        alt.Chart(esq)
        .mark_point(size=260, filled=True, color="#34495e")
        .encode(x="x:Q", y="y:Q",
                tooltip=[alt.Tooltip("documento:N"),
                         alt.Tooltip("titulo_documento:N", title="titulo")])
    )

    texto_doc = (
        alt.Chart(esq)
        .mark_text(align="right", dx=-14, fontSize=12, fontWeight="bold", color="#34495e")
        .encode(x="x:Q", y="y:Q", text="documento:N")
    )

    pontos_fam = (
        alt.Chart(dir_)
        .mark_point(filled=True, opacity=0.9)
        .encode(
            x="x:Q",
            y="y:Q",
            size=alt.Size("n_leituras:Q", scale=alt.Scale(range=[60, 900]), legend=None),
            color=alt.Color("cobertura:N",
                            scale=alt.Scale(domain=list(CORES), range=list(CORES.values())),
                            legend=alt.Legend(title="situacao", orient="bottom")),
            tooltip=["familia", "cobertura", "documento", "n_rotulos", "n_leituras"],
        )
    )

    texto_fam = (
        alt.Chart(dir_)
        .mark_text(align="left", dx=20, fontSize=12)
        .encode(
            x="x:Q",
            y="y:Q",
            text="familia:N",
            color=alt.Color("cobertura:N",
                            scale=alt.Scale(domain=list(CORES), range=list(CORES.values())),
                            legend=None),
        )
    )

    st.altair_chart(
        (linhas + pontos_doc + texto_doc + pontos_fam + texto_fam)
        .properties(height=max(420, 34 * len(dir_)))
        .configure_view(stroke=None),
        width="stretch",
    )
