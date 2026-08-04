"""Procedimentos convertidos em Markdown: titulos, campos e cobertura.

Responde tres perguntas, nesta ordem:
  1. Quais documentos existem e o que cada um tem dentro?
  2. Que campos faltam em cada um?
  3. Que familia de falha do banner.csv fica sem documento — ou seja, onde o
     guardrail G3 vai recusar prescricao?
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Documentos", "📄")

st.title("📄 Documentos de Procedimento")
st.caption("PDFs convertidos em Markdown, com a cobertura de campos e a ligacao com `fault`.")

# ==========================================================================
# 0. Conversao
# ==========================================================================
docs = D.r_docs()

col_a, col_b = st.columns([3, 1])
with col_a:
    if docs:
        st.caption(f"{len(docs)} documentos em `{D.config.DOCS_MD_DIR}`")
    else:
        st.warning(
            "Nenhum Markdown gerado ainda. Clique em **Converter PDFs** para "
            "extrair os procedimentos de `data/raw/*.pdf`."
        )
with col_b:
    if st.button("Converter PDFs", type="primary", width="stretch"):
        with st.spinner("Convertendo..."):
            try:
                resultado = D.converter_pdfs()
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
        st.session_state["conversao"] = resultado
        st.rerun()

if (res := st.session_state.get("conversao")) is not None:
    with st.expander("Resultado da ultima conversao", expanded=False):
        st.dataframe(
            res[["documento", "origem", "titulo", "secoes", "ok", "aviso"]],
            hide_index=True,
        )

if not docs:
    st.stop()

# ==========================================================================
# 1. Titulos
# ==========================================================================
st.header("1. Titulos dos arquivos `.md`")

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
c1.metric("Documentos", len(docs))
c2.metric("Secoes totais", int(tabela_docs["secoes"].sum()))
c3.metric("Campos canonicos", len(D.config.CAMPOS_CANONICOS))
c4.metric("Extraidos por OCR/transcricao", int((tabela_docs["origem_texto"] == "sidecar").sum()))

st.dataframe(
    tabela_docs,
    hide_index=True,
    column_config={
        "titulo": st.column_config.TextColumn("titulo", width="large"),
        "secoes": st.column_config.NumberColumn("secoes", format="%d"),
        "campos": st.column_config.NumberColumn("campos", format="%d",
                                                help="de 15 campos canonicos"),
        "origem_texto": st.column_config.TextColumn(
            "origem do texto",
            help="pdf = camada de texto do arquivo; sidecar = transcricao manual "
                 "de um PDF digitalizado",
        ),
        "caracteres": st.column_config.NumberColumn("caracteres", format="%d"),
    },
)

if (tabela_docs["origem_texto"] == "sidecar").any():
    nomes = tabela_docs.loc[tabela_docs["origem_texto"] == "sidecar", "arquivo"].tolist()
    st.info(
        f"**{', '.join(nomes)}** veio de um PDF **digitalizado**, sem camada de texto — "
        "17 paginas de imagem. `pypdf` extrai 52 caracteres dele, todos de cabecalho. "
        "O conteudo foi transcrito para `data/raw/Doc1.txt` e o conversor usa esse "
        "sidecar quando o PDF nao tem texto. A origem fica registrada no front matter "
        "de cada `.md`, para ninguem confundir com extracao automatica."
    )

# ==========================================================================
# 2. Campos por artigo
# ==========================================================================
st.header("2. Campos por artigo")

st.markdown(
    """
Um procedimento de manutencao deveria percorrer o fluxo **entender → diagnosticar
→ corrigir → validar → registrar**. Os 15 campos abaixo sao esse fluxo. Cada secao
numerada do documento e classificada num campo pelo titulo.

Os campos importam alem da conferencia editorial: na Parte 4 eles viram o
**metadado do chunk**, e a pergunta prescritiva prioriza `correcao` e `validacao`.
Documento sem esses campos nao gera resposta util, mesmo indexado.
"""
)

escolhido = st.selectbox(
    "Documento",
    [d["documento"] for d in docs],
    format_func=lambda x: f"{x} — {next(d['titulo'] for d in docs if d['documento'] == x)[:70]}",
)
doc = next(d for d in docs if d["documento"] == escolhido)

rotulos = {c: r for c, r, _ in D.config.CAMPOS_CANONICOS}

presentes = pd.DataFrame(
    [
        {
            "campo": rotulos[chave],
            "secoes": ", ".join(s["numero"] for s in doc["secoes"] if s["campo"] == chave),
            "titulos": " | ".join(
                s["titulo"] for s in doc["secoes"] if s["campo"] == chave
            ),
            "presente": chave in doc["campos"],
        }
        for chave in rotulos
    ]
)

col_e, col_d = st.columns([1, 1])
with col_e:
    st.metric("Campos presentes", f"{len(doc['campos'])} / {len(rotulos)}")
with col_d:
    faltando = len(rotulos) - len(doc["campos"])
    st.metric("Campos pendentes", faltando, delta_color="inverse")

st.dataframe(
    presentes,
    hide_index=True,
    height=560,
    column_config={
        "presente": st.column_config.CheckboxColumn("tem?"),
        "secoes": st.column_config.TextColumn("secao(oes)"),
        "titulos": st.column_config.TextColumn("titulo no documento", width="large"),
    },
)

with st.expander(f"Todas as {doc['n_secoes']} secoes de {escolhido}"):
    st.dataframe(
        pd.DataFrame(doc["secoes"])[["numero", "nivel", "titulo", "campo"]],
        hide_index=True,
        height=400,
    )

with st.expander(f"Markdown gerado — {doc['arquivo'].name}"):
    st.code(doc["arquivo"].read_text(encoding="utf-8")[:6000], language="markdown")

# ==========================================================================
# 3. Matriz campo x documento
# ==========================================================================
st.header("3. Que campo esta em que arquivo")

matriz = D.r_matriz_campos()
colunas_doc = [d["documento"] for d in docs]

st.caption(
    "A celula traz o **numero da secao**, nao um 'X'. O numero e o endereco da "
    "citacao que o LLM tera de produzir na Parte 5 (`Doc2, secao 9`), e o "
    "guardrail G5 confere se a citacao existe mesmo."
)

st.dataframe(
    matriz[["campo"] + colunas_doc + ["documentos_com", "pendente_em"]],
    hide_index=True,
    height=560,
    column_config={
        "campo": st.column_config.TextColumn("campo", width="medium"),
        "documentos_com": st.column_config.NumberColumn(
            "presente em", format="%d/6", help="Quantos dos 6 documentos tem o campo"
        ),
        "pendente_em": st.column_config.NumberColumn("pendente em", format="%d"),
    },
)

# Heatmap da mesma matriz — le mais rapido que a tabela.
longo = matriz.melt(
    id_vars=["campo", "chave"], value_vars=colunas_doc,
    var_name="documento", value_name="secoes",
)
longo["presente"] = longo["secoes"] != ""
ordem_campos = matriz["campo"].tolist()

st.altair_chart(
    alt.Chart(longo)
    .mark_rect(stroke="white", strokeWidth=2)
    .encode(
        x=alt.X("documento:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("campo:N", title=None, sort=ordem_campos),
        color=alt.Color(
            "presente:N",
            title=None,
            scale=alt.Scale(domain=[True, False], range=["#2d6a4f", "#d1495b"]),
            legend=alt.Legend(labelExpr="datum.label == 'true' ? 'presente' : 'pendente'"),
        ),
        tooltip=["documento", "campo", alt.Tooltip("secoes:N", title="secao(oes)")],
    )
    .properties(height=28 * len(ordem_campos)),
    width="stretch",
)

st.subheader("Pendencias")
pendentes = D.r_pendentes()

if pendentes.empty:
    st.success("Todos os documentos cobrem os 15 campos canonicos.")
else:
    st.warning(
        f"**{len(pendentes)} pendencias** em {pendentes['documento'].nunique()} documentos."
    )
    st.dataframe(
        pendentes[["documento", "campo_ausente", "titulo"]],
        hide_index=True,
        column_config={
            "campo_ausente": "campo ausente",
            "titulo": st.column_config.TextColumn("documento", width="large"),
        },
    )
    st.markdown(
        """
Nem toda ausencia e defeito. **Indicadores de monitoramento** falta no Doc2
(desalinhamento) e no Doc3 (desbalanceamento), enquanto Doc1, Doc4, Doc5 e Doc6
listam quais grandezas acompanhar. Como o Doc1 nomeia exatamente `Kurtosis`,
`Crest Factor` e `RMS global` — colunas que existem no `banner.csv` — a ausencia
nos outros dois e uma lacuna real para o cruzamento entre sensor e procedimento,
nao uma questao de estilo.
"""
    )

# ==========================================================================
# 4. Diagrama: problema do artigo x coluna `fault`
# ==========================================================================
st.header("4. Problema do artigo × coluna `fault` do banner")

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
c1.metric("Familias documentadas", n_doc)
c2.metric("Cobertura parcial", n_par)
c3.metric("Sem documento", n_sem)
c4.metric("Defeitos sem documento", len(sem_doc_problema), delta_color="inverse")

st.markdown(
    """
O diagrama liga **cada procedimento** (esquerda) a **cada familia de `fault`**
(direita). O tamanho do circulo da familia e proporcional ao numero de leituras
no `banner.csv`.

Esta ligacao e a espinha do sistema: pelo principio 1 do projeto, numero nunca e
comparado com texto. O evento resolve para um rotulo, o rotulo resolve para uma
familia, e a familia resolve para um documento — por `SELECT`, nunca por
similaridade semantica.
"""
)

# --- montagem do diagrama bipartido ---------------------------------------
# Altair nao tem layout de grafo. Calculamos as coordenadas na mao: documentos
# numa coluna a esquerda, familias a direita, e uma linha por par.
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
            tooltip=[alt.Tooltip("documento:N"), alt.Tooltip("titulo_documento:N", title="titulo")])
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
                        legend=alt.Legend(title="cobertura", orient="bottom")),
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

st.dataframe(
    base[["familia", "cobertura", "documento", "titulo_documento", "n_rotulos",
          "n_leituras", "e_problema", "g3_libera"]],
    hide_index=True,
    height=420,
    column_config={
        "titulo_documento": st.column_config.TextColumn("titulo", width="large"),
        "n_rotulos": st.column_config.NumberColumn("rotulos", format="%d"),
        "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
        "e_problema": st.column_config.CheckboxColumn("e defeito?"),
        "g3_libera": st.column_config.CheckboxColumn(
            "G3 libera?", help="Se falso, o fluxo prescritivo para antes de chamar o LLM"
        ),
    },
)

st.subheader("Leitura do diagrama")

lista_sem = ", ".join(f"`{f}`" for f in sem_doc_problema["familia"])
leituras_sem = int(sem_doc_problema["n_leituras"].sum())

st.error(
    f"**{len(sem_doc_problema)} familias de defeito nao tem procedimento: {lista_sem}.** "
    f"Somam {leituras_sem:,} leituras no banner.csv. ".replace(",", ".")
    + "Para elas o guardrail **G3** encerra o fluxo com a mensagem padronizada "
    "*'Sem documentacao — registre um documento'*, sem chamar o LLM. Nao e falha "
    "do sistema: e o comportamento correto. Inventar procedimento a partir do "
    "conhecimento do modelo seria pior que recusar."
)

st.warning(
    "**`eccentric_rotor` tem cobertura apenas parcial (linha tracejada).** A secao "
    "3.1 do Doc5 descreve excentricidade, mas **de polia** — e o rotulo do banner e "
    "excentricidade **de rotor**. Mesmo fenomeno, componente diferente. Tratamos "
    "como *parcial* e o G3 **nao** libera: aceitar essa ligacao faria o sistema "
    "prescrever ajuste de polia para um problema de rotor. A decisao final vai para "
    "o `fault_map.yaml`, e esta e exatamente a divergencia que o cruzamento entre "
    "assinatura e documento deveria expor."
)

st.info(
    "As familias `normal`, `teste`, `acelerando` e `motor_desligado` tambem aparecem "
    "sem documento, mas por outro motivo: sao **estados**, nao defeitos. O guardrail "
    "**G2** encerra o fluxo antes do G3 — nao ha acao corretiva para uma maquina que "
    "esta operando normalmente."
)
