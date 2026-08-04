"""Os manuais de procedimento convertidos em Markdown.

Responde tres perguntas:
  1. Quais manuais existem e o que tem dentro de cada um?
  2. O que falta em cada manual?
  3. Que tipo de falha fica sem manual — ou seja, onde o sistema tera de recusar?
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Documentos", "📄")

st.title("📄 Manuais de Procedimento")
st.caption("Os 6 PDFs da empresa, convertidos para texto e ligados aos tipos de falha.")

st.markdown(
    """
Os manuais chegaram em PDF. PDF e otimo para ler, mas ruim para o computador
procurar dentro. Aqui eles sao convertidos para **Markdown** — texto simples, com
as secoes numeradas preservadas.

Isso importa porque, mais adiante, o sistema precisa **citar a fonte**: nao basta
dizer "alinhe o motor", tem de dizer *"conforme o Doc2, secao 9"*. Para isso, as
secoes precisam estar separadas e identificadas.
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
    with st.expander("Resultado da ultima conversao"):
        st.dataframe(
            res[["documento", "origem", "titulo", "secoes", "ok", "aviso"]],
            hide_index=True,
        )

if not docs:
    st.stop()

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

Sao 17 paginas de imagem, sem texto de verdade dentro — o computador nao consegue
copiar nada dali (extrai 52 caracteres, todos de cabecalho).

A saida normal seria OCR (leitura automatica de imagem), mas isso exige instalar um
programa a parte, o que quebraria a promessa de "baixar o projeto e rodar". Entao o
conteudo foi **transcrito** e guardado ao lado do PDF. A origem fica registrada em
cada arquivo gerado, para ninguem confundir transcricao com extracao automatica.

Como este e justamente o manual de rolamentos — o maior grupo de falhas do arquivo —
vale conferir a transcricao contra o PDF antes da entrega.
"""
    )

# ==========================================================================
# 2. Campos por manual
# ==========================================================================
st.header("2. O que tem dentro de cada manual")

st.markdown(
    f"""
Um bom procedimento de manutencao percorre sempre o mesmo caminho:
**entender o problema → diagnosticar → corrigir → validar → registrar**.

Transformamos esse caminho numa lista de **{len(D.config.CAMPOS_CANONICOS)} itens**.
Cada secao numerada do manual e encaixada num deles pelo titulo.

Isso serve para duas coisas:

1. **Conferir se falta algo** no manual — um procedimento sem a secao de correcao
   nao responde "o que fazer".
2. **Marcar cada pedaco do texto** quando ele for para o sistema de busca. Na hora
   de responder "o que devo fazer?", o sistema prioriza os pedacos marcados como
   *correcao* e *validacao*.
"""
)

escolhido = st.selectbox(
    "Manual",
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
    st.metric("Itens presentes", f"{len(doc['campos'])} de {len(rotulos)}")
with col_d:
    st.metric("Itens em falta", len(rotulos) - len(doc["campos"]), delta_color="inverse")

st.dataframe(
    presentes,
    hide_index=True,
    height=560,
    column_config={
        "campo": "item esperado",
        "presente": st.column_config.CheckboxColumn("tem?"),
        "secoes": st.column_config.TextColumn("secao(oes)"),
        "titulos": st.column_config.TextColumn("como aparece no manual", width="large"),
    },
)

with st.expander(f"Ver as {doc['n_secoes']} secoes de {escolhido}"):
    st.dataframe(
        pd.DataFrame(doc["secoes"])[["numero", "nivel", "titulo", "campo"]],
        hide_index=True,
        height=400,
        column_config={
            "numero": "n",
            "nivel": "nivel",
            "titulo": st.column_config.TextColumn("titulo", width="large"),
            "campo": "item",
        },
    )

with st.expander(f"Ver o texto gerado — {doc['arquivo'].name}"):
    st.code(doc["arquivo"].read_text(encoding="utf-8")[:6000], language="markdown")

# ==========================================================================
# 3. Matriz
# ==========================================================================
st.header("3. Quais itens estao em quais manuais")

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

st.dataframe(
    matriz[["campo"] + colunas_doc + ["documentos_com", "pendente_em"]],
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

st.caption("O mesmo em cores: verde = tem, vermelho = falta.")

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
            legend=alt.Legend(labelExpr="datum.label == 'true' ? 'tem' : 'falta'"),
        ),
        tooltip=["documento", "campo", alt.Tooltip("secoes:N", title="secao(oes)")],
    )
    .properties(height=28 * len(ordem_campos)),
    width="stretch",
)

st.subheader("O que esta faltando")
pendentes = D.r_pendentes()

if pendentes.empty:
    st.success("Todos os manuais cobrem os itens esperados.")
else:
    st.warning(
        f"**{len(pendentes)} itens em falta**, em {pendentes['documento'].nunique()} manuais."
    )
    st.dataframe(
        pendentes[["documento", "campo_ausente", "titulo"]],
        hide_index=True,
        column_config={
            "documento": "manual",
            "campo_ausente": "item que falta",
            "titulo": st.column_config.TextColumn("titulo do manual", width="large"),
        },
    )
    st.markdown(
        """
**Nem toda falta e um problema. Uma delas e.**

O item **Indicadores de monitoramento** falta no manual de desalinhamento (Doc2) e no
de desbalanceamento (Doc3). Os outros quatro manuais tem esse item, e o de rolamentos
chega a listar `Kurtosis`, `Crest Factor` e `RMS` — que sao exatamente colunas do
arquivo de sensores.

Ou seja: para rolamento, correia, polia e rotor inclinado existe uma ponte explicita
entre **o que o sensor mede** e **o que o manual manda acompanhar**. Para
desalinhamento e desbalanceamento, essa ponte precisa ser deduzida.
"""
    )

# ==========================================================================
# 4. Ligacao com a coluna fault
# ==========================================================================
st.header("4. Qual manual atende qual falha")

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

### Por que essa ligacao e feita a mao

Poderiamos deixar o computador procurar o manual mais parecido com a falha. O
problema e que **uma busca por semelhanca sempre devolve alguma coisa** — mesmo
quando nao existe manual nenhum para aquele defeito, ela devolveria o "menos
diferente", e o sistema responderia com confianca sobre algo que nao tem base.

Por isso a ligacao e uma **lista fixa, escrita a mao** e guardada em
`data/fault_map.yaml`. Quando a falha nao esta na lista, o sistema **recusa** em vez
de improvisar.
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

st.dataframe(
    base[["familia", "cobertura", "documento", "titulo_documento", "n_rotulos",
          "n_leituras", "e_problema", "g3_libera"]],
    hide_index=True,
    height=420,
    column_config={
        "familia": "falha",
        "cobertura": "situacao",
        "documento": "manual",
        "titulo_documento": st.column_config.TextColumn("titulo", width="large"),
        "n_rotulos": st.column_config.NumberColumn("nomes", format="%d"),
        "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
        "e_problema": st.column_config.CheckboxColumn("e defeito?"),
        "g3_libera": st.column_config.CheckboxColumn(
            "pode prescrever?",
            help="Se nao, o sistema para antes de consultar o modelo de linguagem."
        ),
    },
)

st.subheader("Lendo o desenho")

lista_sem = ", ".join(f"`{f}`" for f in sem_doc_problema["familia"])
leituras_sem = int(sem_doc_problema["n_leituras"].sum())

st.error(
    f"""
**{len(sem_doc_problema)} defeitos nao tem manual: {lista_sem}.**

Sao {leituras_sem:,} leituras no arquivo — dados existem, procedimento nao.

Para esses casos o sistema vai responder *"Sem documentacao — registre um
documento"* e **nao** vai consultar o modelo de linguagem.

Isso e o comportamento correto, nao uma falha. Um modelo de linguagem sabe falar
sobre ventoinha e falta de fase por conhecimento geral, e produziria uma resposta
convincente. Mas essa resposta nao seria o procedimento **desta empresa**, e nao
haveria fonte para citar. Recusar e melhor que inventar.
""".replace(",", ".")
)

st.warning(
    """
**`eccentric_rotor` fica no meio do caminho (linha tracejada).**

O manual de polias (Doc5) tem uma secao sobre excentricidade. Mas e excentricidade
**de polia**, e o dado aqui e excentricidade **de rotor**. Mesmo fenomeno fisico,
peca diferente.

Marcamos como **cobertura parcial** e **nao** liberamos a prescricao. Aceitar essa
ligacao faria o sistema mandar ajustar a polia quando o problema esta no rotor.

Isso vale a atencao: e a segunda maior familia de falhas do arquivo.
"""
)

st.info(
    """
**`normal`, `teste`, `acelerando` e `motor_desligado` tambem aparecem sem manual —
mas por outro motivo.**

Nao sao defeitos, sao estados da maquina. Nao existe procedimento de correcao para
uma maquina que esta funcionando bem, e o sistema encerra o atendimento antes mesmo
de procurar manual.
"""
)
