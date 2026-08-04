"""Como os dados chegaram: campos vazios, ritmo da coleta, colunas repetidas,
leituras duplicadas e valores fora do normal.

Nada e corrigido nesta tela. Esta etapa descreve; a proxima trata.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Qualidade dos Dados", "🧪")

st.title("🧪 Qualidade dos Dados")
st.caption("O que veio certo e o que veio torto no arquivo. Nada e alterado aqui.")

try:
    resumo = D.r_resumo()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

nulos = D.r_nulos()
janela = D.r_janela()
amostragem = D.r_amostragem()
constantes = D.r_constantes()
redundantes = D.r_redundantes()
duplicatas = D.r_duplicatas()
tempos_dup = D.r_tempos_duplicados()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Leituras", f"{resumo['linhas']:,}".replace(",", "."))
c2.metric("Campos vazios", f"{resumo['celulas_nulas']:,}".replace(",", "."))
c3.metric(
    "Leituras repetidas",
    f"{duplicatas['total']:,}".replace(",", "."),
    f"{duplicatas['pct']}% do total",
    delta_color="inverse",
)
c4.metric("Colunas descartaveis", len(D.r_descartar()))

st.divider()

# ==========================================================================
# 1. Campos vazios
# ==========================================================================
st.header("1. Campos vazios")

total_nulos = int(nulos["nulos"].sum())
if total_nulos == 0:
    st.success(
        f"**Nao ha nenhum campo vazio.** Todas as {resumo['colunas']} colunas estao "
        f"preenchidas nas {resumo['linhas']:,} linhas.".replace(",", ".")
    )
    st.warning(
        """
**Isso nao quer dizer que nao falta dado.**

Um sensor que nao conseguiu ler pode ter gravado `0.0` em vez de deixar o campo em
branco. Do ponto de vista do arquivo, o campo esta preenchido; na pratica, e um
dado que nao existe.

Esses casos aparecem nas secoes seguintes, olhando para valores estranhos e colunas
com poucos valores diferentes.
"""
    )
else:
    st.altair_chart(
        alt.Chart(nulos[nulos["nulos"] > 0])
        .mark_bar(color="#d1495b")
        .encode(
            x=alt.X("pct_nulos:Q", title="% vazio"),
            y=alt.Y("coluna:N", sort="-x", title=None),
            tooltip=["coluna", "nulos", "pct_nulos"],
        ),
        width="stretch",
    )

st.caption(
    "A coluna **valores diferentes** ajuda a achar problema: uma coluna com pouquissimos "
    "valores distintos em 166 mil linhas provavelmente nao e uma medida continua."
)
st.dataframe(
    nulos[["coluna", "tipo", "nulos", "distintos"]],
    hide_index=True,
    height=330,
    column_config={
        "nulos": st.column_config.NumberColumn("vazios", format="%d"),
        "distintos": st.column_config.NumberColumn("valores diferentes", format="%d"),
    },
)

st.divider()

# ==========================================================================
# 2. Ritmo da coleta
# ==========================================================================
st.header("2. Ritmo e continuidade da coleta")

st.markdown(
    """
Aqui olhamos o **tempo entre uma leitura e a seguinte**, depois de colocar tudo em
ordem de data.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Ritmo tipico", f"{amostragem['intervalo_mediano_s']:.0f} s",
    help="Tempo mais comum entre duas leituras seguidas."
)
c2.metric(
    "Leituras nesse ritmo", f"{amostragem['pct_na_cadencia']:.0f}%",
    help="Quantas leituras respeitam esse intervalo, com folga de 0,25 s."
)
c3.metric(
    "Pausas longas", amostragem["cortes"],
    help="Quantas vezes passou mais de 1 minuto sem leitura."
)
c4.metric(
    "Maior pausa", f"{amostragem['maior_gap_horas']:.0f} h",
    help="A maior interrupcao entre duas leituras."
)

st.markdown(
    f"""
**O que esses numeros dizem:**

- O sensor grava uma leitura a cada **{amostragem['intervalo_mediano_s']:.0f} segundos**,
  e {amostragem['pct_na_cadencia']:.0f}% das leituras seguem esse ritmo.
- Houve **{amostragem['cortes']} pausas** de mais de 1 minuto. Cada pausa separa uma
  gravacao da seguinte.
- A maior pausa foi de **{amostragem['maior_gap_horas']:.0f} horas** — quase
  {amostragem['maior_gap_horas'] / 24:.0f} dias sem nenhuma leitura.

**Conclusao:** as {resumo['linhas']:,} linhas nao sao uma medicao continua de
{janela['duracao_dias']:.0f} dias. Sao **{amostragem['sessoes_estimadas']} gravacoes
curtas** espalhadas nesse periodo.
""".replace(",", ".")
)

if not janela["monotonico"]:
    st.error(
        """
**O arquivo nao esta em ordem de data.**

Uma linha pode ser de 3 de junho e a linha logo abaixo, de 15 de maio. As gravacoes
foram juntadas fora de ordem.

*O que fazemos com isso:* qualquer calculo que dependa da leitura anterior — media
movel, formacao de eventos, tempo entre leituras — precisa ordenar por data primeiro.
Todos os calculos desta interface ja fazem isso.
"""
    )

st.subheader("Distribuicao do tempo entre leituras")

h, bordas = np.histogram(amostragem["intervalos"], bins=60, range=(0, 10))
hist_int = pd.DataFrame({"intervalo_s": (bordas[:-1] + bordas[1:]) / 2, "leituras": h})

st.altair_chart(
    alt.Chart(hist_int)
    .mark_bar(color="#4c78a8")
    .encode(
        x=alt.X("intervalo_s:Q", title="segundos entre uma leitura e a seguinte"),
        y=alt.Y("leituras:Q", title="quantas vezes", scale=alt.Scale(type="symlog")),
        tooltip=[alt.Tooltip("intervalo_s:Q", format=".2f"), "leituras"],
    )
    .properties(height=280),
    width="stretch",
)
st.caption(
    "O grafico vai so ate 10 segundos, senao as pausas de horas achatariam tudo. "
    "A altura usa escala comprimida para os valores raros continuarem visiveis. "
    "Ha dois picos: a maioria em 2 s, e um segundo grupo perto de 5,3 s — "
    "provavelmente outra configuracao do equipamento em parte das gravacoes."
)

# --------------------------------------------------------------------------
# 2.1 Leituras com a mesma data e hora
# --------------------------------------------------------------------------
st.subheader("Leituras gravadas com a mesma data e hora")

if tempos_dup["total"] == 0:
    st.success("Cada leitura tem um instante proprio. Nenhuma data e hora se repete.")
else:
    resumo_dup = tempos_dup["resumo"]
    linha = resumo_dup.iloc[0]

    st.markdown(
        f"""
**{tempos_dup['total']:,} leituras compartilham a mesma data e hora.**

Nao sao varios instantes repetidos: e **um unico instante** com
{tempos_dup['total']:,} leituras dentro.
""".replace(",", ".")
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Leituras afetadas", f"{tempos_dup['total']:,}".replace(",", "."))
    m2.metric("Instantes repetidos", tempos_dup["instantes"])
    m3.metric("Sao copias identicas?", "Nao" if not linha["medidas_identicas"] else "Sim")
    m4.metric("Ids em sequencia?", "Sim" if linha["ids_contiguos"] else "Nao")

    st.dataframe(
        resumo_dup[
            ["created_at", "linhas", "rotulos", "id_min", "id_max",
             "medidas_identicas", "linhas_repetidas", "ids_contiguos"]
        ],
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "data e hora", format="DD/MM/YYYY HH:mm:ss.SSSSSS"
            ),
            "linhas": st.column_config.NumberColumn("leituras", format="%d"),
            "rotulos": "tipo de falha",
            "id_min": st.column_config.NumberColumn("primeiro id", format="%d"),
            "id_max": st.column_config.NumberColumn("ultimo id", format="%d"),
            "medidas_identicas": st.column_config.CheckboxColumn(
                "medidas iguais?", help="Se sim, sao copias do mesmo registro."
            ),
            "linhas_repetidas": st.column_config.NumberColumn(
                "linhas repetidas", format="%d",
                help="Quantas leituras do bloco sao copia exata de outra do bloco."
            ),
            "ids_contiguos": st.column_config.CheckboxColumn(
                "ids em sequencia?",
                help="Se sim, o bloco entrou no arquivo de uma vez so."
            ),
        },
    )

    st.error(
        f"""
**O que aconteceu aqui, em palavras simples**

{tempos_dup['total']:,} leituras do tipo **{linha['rotulos']}** receberam todas a
mesma data e hora: **{linha['created_at']:%d/%m/%Y as %H:%M:%S}**.

Elas **nao sao copias**: os valores medidos variam normalmente entre elas
(so {linha['linhas_repetidas']} sao iguais a outra). Ou seja, as **medidas sao
reais**; o que esta errado e o **horario**.

Tres sinais de que foi uma carga em lote, e nao uma coleta:

1. Os ids vao de {linha['id_min']:,} a {linha['id_max']:,} **sem pular nenhum** — o
   bloco entrou no arquivo de uma vez.
2. Logo antes e logo depois desse bloco, o arquivo tem leituras de outro tipo de
   falha (`rolamento_ball_2`) separadas por 2 segundos normais.
3. Esse bloco e **um terco de todas as leituras** desse tipo de falha.

*O que fazemos com isso:* as medidas continuam validas para comparar vibracao. Mas
essas leituras **nao podem sustentar nada que dependa de tempo** — duracao do
evento, frequencia com que acontece, ordem dos acontecimentos. Precisam ficar
marcadas na proxima etapa.
""".replace(",", ".")
    )

    linhas_dup = tempos_dup["linhas"]

    st.markdown(
        f"""
**As {len(linhas_dup):,} leituras do bloco, com todas as {linhas_dup.shape[1]} colunas.**

Estao em ordem de `id`. Role a tabela para o lado para ver todas as medidas.

Repare no essencial: a coluna **data e hora e sempre a mesma**, enquanto todas as
medidas de vibracao **mudam de linha para linha**. E isso que prova que sao leituras
diferentes com o horario errado, e nao o mesmo registro copiado.
""".replace(",", ".")
    )

    st.dataframe(
        linhas_dup,
        hide_index=True,
        height=460,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "data e hora", format="DD/MM/YYYY HH:mm:ss.SSSSSS"
            ),
            "fault": "tipo de falha",
        },
    )

    with st.expander("Quanto cada medida variou dentro do bloco"):
        st.caption(
            "Se fossem copias do mesmo registro, minimo e maximo seriam iguais em "
            "todas as linhas."
        )
        medidas_bloco = linhas_dup.select_dtypes(include="number").drop(
            columns=["id"], errors="ignore"
        )
        variacao = medidas_bloco.describe().T[["min", "50%", "max", "std"]].reset_index()
        variacao.columns = ["medida", "minimo", "valor do meio", "maximo", "desvio"]
        variacao["valores diferentes"] = [
            int(medidas_bloco[c].nunique()) for c in medidas_bloco.columns
        ]
        st.dataframe(
            variacao.round(4), hide_index=True, height=400,
            column_config={
                "valores diferentes": st.column_config.NumberColumn(
                    "valores diferentes", format="%d",
                    help="Quantos valores distintos essa medida assume nas 1000 linhas.",
                ),
            },
        )

    st.download_button(
        "Baixar as leituras afetadas (CSV)",
        tempos_dup["linhas"].to_csv(index=False).encode("utf-8"),
        file_name="leituras_com_data_repetida.csv",
        mime="text/csv",
    )

st.divider()

# ==========================================================================
# 3. Colunas sem informacao propria
# ==========================================================================
st.header("3. Colunas que nao acrescentam informacao")

st.subheader("3.1 Colunas com sempre o mesmo valor")
st.markdown(
    """
Uma coluna que tem sempre o mesmo valor nao distingue nada — e como uma pergunta
em que todo mundo responde igual. Alem de inutil, ela quebra o calculo de
padronizacao usado mais adiante.
"""
)

n_const = int(constantes["constante"].sum())
if n_const == 0:
    st.success("**Nenhuma coluna tem sempre o mesmo valor.** Todas variam.")
    st.info(
        """
**Isto desmente uma suspeita que tinhamos.**

Achavamos que `z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` fossem sempre
61 Hz. Nao sao: tem 79 e 50 valores diferentes. 61 Hz e so o valor mais comum
(aparece em 60% e 49% das linhas).

Ou seja, **essas colunas ficam**. A frequencia muda justamente em alguns defeitos,
que e a informacao que queremos.
"""
    )
else:
    st.warning(f"{n_const} coluna(s) com sempre o mesmo valor.")

st.dataframe(
    constantes.head(10),
    hide_index=True,
    column_config={
        "distintos": st.column_config.NumberColumn("valores diferentes", format="%d"),
        "valor_dominante": "valor mais comum",
        "pct_dominante": st.column_config.NumberColumn("% desse valor", format="%.2f%%"),
        "constante": st.column_config.CheckboxColumn("sempre igual?"),
    },
)
st.caption(
    "Repare no `rpm`: so 5 valores em 166 mil linhas (0, 500, 1000, 2000 e 3000). "
    "Nao e uma medida continua, sao 5 regimes de rotacao. Isso muda como ele deve "
    "ser usado na proxima etapa."
)

st.subheader("3.2 Colunas que sao a mesma medida em outra unidade")
st.markdown(
    """
Algumas colunas repetem a mesma medida convertida: polegadas e milimetros,
Fahrenheit e Celsius.

Para confirmar, nao olhamos se elas "andam juntas" — fizemos a conta.
Multiplicamos a coluna em polegada por 25,4 e comparamos com a coluna em milimetro.
Se der igual, uma e copia da outra.
"""
)

st.dataframe(
    redundantes,
    hide_index=True,
    column_config={
        "coluna_descartavel": "pode sair",
        "coluna_mantida": "fica",
        "relacao": "conta usada",
        "erro_max": st.column_config.NumberColumn("maior diferenca", format="%.6f"),
        "erro_medio": st.column_config.NumberColumn("diferenca media", format="%.6f"),
        "redundante": st.column_config.CheckboxColumn("e copia?"),
    },
)
st.success(
    f"""
**{int(redundantes['redundante'].sum())} de {len(redundantes)} pares confirmados como copia.**

A maior diferenca encontrada foi de 0,016 — o tamanho do arredondamento do proprio
arquivo. Nao ha medida independente ali.

Ficamos com milimetro e Celsius. Manter as duas versoes faria o sistema contar a
mesma grandeza duas vezes e dar peso dobrado a ela.
"""
)

st.divider()

# ==========================================================================
# 4. Leituras repetidas
# ==========================================================================
st.header("4. Leituras identicas a anterior")

st.markdown(
    f"""
**{duplicatas['total']:,} linhas ({duplicatas['pct']}%)** sao identicas a linha
imediatamente anterior em **todas** as medidas.

A comparacao ignora `id` e `created_at` de proposito: esses dois sempre mudam, e
inclui-los faria nenhuma duplicata aparecer.

Duas leituras exatamente iguais, ate a quarta casa decimal, tiradas com 2 segundos
de diferenca, quase certamente sao **a mesma medicao gravada duas vezes**. Uma
maquina real nao repete valor com essa precisao.

*Por que atrapalha:* inflam a contagem de ocorrencias e, na etapa de busca por
similaridade, viram vizinhos de distancia zero que nao acrescentam nada.
""".replace(",", ".")
)

por_rot = duplicatas["por_rotulo"]
st.caption("Os 20 tipos com maior proporcao de leituras repetidas:")
st.altair_chart(
    alt.Chart(por_rot.head(20))
    .mark_bar(color="#e2a03f")
    .encode(
        x=alt.X("pct:Q", title="% das leituras desse tipo"),
        y=alt.Y("fault:N", sort="-x", title=None),
        tooltip=["fault", "duplicadas", "total", "pct"],
    )
    .properties(height=440),
    width="stretch",
)

with st.expander("Ver a tabela completa"):
    st.dataframe(
        por_rot, hide_index=True, height=400,
        column_config={
            "fault": "tipo de falha",
            "duplicadas": st.column_config.NumberColumn("repetidas", format="%d"),
            "total": st.column_config.NumberColumn("total", format="%d"),
            "pct": st.column_config.NumberColumn("%", format="%.2f%%"),
        },
    )

st.divider()

# ==========================================================================
# 5. Valores fora do normal
# ==========================================================================
st.header("5. Valores fora do normal")

st.markdown(
    """
### Como decidimos o que e "fora do normal"

Ordenamos os valores de cada coluna e vemos onde fica a metade do meio. Um valor
que se afasta muito dessa faixa central e marcado como fora do normal.

Nao usamos media e desvio padrao porque varias colunas tem valores extremos —
`z_kurtosis` tem valor tipico 2,5 e maximo 65. A media ja estaria contaminada
justamente pelos exageros que ela deveria encontrar.
"""
)

st.warning(
    """
**Nada e removido nem corrigido aqui.**

Em vibracao, o pico raro costuma ser **o sinal**, nao o ruido. Kurtosis alta e
exatamente a marca de um rolamento batendo. Apagar esses valores por regra
estatistica jogaria fora a falha que o sistema existe para encontrar.
"""
)

escopo = st.radio(
    "Calcular sobre",
    ["O arquivo inteiro", "Um tipo de falha especifico"],
    horizontal=True,
)

if escopo.startswith("O arquivo"):
    out = D.r_outliers_global()
    legenda = "todas as leituras do arquivo"
else:
    alvo = st.selectbox("Tipo de falha", D.r_rotulos()["fault"].tolist())
    out = D.r_outliers_do_rotulo(alvo)
    legenda = f"apenas as leituras de `{alvo}`"

st.caption(f"Limites calculados sobre {legenda}.")

st.altair_chart(
    alt.Chart(out.head(15))
    .mark_bar()
    .encode(
        x=alt.X("pct_outliers:Q", title="% de leituras fora dos limites"),
        y=alt.Y("coluna:N", sort="-x", title=None),
        color=alt.Color("pct_extremos:Q", title="% muito fora",
                        scale=alt.Scale(scheme="reds")),
        tooltip=[
            "coluna", "mediana", "lim_inferior", "lim_superior",
            "min", "max", "outliers", "pct_outliers", "extremos", "pct_extremos",
        ],
    )
    .properties(height=380),
    width="stretch",
)

st.dataframe(
    out,
    hide_index=True,
    height=400,
    column_config={
        "coluna": "medida",
        "mediana": st.column_config.NumberColumn("valor do meio", format="%.4f"),
        "lim_inferior": st.column_config.NumberColumn("limite baixo", format="%.4f"),
        "lim_superior": st.column_config.NumberColumn("limite alto", format="%.4f"),
        "outliers": st.column_config.NumberColumn("fora", format="%d"),
        "pct_outliers": st.column_config.NumberColumn("% fora", format="%.2f%%"),
        "extremos": st.column_config.NumberColumn("muito fora", format="%d"),
        "pct_extremos": st.column_config.NumberColumn("% muito fora", format="%.2f%%"),
        "max_sobre_limite": st.column_config.NumberColumn(
            "quantas vezes o limite", format="%.2f x",
            help="Quantas vezes o maior valor ultrapassa o limite alto."
        ),
    },
)

st.info(
    """
### Duas situacoes bem diferentes na mesma tabela

**Muitos valores fora, mas nenhum muito longe** (coluna *quantas vezes o limite*
perto de 1). E o caso de `z_peak_vel_comp_freq_hz`, com 25% fora. Acontece porque
quase tudo esta grudado em 61 Hz, entao a faixa central fica estreitissima e
qualquer variacao legitima cai fora. **E efeito do metodo, nao anomalia.**

**Poucos valores fora, mas muito longe** (coluna *quantas vezes o limite* na casa
das dezenas). `z_peak_acceleration_g` chega a **49 vezes** o limite. Sao poucos
eventos, mas violentos. **Este e o sinal que interessa** — provavelmente o impacto
de um rolamento com defeito.
"""
)

st.divider()

# ==========================================================================
# 6. Decisao
# ==========================================================================
st.header("6. Colunas que vamos descartar")

st.markdown(
    """
Resumo das decisoes tomadas nesta tela. A proxima etapa aplica esta lista.

Sao tres motivos:

- **copia** — a coluna e a mesma medida em outra unidade
- **vazamento** — a coluna nao mede a maquina, mede a ordem em que os dados foram
  gravados. Se ela entrar no modelo, ele acerta pelo motivo errado: descobre
  *quando* o dado foi coletado em vez de *como* a maquina vibra. Num equipamento
  novo isso nao funcionaria.
- **sempre igual** — a coluna nao varia
"""
)

st.dataframe(
    D.r_descartar(),
    hide_index=True,
    column_config={
        "coluna": "coluna",
        "motivo": "motivo",
        "detalhe": st.column_config.TextColumn("explicacao", width="large"),
    },
)

st.markdown(
    """
### O que **nao** vai ser descartado, apesar da suspeita inicial

| Coluna | Por que fica |
|---|---|
| `z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` | Nao sao fixas em 61 Hz — tem 79 e 50 valores diferentes |
| `temperature_f` | Sai, mas por ser copia de `temperature_c`, nao por falta de informacao |
| `rpm` | Fica. So que sera tratada como **5 regimes de rotacao**, nao como medida continua |
"""
)
