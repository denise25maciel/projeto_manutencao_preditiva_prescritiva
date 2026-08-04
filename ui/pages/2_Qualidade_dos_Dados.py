"""Como os dados chegaram: nulos, cadencia, redundancias, duplicatas e outliers.

Nada e corrigido nesta tela. Parte 0 descreve; Parte 1 trata.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D

D.configurar_pagina("Qualidade dos Dados", "🧪")

st.title("🧪 Qualidade dos Dados")
st.caption("Diagnostico do dado bruto. Nenhum valor e alterado aqui.")

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Leituras", f"{resumo['linhas']:,}".replace(",", "."))
c2.metric("Celulas nulas", f"{resumo['celulas_nulas']:,}".replace(",", "."))
c3.metric(
    "Duplicatas consecutivas",
    f"{duplicatas['total']:,}".replace(",", "."),
    f"{duplicatas['pct']}% das linhas",
    delta_color="inverse",
)
c4.metric("Colunas descartaveis", len(D.r_descartar()))

st.divider()

# ==========================================================================
# 1. Nulos
# ==========================================================================
st.header("1. Valores nulos por coluna")

total_nulos = int(nulos["nulos"].sum())
if total_nulos == 0:
    st.success(
        f"**Nenhum nulo declarado** em nenhuma das {resumo['colunas']} colunas, nas "
        f"{resumo['linhas']:,} linhas. O dataset chegou completo.".replace(",", ".")
    )
    st.warning(
        "Ausencia de `NaN` nao e ausencia de dado faltante. Um sensor sem leitura "
        "pode ter sido gravado como `0.0` — nulo disfarcado de medida. Os candidatos "
        "estao nas secoes de constantes e de outliers, nao aqui."
    )
else:
    st.altair_chart(
        alt.Chart(nulos[nulos["nulos"] > 0])
        .mark_bar(color="#d1495b")
        .encode(
            x=alt.X("pct_nulos:Q", title="% nulos"),
            y=alt.Y("coluna:N", sort="-x", title=None),
            tooltip=["coluna", "nulos", "pct_nulos"],
        ),
        width="stretch",
    )

# `pct_nulos` e `preenchidos` sao redundantes na tela: com zero nulos, o
# percentual e sempre 0,000% e `preenchidos` repete o total de linhas em todas
# as 26 colunas. Continuam no retorno de `nulos_por_coluna` porque o grafico
# acima usa o percentual quando existe nulo.
st.dataframe(
    nulos[["coluna", "tipo", "nulos", "distintos"]],
    hide_index=True,
    height=330,
    column_config={
        "nulos": st.column_config.NumberColumn("nulos", format="%d"),
        "distintos": st.column_config.NumberColumn(
            "valores distintos",
            format="%d",
            help="Poucos valores distintos em 166 mil linhas = candidata a "
            "categorica disfarcada de numerica.",
        ),
    },
)

st.divider()

# ==========================================================================
# 2. Como o dado foi coletado
# ==========================================================================
st.header("2. Cadencia e continuidade da coleta")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Intervalo mediano", f"{amostragem['intervalo_mediano_s']} s", help="O esperado era ~2 s")
c2.metric(
    "Na cadencia nominal",
    f"{amostragem['pct_na_cadencia']}%",
    help="Fracao das leituras com intervalo de 2 s +- 0,25 s",
)
c3.metric("Cortes (> 60 s)", amostragem["cortes"])
c4.metric("Maior gap", f"{amostragem['maior_gap_horas']:.1f} h")

st.markdown(
    f"""
Coleta de **{janela['inicio']:%d/%m/%Y %H:%M}** a **{janela['fim']:%d/%m/%Y %H:%M}** UTC
({janela['duracao_dias']} dias). {janela['timestamps_repetidos']} timestamps aparecem
repetidos.
"""
)

if not janela["monotonico"]:
    st.error(
        "**`created_at` nao e monotonico.** O arquivo nao esta em ordem cronologica: "
        "ha saltos negativos de dezenas de dias entre linhas vizinhas. Sao "
        f"{amostragem['sessoes_estimadas']} sessoes gravadas em epocas diferentes e "
        "concatenadas fora de ordem.\n\n"
        "**Consequencia:** toda operacao que depende de vizinhanca temporal — a "
        "mediana movel da Parte 3, a formacao de episodios da Parte 1 — precisa "
        "ordenar por `created_at` antes, e nunca atravessar a fronteira entre sessoes."
    )

st.subheader("Distribuicao do intervalo entre leituras")
st.caption(
    "Calculado apos ordenar por tempo. O eixo para em 10 s para continuar legivel — "
    "os gaps entre sessoes chegam a 122 h e estao contados no cartao 'Cortes'."
)

h, bordas = np.histogram(amostragem["intervalos"], bins=60, range=(0, 10))
hist_int = pd.DataFrame({"intervalo_s": (bordas[:-1] + bordas[1:]) / 2, "leituras": h})

st.altair_chart(
    alt.Chart(hist_int)
    .mark_bar(color="#4c78a8")
    .encode(
        x=alt.X("intervalo_s:Q", title="intervalo entre leituras (s)"),
        y=alt.Y("leituras:Q", title="leituras", scale=alt.Scale(type="symlog")),
        tooltip=[alt.Tooltip("intervalo_s:Q", format=".2f"), "leituras"],
    )
    .properties(height=280),
    width="stretch",
)
st.caption(
    "Eixo Y em escala simlog. A massa esta em 2 s, com um segundo modo perto de "
    "5,3 s — provavelmente outra configuracao de datalogger em parte das sessoes."
)

st.divider()

# ==========================================================================
# 3. Colunas constantes e redundantes
# ==========================================================================
st.header("3. Colunas sem informacao propria")

st.subheader("3.1 Constantes e quase-constantes")
st.caption(
    "Variancia nula nao distingue nada e quebra o `StandardScaler` (divisao por "
    "desvio padrao zero)."
)

n_const = int(constantes["constante"].sum())
if n_const == 0:
    st.success("**Nenhuma coluna constante.** Todas tem mais de um valor distinto.")
    st.info(
        "Isso **contraria a suspeita registrada no GUIA.md** de que "
        "`z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` seriam fixas em 61 Hz. "
        "Elas tem 79 e 50 valores distintos; 61 Hz e a moda (60% e 49% das linhas), "
        "nao o valor unico. As colunas carregam informacao e **nao devem ser "
        "descartadas** — a frequencia do pico se desloca justamente em alguns defeitos."
    )
else:
    st.warning(f"{n_const} coluna(s) constante(s).")

st.dataframe(
    constantes.head(10),
    hide_index=True,
    column_config={
        "distintos": st.column_config.NumberColumn("valores distintos", format="%d"),
        "pct_dominante": st.column_config.NumberColumn("% do valor dominante", format="%.2f%%"),
    },
)
st.caption(
    "`rpm` tem 5 valores (0, 500, 1000, 2000, 3000): e uma categorica de regime "
    "disfarcada de numerica, nao uma medida continua. Importa para o kNN da Parte 3."
)

st.subheader("3.2 Unidades duplicadas")
st.caption(
    "Testamos a identidade numerica, nao a correlacao: mm/s = in/s x 25,4 e "
    "F = C x 9/5 + 32. Se bater dentro do arredondamento do arquivo, uma coluna e "
    "conversao da outra — nao duas medidas independentes."
)

st.dataframe(
    redundantes,
    hide_index=True,
    column_config={
        "coluna_descartavel": "descartar",
        "coluna_mantida": "manter",
        "erro_max": st.column_config.NumberColumn("erro max", format="%.6f"),
        "erro_medio": st.column_config.NumberColumn("erro medio", format="%.6f"),
        "redundante": st.column_config.CheckboxColumn("redundante?"),
    },
)
st.success(
    f"**{int(redundantes['redundante'].sum())} de {len(redundantes)} pares confirmados.** "
    "O erro maximo fica na casa do arredondamento do arquivo. Mantemos o SI (mm/s, C) "
    "e descartamos a versao imperial: duplicar a mesma grandeza faz o `StandardScaler` "
    "conta-la duas vezes e infla o peso dela na distancia do kNN."
)

st.divider()

# ==========================================================================
# 4. Duplicatas consecutivas
# ==========================================================================
st.header("4. Leituras repetidas")

st.markdown(
    f"""
**{duplicatas['total']:,} linhas ({duplicatas['pct']}%)** sao identicas a linha
anterior em todas as colunas de medida. A comparacao ignora `id` e `created_at`
de proposito — eles sempre mudam, e inclui-los nunca acusaria duplicata nenhuma.

Duas leituras iguais em 4 casas decimais a 2 s de distancia sao, quase certamente,
a mesma amostra repetida pelo datalogger. Elas inflam a contagem de ocorrencias e,
na Parte 3, viram vizinhos de distancia zero que nao acrescentam informacao.
""".replace(",", ".")
)

por_rot = duplicatas["por_rotulo"]
st.altair_chart(
    alt.Chart(por_rot.head(20))
    .mark_bar(color="#e2a03f")
    .encode(
        x=alt.X("pct:Q", title="% de linhas duplicadas no rotulo"),
        y=alt.Y("fault:N", sort="-x", title=None),
        tooltip=["fault", "duplicadas", "total", "pct"],
    )
    .properties(height=440),
    width="stretch",
)

with st.expander("Tabela completa por rotulo"):
    st.dataframe(por_rot, hide_index=True, height=400)

st.divider()

# ==========================================================================
# 5. Outliers
# ==========================================================================
st.header("5. Outliers")

st.markdown(
    """
Criterio de **Tukey (IQR)**: um valor e outlier se sai de `Q1 - 1,5 x IQR` a
`Q3 + 1,5 x IQR`, e **extremo** com fator 3,0.

Usamos IQR e nao z-score porque varias colunas sao fortemente assimetricas
(`z_kurtosis` tem mediana 2,5 e maximo 65): a media e o desvio padrao que o
z-score usa ja estao contaminados pelos proprios extremos que deveriam detectar.
"""
)

st.warning(
    "**Nada e removido nem corrigido.** Nesta etapa os outliers sao apenas "
    "identificados. Em vibracao, o pico raro costuma ser o sinal — nao o ruido: "
    "kurtosis alta e exatamente a assinatura de impacto de rolamento. Descartar "
    "por regra estatistica apagaria a falha que o sistema existe para detectar."
)

escopo = st.radio(
    "Escopo do calculo",
    ["Global (dataset inteiro)", "Dentro de um rotulo"],
    horizontal=True,
    help="Globalmente, toda leitura de um defeito severo parece outlier — o que e "
    "esperado, e nao erro de medicao.",
)

if escopo.startswith("Global"):
    out = D.r_outliers_global()
    legenda = "dataset inteiro"
else:
    alvo = st.selectbox("Rotulo", D.r_rotulos()["fault"].tolist())
    out = D.r_outliers_do_rotulo(alvo)
    legenda = f"rotulo `{alvo}`"

st.caption(f"Limites calculados sobre o {legenda}.")

st.altair_chart(
    alt.Chart(out.head(15))
    .mark_bar()
    .encode(
        x=alt.X("pct_outliers:Q", title="% de leituras fora dos limites"),
        y=alt.Y("coluna:N", sort="-x", title=None),
        color=alt.Color("pct_extremos:Q", title="% extremos", scale=alt.Scale(scheme="reds")),
        tooltip=[
            "coluna",
            "mediana",
            "lim_inferior",
            "lim_superior",
            "min",
            "max",
            "outliers",
            "pct_outliers",
            "extremos",
            "pct_extremos",
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
        "pct_outliers": st.column_config.NumberColumn("% outliers", format="%.2f%%"),
        "pct_extremos": st.column_config.NumberColumn("% extremos", format="%.2f%%"),
        "max_sobre_limite": st.column_config.NumberColumn(
            "max / limite sup",
            format="%.2f",
            help="Quantas vezes o maximo ultrapassa o limite superior. Valor alto "
            "indica cauda longa de impacto, nao ruido disperso.",
        ),
    },
)

st.info(
    "Duas leituras diferentes na mesma tabela: `% outliers` alto com "
    "`max / limite sup` proximo de 1 e dispersao larga e uniforme — provavelmente "
    "mistura de regimes. `% outliers` baixo com `max / limite sup` na casa das "
    "dezenas (`z_peak_acceleration_g` chega a 49x) e cauda longa de impacto: "
    "poucos eventos, muito acima do normal. Este segundo caso e o que interessa."
)

st.divider()

# ==========================================================================
# 6. Decisao consolidada
# ==========================================================================
st.header("6. Colunas a descartar — decisao da Parte 0")
st.caption("Artefato que a Parte 1 aplica. Cada linha traz o motivo verificado.")

st.dataframe(
    D.r_descartar(),
    hide_index=True,
    column_config={"detalhe": st.column_config.TextColumn("detalhe", width="large")},
)

st.markdown(
    """
**Nao entram na lista de descarte**, apesar da suspeita inicial:

- `z_peak_vel_comp_freq_hz` / `x_peak_vel_comp_freq_hz` — nao sao constantes (secao 3.1)
- `temperature_f` — descartada por redundancia com `temperature_c`, nao por falta de sinal
- `rpm` — mantida, mas tratada como **categorica de regime** (5 patamares), nao continua
"""
)
