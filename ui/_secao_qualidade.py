"""Ato 2 da narrativa — **da para confiar no que chegou?**

Campos vazios, ritmo da coleta, colunas repetidas, leituras duplicadas e valores
fora do normal. Nada e corrigido aqui: esta etapa descreve, a proxima trata.

Era uma pagina propria (`pages/2_Qualidade_dos_Dados.py`). Virou secao de
`app.py` para a analise ser lida como uma historia so, em vez de tres telas
soltas que o avaliador precisa costurar de cabeca. O conteudo e o mesmo — a
conversao foi mecanica e conferida literal por literal.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

import _dados as D



def render() -> None:
    """Desenha esta secao dentro da narrativa unica."""

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

    # "Leituras" e "Campos vazios" sairam: as duas ja sao metricas do ato 1, e
    # aqui nao explicavam nada — a secao de campos vazios foi removida porque nao
    # ha nenhum. Ficam as duas que este ato apura.
    c1, c2 = st.columns(2)
    c1.metric(
        "Leituras repetidas",
        f"{duplicatas['total']:,}".replace(",", "."),
        f"{duplicatas['pct']}% do total",
        delta_color="inverse",
    )
    c2.metric("Colunas descartaveis", len(D.r_descartar()))

    st.divider()

    # ==========================================================================
    # As colunas do arquivo — o inventario, antes de qualquer medicao
    # ==========================================================================
    #
    # Sobe para o topo porque e o mapa do que vem depois: as secoes seguintes
    # falam de coluna constante, coluna redundante e coluna descartavel, e todas
    # as tres se leem melhor com a lista das colunas ja a vista.
    #
    # A tabela era o fecho da secao "Campos vazios", que saiu — nao ha nenhum
    # campo vazio no arquivo, entao ela gastava a primeira tela do ato para
    # dizer que nao havia o que dizer.
    st.subheader("As colunas do arquivo")

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

    # 1. Colunas sem informacao propria
    # ==========================================================================
    
    st.subheader("1.2 Colunas que sao a mesma medida em outra unidade")

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
Manter as duas versoes faria o sistema contar a
mesma grandeza duas vezes e dar peso dobrado a ela.
"""
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


    st.divider()

    # --------------------------------------------------------------------------
    # 2.2 Leituras com a mesma data e hora
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
""".replace(",", ".")
        )

    st.divider()

    # ==========================================================================

    # ==========================================================================

    # ==========================================================================
    # 3. Valores fora do normal
    # ==========================================================================
    st.header("3. Valores fora do normal")

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

    st.divider()

    # ==========================================================================
    # 4. Decisao
    # ==========================================================================
    st.header("4. Colunas que vamos descartar")

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


