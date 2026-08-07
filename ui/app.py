"""A tela de analise inteira: a narrativa e os seis atos, num arquivo so.

    ato 1   o que chegou          quanto dado, de quando, com que rotulos
    ato 2   o arquivo cru         a serie sem ordenar, agrupar nem reamostrar
    ato 3   da para confiar?      o que veio torto, e quanto disso pesa
    ato 4   o que os dados dizem  o comportamento medido de cada falha
    ato 5   de leitura a evento   as linhas viram ocorrencias contaveis
    ato 6   os procedimentos      a outra fonte: os 6 manuais da empresa

A ordem carrega o argumento. O ato 2 mostra o arquivo **antes** de qualquer
correcao — e a unica secao que nao ordena por data —, e por isso vem antes do
levantamento de qualidade: primeiro se ve o problema, depois se mede. O ato 3
vem antes do 4 porque ler a assinatura de uma falha sem saber que o arquivo tem
leituras repetidas e horarios errados e ler numero sem saber a margem dele.

Os atos 5 e 6 fecham a preparacao das **duas fontes** que o resto do sistema
cruza. Elas so se encontram pela coluna `fault`, via `fault_map.yaml` — numero
nunca e comparado com texto.

**Cada ato e uma funcao, e isso nao e detalhe.** Os atos moravam em cinco
arquivos `_secao_*.py`, cada um com um `render()`. Ao junta-los, manter as
funcoes preserva a unica coisa que a separacao em arquivos garantia de graca: o
**escopo isolado**. `filtro`, `colunas`, `total` e `inicio` sao nomes que quase
todos os atos usam, e no nivel do modulo eles colidiriam em silencio — o valor
de um ato vazaria para o seguinte, sem erro nenhum.

Rodar com:  streamlit run ui/app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D


# ==========================================================================
# Os atos 2 a 6, um por funcao
# ==========================================================================
# O ato 1 nao e funcao: ele e o cabecalho da narrativa e mora no corpo do
# script, mais abaixo, junto do titulo e da introducao.


def ato_2_arquivo_cru() -> None:
    """Ato 2 da narrativa — **o arquivo cru**, sem ordenar, agrupar ou reamostrar.
    
    Todas as outras telas corrigem alguma coisa antes de desenhar: ordenam por data,
    separam sessoes, agregam pontos vizinhos. Esta nao corrige nada. O eixo x e a
    posicao da linha no arquivo, e cada ponto e uma leitura.
    
    Era uma pagina propria (`pages/0_Dados_Brutos.py`). Virou secao da narrativa
    unica pelo mesmo motivo dos outros atos, e com a mesma conversao conferida
    literal por literal.
    """
    st.header("📄 O arquivo", divider="gray")
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


    # O explorador interativo saiu daqui.
    #
    # Eram o filtro por falha, os dois sliders de posicao, o seletor de colunas
    # e a serie desenhada a partir deles. O que esta secao tem de proprio e
    # mostrar o arquivo **como ele e** — as primeiras linhas e os numeros que
    # provam que ele nao esta em ordem de data. Percorrer o arquivo a mao nao
    # acrescentava a esse argumento, e ocupava a maior parte da tela.
    #
    # O comportamento de cada falha ao longo do tempo continua no bloco
    # "As falhas", que e onde a pergunta faz sentido.



def dados_as_colunas() -> None:
    """O que cada coluna e: quantos valores tem, quais se repetem, quais saem.

    Reune o inventario, os pares redundantes, os valores fora do normal e a
    lista de descartes — as quatro perguntas sobre **coluna**. O ritmo da coleta
    saiu daqui: e assunto do arquivo, e virou parte do primeiro ato.
    """
    try:
        resumo = D.r_resumo()
    except FileNotFoundError as e:
        D.aviso_csv_ausente(e)

    nulos = D.r_nulos()
    constantes = D.r_constantes()  # noqa: F841 — usado pelos blocos abaixo
    redundantes = D.r_redundantes()

    st.metric("Colunas descartaveis", len(D.r_descartar()))

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
    
    st.subheader("Colunas que sao a mesma medida em outra unidade")

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

    st.header("Valores fora do normal")

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
    st.header("Colunas que vamos descartar")

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



def ato_4_falhas() -> None:
    """Ato 3 da narrativa — **o que os dados dizem**.
    
    Valores unicos de `fault` e o comportamento medido de cada um.
    
    Ordem dos blocos: primeiro a serie ao longo do tempo, depois o resumo em
    medianas. Ver a curva antes da estatistica evita ler uma mediana sem saber se
    ela descreve um patamar estavel ou a media de dois regimes distintos.
    
    Era uma pagina propria (`pages/1_Analise_de_Falhas.py`). Virou secao de `app.py`
    pelo mesmo motivo do ato 2, e com a mesma conversao conferida.
    """
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
    st.header("Os tipos de falha do arquivo")

    n_total = len(tabela)
    n_problema = int(tabela["e_problema"].sum())
    n_familias = tabela["familia_sugerida"].nunique()

    # Quantos nomes ha, quantos sao defeito e quantos sao estado ja foram ditos
    # no ato 1. O que este ato acrescenta e o **agrupamento** — e so ele fica.
    st.metric("Grupos (familias)", n_familias)

    st.markdown(
        f"""
Os {n_total} nomes do ato 1 nao sao {n_total} defeitos: o mesmo problema aparece
escrito de varias formas. Agrupando por radical, eles viram **{n_familias}
familias**.

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
    with st.expander("Ver o total de leituras por familia", expanded=True):
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


def ato_5_eventos() -> None:
    """Ato 5 da narrativa — **de leituras para ocorrencias**.
    
    Um evento e uma vez em que a maquina foi medida com o mesmo defeito, na mesma
    rotacao. Agrupar as 166.796 linhas em eventos e o que permite responder *"quantas
    vezes isso aconteceu?"* — contar linhas daria "4.200 ocorrencias" para uma unica
    sessao de bancada.
    
    A secao e um experimento com duas bases: a esquerda ordena por data antes de
    separar por falha; a direita faz o contrario.
    
    Era uma pagina propria (`pages/4_Eventos.py`). Virou secao de `app.py` pelo mesmo
    motivo das outras. O conteudo e o mesmo — a conversao foi mecanica.
    """
    st.header("🧩 Eventos — de leituras para ocorrencias", divider="gray")

    st.markdown(
        """
    Um **evento** e uma vez em que a maquina foi medida com o mesmo defeito, na mesma
    rotacao. Agrupar as 166.796 linhas em eventos e o que permite responder *"quantas
    vezes isso aconteceu?"*.

    Ha duas ordens possiveis para monta-los, e elas nao dao o mesmo resultado. As
    duas rodam aqui, mas a comparacao fica **recolhida** — ela sustenta a escolha,
    nao e o que se veio ver.
    """
    )

    with st.expander("Por que a rotacao encerra um evento"):
        st.markdown(
            """
    No comeco a regra quebrava so na troca de falha. O resultado eram **136 dos 205
    eventos misturando rotacoes** — 95% das leituras.

    A bancada rodava 500, 1000 e 2000 rpm em sequencia sem trocar o nome da falha,
    entao tres ensaios viravam um evento so.

    O caso extremo era o evento de `rolamento_combination_pos_2`:

    | rotacao | leituras | velocidade RMS |
    |---|---|---|
    | 500 rpm | 50 | 3,5 |
    | 1.000 rpm | 50 | 5,4 |
    | 2.000 rpm | 50 | **21,1** |

    Seis vezes maior dentro do "mesmo" evento. A mediana dele nao descrevia nenhum
    dos tres regimes.

    Incluindo a rotacao na regra: **205 → 526 eventos**, dispersao interna tipica de
    **2,40 → 1,31**, e zero eventos com regime misturado.
    """
        )

    try:
        comp = D.r_comparar_abordagens()
    except FileNotFoundError as e:
        D.aviso_csv_ausente(e)

    eventos_a_todos = comp["eventos_a"]
    eventos_b_todos = comp["eventos_b"]
    resumo = comp["resumo"]
    linha_a = resumo.iloc[0]
    linha_b = resumo.iloc[1]

    NOME_A = "🅐 Data → Falha"
    NOME_B = "🅑 Falha → Data"
    COR_A = "#2d6a4f"
    COR_B = "#d1495b"

    st.divider()

    # ==========================================================================
    # Filtro por familia — vale para tudo o que vem abaixo
    # ==========================================================================
    familias = sorted(
        set(eventos_a_todos["familia"].dropna()) | set(eventos_b_todos["familia"].dropna())
    )

    escolhidas = st.multiselect(
        "Familia de falhas",
        familias,
        default=familias,
        help="Filtra as duas bases e o grafico. Deixe vazio para ver tudo.",
        key="ev_familias",
    )
    filtro = escolhidas or familias

    eventos_a = eventos_a_todos[eventos_a_todos["familia"].isin(filtro)]
    eventos_b = eventos_b_todos[eventos_b_todos["familia"].isin(filtro)]

    if len(escolhidas) < len(familias):
        st.caption(
            f"Mostrando {len(escolhidas)} de {len(familias)} familias: "
            f"{len(eventos_a)} eventos na base 🅐 e {len(eventos_b)} na 🅑."
        )

    # ==========================================================================
    # As duas bases, lado a lado — recolhido
    # ==========================================================================
    #
    # Comeca fechado porque e **metodologia, nao resultado**. Quem abre a tela
    # quer saber quantas vezes cada falha aconteceu; qual das duas ordens de
    # operacao monta os eventos e a pergunta de quem for auditar a decisao.
    #
    # Continua aqui, e nao apagado, porque a comparacao e o que sustenta a
    # escolha — mas ocupava a primeira tela inteira do ato com duas tabelas de
    # 380 px antes de qualquer dado medido aparecer.
    with st.expander(
        "🔬 O teste das duas bases — como os eventos foram montados", expanded=False
    ):
        st.caption(
            "Ha duas ordens possiveis de operacao, e elas nao dao o mesmo "
            "resultado. As duas rodam aqui sobre o mesmo arquivo, para a escolha "
            "poder ser conferida em vez de aceita."
        )

        esquerda, direita = st.columns(2, gap="large")

        with esquerda:
            st.markdown(
                f"<h3 style='color:{COR_A};margin-bottom:0'>{NOME_A}</h3>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Ordena o arquivo inteiro por data. Depois percorre de cima a baixo e "
                "comeca um evento novo quando muda a falha ou a rotacao."
            )
            st.metric("Eventos", len(eventos_a))
            a1, a2 = st.columns(2)
            a1.metric(
                "Maior duracao",
                f"{eventos_a['duracao_s'].max() / 3600:.0f} h" if len(eventos_a) else "—",
            )
            a2.metric(
                "Duracao tipica",
                f"{eventos_a['duracao_min'].median():.0f} min" if len(eventos_a) else "—",
            )

            st.dataframe(
                eventos_a[["evento", "fault", "n_leituras", "inicio", "duracao_min"]],
                hide_index=True,
                height=380,
                column_config={
                    "evento": st.column_config.NumberColumn("n", format="%d"),
                    "fault": "falha",
                    "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
                    "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
                    "duracao_min": st.column_config.NumberColumn("min", format="%.0f"),
                },
            )
            st.download_button(
                "Baixar base 🅐",
                eventos_a.to_csv(index=False).encode("utf-8"),
                file_name="eventos_A_data_depois_falha.csv",
                mime="text/csv",
                key="dl_a",
            )

        with direita:
            st.markdown(
                f"<h3 style='color:{COR_B};margin-bottom:0'>{NOME_B}</h3>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Separa as leituras por falha. Depois ordena cada grupo por data. "
                "Como nada muda dentro do grupo, cada combinacao vira um evento."
            )
            st.metric("Eventos", len(eventos_b))
            b1, b2 = st.columns(2)
            b1.metric(
                "Maior duracao",
                f"{eventos_b['duracao_s'].max() / 3600:.0f} h" if len(eventos_b) else "—",
            )
            b2.metric(
                "Duracao tipica",
                f"{eventos_b['duracao_min'].median():.0f} min" if len(eventos_b) else "—",
            )

            st.dataframe(
                eventos_b[["evento", "fault", "n_leituras", "inicio", "duracao_min"]],
                hide_index=True,
                height=380,
                column_config={
                    "evento": st.column_config.NumberColumn("n", format="%d"),
                    "fault": "falha",
                    "n_leituras": st.column_config.NumberColumn("leituras", format="%d"),
                    "inicio": st.column_config.DatetimeColumn("comecou", format="DD/MM/YY HH:mm"),
                    "duracao_min": st.column_config.NumberColumn("min", format="%.0f"),
                },
            )
            st.download_button(
                "Baixar base 🅑",
                eventos_b.to_csv(index=False).encode("utf-8"),
                file_name="eventos_B_falha_depois_data.csv",
                mime="text/csv",
                key="dl_b",
            )

        # ------------------------------------------------------------------
        # O mesmo experimento, evento a evento
        # ------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # Um evento por grafico, para comparar formas
        # --------------------------------------------------------------------------
        st.subheader("Um evento abaixo do outro, com todas as medidas")

        st.markdown(
            """
        Aqui cada evento ganha o proprio grafico, com **todas as medidas juntas**. Serve
        para responder olhando: *estes eventos se parecem?*

        Duas coisas tornam a comparacao possivel:

        - **Todas as medidas na mesma escala.** Cada uma vira "quantos desvios acima ou
          abaixo da media do arquivo". Sem isso, `rpm` (0 a 3000) e `z_kurtosis` (2 a 65)
          nao caberiam no mesmo eixo.
        - **Todos os eventos comecam no zero.** O eixo horizontal e o tempo decorrido
          desde o inicio de cada evento, nao a data. Em data absoluta, cada um apareceria
          num canto da tela e as formas nao poderiam ser comparadas. A data real de inicio
          esta no titulo de cada grafico.

        **Eventos parecidos tem desenhos parecidos.**
        """
        )

        versao_perfil = st.radio(
            "Base",
            [NOME_A, NOME_B],
            horizontal=True,
            key="ev_versao_perfil",
        )
        eventos_da_base = eventos_a if versao_perfil == NOME_A else eventos_b
        letra = "A" if versao_perfil == NOME_A else "B"

        if eventos_da_base.empty:
            st.info("Nenhum evento nesta selecao.")
        else:
            rotulos_evento = {
                int(r["evento"]): (
                    f"{int(r['evento'])} · {r['fault']} · {r['inicio']:%d/%m %H:%M} · "
                    f"{int(r['n_leituras'])} leituras"
                )
                for _, r in eventos_da_base.iterrows()
            }
            opcoes_evento = list(rotulos_evento)

            escolhidos = st.multiselect(
                "Eventos para comparar",
                opcoes_evento,
                default=opcoes_evento[:4],
                max_selections=8,
                format_func=lambda e: rotulos_evento[e],
                help="Ate 8. Cada um vira um grafico, um abaixo do outro.",
                key="ev_rotulos_perfil",
            )

            # `numericas` era definida no bloco da serie temporal, que saiu da
            # secao. Fica aqui, junto de quem a usa — e cacheada em `_dados`,
            # entao chamar de novo nao custa leitura de arquivo.
            numericas = D.r_numericas()

            medidas = st.multiselect(
                "Medidas",
                numericas,
                default=[c for c in D.config.COLUNAS_ASSINATURA if c in numericas],
                help="Por padrao, as medidas que compoem a assinatura de vibracao.",
                key="ev_medidas_perfil",
            )

            if not escolhidos:
                st.info("Escolha ao menos um evento.")
            elif not medidas:
                st.info("Escolha ao menos uma medida.")
            else:
                # Orcamento por grafico: n_medidas x pontos precisa ficar abaixo do
                # limite de 5000 linhas do Vega.
                pontos = max(40, min(200, 4000 // max(len(medidas), 1)))
                perfis = D.r_series_por_evento(
                    letra, tuple(escolhidos), tuple(medidas), pontos
                )

                if perfis.empty:
                    st.info("Sem dados para estes eventos.")
                else:
                    # Escala de Y compartilhada: sem isso, cada grafico se ajustaria ao
                    # proprio maximo e formas diferentes pareceriam iguais.
                    lim = float(perfis["valor"].abs().quantile(0.995))
                    dominio = [-lim, lim]

                    for ev in escolhidos:
                        d = perfis[perfis["evento"] == ev]
                        if d.empty:
                            continue
                        info_ev = eventos_da_base[eventos_da_base["evento"] == ev].iloc[0]

                        st.markdown(
                            f"**Evento {ev} · `{info_ev['fault']}`** — "
                            f"{info_ev['inicio']:%d/%m/%Y %H:%M} · "
                            f"{int(info_ev['n_leituras'])} leituras · "
                            f"{info_ev['duracao_min']:.0f} min · "
                            f"dispersao {info_ev['dispersao']:.2f}"
                        )

                        st.altair_chart(
                            alt.Chart(d)
                            .mark_line(strokeWidth=1.2, opacity=0.85)
                            .encode(
                                x=alt.X("minuto:Q", title="minutos desde o inicio do evento"),
                                y=alt.Y(
                                    "valor:Q",
                                    title="desvios da media do arquivo",
                                    scale=alt.Scale(domain=dominio, clamp=True),
                                ),
                                color=alt.Color("coluna:N", title="medida",
                                                legend=alt.Legend(orient="right", columns=1)),
                                tooltip=[
                                    alt.Tooltip("coluna:N", title="medida"),
                                    alt.Tooltip("minuto:Q", title="minuto", format=".1f"),
                                    alt.Tooltip("valor:Q", title="desvios", format=".2f"),
                                ],
                            )
                            .properties(height=240),
                            width="stretch",
                        )

                    st.caption(
                        f"Eixo Y igual em todos os graficos (de {dominio[0]:.1f} a "
                        f"{dominio[1]:.1f} desvios), senao cada um se ajustaria ao proprio "
                        "maximo e formas diferentes pareceriam iguais. A linha em zero e a "
                        "media do arquivo inteiro."
                    )





    st.divider()


def ato_6_documentos() -> None:
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
    st.header("Quais manuais temos")

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
    st.header("Quais itens estao em quais manuais")

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
    st.header("Qual manual atende qual falha")

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


# ==========================================================================
# A narrativa
# ==========================================================================

D.configurar_pagina("Analise de Dados")

st.title("🔧 Manutencao Prescritiva — Dos dados aos procedimentos")
st.caption("Entender as duas fontes antes de cruzar qualquer coisa.")

try:
    resumo = D.r_resumo()
except FileNotFoundError as e:
    D.aviso_csv_ausente(e)

janela = D.r_janela()
amostragem = D.r_amostragem()
rotulos = D.r_rotulos()

# --------------------------------------------------------------------------
# O shape, antes de qualquer outra coisa
# --------------------------------------------------------------------------
#
# A primeira pergunta que se faz a um dataset: quantas linhas, quantas colunas.
# Vem antes da narrativa porque e o que dimensiona tudo o que vem depois.
st.markdown(
    f"### `dataset.shape` → **({resumo['linhas']:,}, {resumo['colunas']})**"
    .replace(",", ".")
)
st.caption(
    f"{resumo['linhas']:,} linhas × {resumo['colunas']} colunas".replace(",", ".")
    + " — uma linha por leitura do sensor, uma coluna por medida "
      "(mais `id`, `created_at` e o rotulo `fault`)."
)

st.markdown(
    """
Esta e a etapa de **exploracao**. Aqui nada e corrigido, filtrado ou apagado:
so descrevemos o que veio no arquivo. Limpar os dados vem depois.

A tela e uma **sequencia, nao um painel**, e vai do geral ao especifico —
quatro perguntas, nesta ordem:
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.info("""**1 — Os dados gerais**

Quanto dado, de quando, em que ritmo.""")
c2.info("""**2 — As colunas**

O que cada medida e, e quais delas ficam.""")
c3.info("""**3 — As falhas**

Que defeitos existem e quantas vezes ocorreram.""")
c4.info("""**4 — Os documentos**

Os 6 manuais e o que cada um cobre.""")

st.caption(
    "A ordem carrega o argumento. Primeiro o **arquivo** — quanto veio, de "
    "quando, se da para confiar no horario. Depois as **colunas**, porque nao da "
    "para ler a assinatura de uma falha sem saber que duas delas medem a mesma "
    "coisa. So entao as **falhas**. E por fim os **documentos**, que sao a outra "
    "fonte — e as duas so se encontram pela coluna `fault`, nunca por semelhanca "
    "entre numero e texto."
)

st.divider()

# ==========================================================================
# 1 — OS DADOS GERAIS
# ==========================================================================
st.header("1 — Os dados gerais", divider="gray")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Leituras", f"{resumo['linhas']:,}".replace(",", "."))
c2.metric("Colunas", resumo["colunas"])
c3.metric("Tipos de falha", resumo["rotulos_distintos"])
c4.metric("Campos vazios", f"{resumo['celulas_nulas']:,}".replace(",", "."))
c5.metric("Sessoes de coleta", amostragem["sessoes_estimadas"])

st.caption(f"Arquivo: `{D.caminho_do_csv()}` — fica fora do Git, e dado da empresa.")

st.caption(f"Arquivo: `{D.caminho_do_csv()}` — fica fora do Git, e dado da empresa.")

st.subheader("O que temos")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        f"""
**Quando os dados foram coletados**

- De {janela['inicio']:%d/%m/%Y} a {janela['fim']:%d/%m/%Y}, ou seja
  {janela['duracao_dias']:.0f} dias
- O sensor gravava uma leitura a cada **{amostragem['intervalo_mediano_s']:.0f} segundos**
- {amostragem['pct_na_cadencia']:.0f}% das leituras seguem esse ritmo
"""
    )

with col_b:
    n_prob = int(rotulos["e_problema"].sum())
    st.markdown(
        f"""
**O que foi medido**

- A coluna `fault` tem {resumo['rotulos_distintos']} valores diferentes
- {n_prob} sao **defeitos**; {resumo['rotulos_distintos'] - n_prob} sao apenas
  **estados da maquina** (normal, teste, motor desligado...)
- Os dados não estão ordenados por data e há repetição
"""
    )

st.markdown(
    """
### Como e o arquivo, sem nenhum tratamento

Sem ordenar por data, sem agrupar, sem reamostrar. O eixo horizontal e a
**posicao da linha no arquivo**, e nao o tempo — e a unica parte da tela que nao
corrige nada, e por isso vem antes de qualquer medicao de qualidade: primeiro se
ve o problema, depois se mede.
"""
)

ato_2_arquivo_cru()

st.divider()


# ==========================================================================
# 2 — AS COLUNAS
# ==========================================================================
st.header("2 — As colunas", divider="gray")

st.markdown(
    """
Sabendo **quanto** dado existe, a pergunta seguinte e o que cada coluna
significa: quantos valores distintos ela tem, se duas colunas medem a mesma
grandeza em unidades diferentes, onde ha valor fora do normal, e quais podem
sair sem perda.

Vem antes das falhas de proposito: ler a assinatura de um defeito sem saber que
`z_rms_velocity_in_s` e `z_rms_velocity_mm_s` sao a mesma medida e contar a
mesma informacao duas vezes.
"""
)

dados_as_colunas()

st.divider()

# ==========================================================================
# 3 — AS FALHAS
# ==========================================================================
st.header("3 — As falhas", divider="gray")

st.markdown(
    """
Com o arquivo dimensionado e as colunas entendidas, da para olhar o que foi
medido: **que defeitos existem**, como cada um se comporta, e quantas vezes cada
um aconteceu de verdade.
"""
)

st.subheader("Quantas leituras cada tipo de falha tem (linhas do arquivo)")

top = st.slider("Quantos mostrar", 5, 60, 25, step=5)
recorte = rotulos.head(top)

st.altair_chart(
    alt.Chart(recorte)
    .mark_bar()
    .encode(
        x=alt.X("n_leituras:Q", title="numero de leituras"),
        y=alt.Y("fault:N", sort="-x", title=None),
        color=alt.Color(
            "e_problema:N",
            title="e defeito?",
            scale=alt.Scale(domain=[True, False], range=["#d1495b", "#5c8a8a"]),
        ),
        tooltip=["fault", "n_leituras", "pct", "e_problema"],
    )
    .properties(height=max(240, 18 * len(recorte))),
    width="stretch",
)

# Caixa colorida virou legenda: o argumento inteiro — por que linha nao e
# ocorrencia — e o **ato 5**, e ele o desenvolve com o exemplo das 13 mil linhas.
# Aqui basta a ressalva de leitura do proprio grafico.
st.caption(
    "A barra conta **linhas do arquivo**, nao quantas vezes o defeito aconteceu. "
    "Agrupar em ocorrencias e o ato 5."
)

st.divider()

ato_4_falhas()

st.divider()

st.markdown(
    """
### Quantas vezes isso aconteceu

Ate aqui a unidade foi a **linha do arquivo**. Ela nao responde a pergunta do
tecnico: `rolamento_inner` tem 17 mil linhas, e isso nao sao 17 mil falhas — sao
horas medindo a mesma. Um **evento** e uma vez em que a maquina foi medida com o
mesmo defeito, na mesma rotacao.
"""
)

ato_5_eventos()

st.divider()

# ==========================================================================
# 4 — OS DOCUMENTOS
# ==========================================================================
st.header("4 — Os documentos", divider="gray")

st.markdown(
    """
Os tres blocos anteriores trataram de **uma** fonte: o que o sensor mediu. Ela
diz o que esta acontecendo e nunca diz o que fazer.

A segunda fonte sao os 6 manuais da empresa. Eles chegam em PDF, viram texto com
as secoes numeradas preservadas, e e dai que sai a citacao — nao basta responder
"alinhe o motor", tem de ser *"conforme o Doc2, secao 9"*.

**As duas fontes nunca se comparam diretamente.** O evento resolve para um
rotulo, o rotulo resolve para uma familia, e a familia aponta o documento pelo
`fault_map.yaml`, que e curado a mao e versionado. E por isso que uma falha sem
manual e recusada por um `SELECT`, e nao por uma busca que sempre acharia "o
trecho menos diferente".
"""
)

ato_6_documentos()

st.divider()

# ==========================================================================
# Fecho
# ==========================================================================
st.header("Depois desta historia", divider="gray")

st.markdown(
    """
Os seis atos acima prepararam as duas fontes: o que a maquina mediu e o que os
manuais mandam fazer. Quem usa isso sao as duas telas do menu a esquerda.

**Diagnostico** — o fluxo completo, em duas abas:

- **Diagnostico e conversa** — o tecnico descreve o problema ou chega o JSON do
  sensor, o sistema acha o procedimento e conversa sobre ele, citando documento,
  secao e pagina
- **Modelo de linguagem** — escolha do provedor (local ou API), as regras do
  prompt e o teste de conexao. Fica ali dentro, e nao numa tela separada, porque
  trocar de modelo no meio de uma conversa nao pode custar sair dela

**Classificacao** — a pergunta anterior a todas as outras: *da para descobrir a
familia so pelos numeros do sensor, sem ninguem anotar o rotulo?* Em tres abas:

- **Preparacao dos dados** — como uma leitura vira um exemplo que se pode
  aprender, e por que uma leitura sozinha nao e um
- **O modelo e o que ele vale** — a floresta, e as duas maneiras de cortar
  treino e teste que dao 92% e 44% sobre exatamente os mesmos dados. So uma das
  duas responde *"vai funcionar numa maquina nova?"*
- **Executar e ver o resultado** — o pipeline rodando de verdade, etapa por
  etapa e cronometrado, terminando no laudo dos testes. Sem cache: o tempo que
  a tela mostra e o tempo que levou
"""
)
