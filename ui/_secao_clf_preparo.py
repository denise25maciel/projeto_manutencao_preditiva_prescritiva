"""Aba 1 da tela de classificacao — **como a leitura vira um exemplo**.

E a adaptacao do `prep.py` do projeto irmao de classificacao, contada na ordem
em que o dado atravessa o pipeline. Cada passo mostra o que entrou, o que saiu e
por que a regra e essa e nao outra.

O fio da aba e uma frase so: **uma leitura de sensor nao e um exemplo de
defeito.** Uma linha do arquivo diz "a vibracao neste instante foi 3,2 mm/s", e
isso nao caracteriza falha nenhuma — o que caracteriza e o comportamento de um
trecho. Os seis passos abaixo sao o caminho de "linha do arquivo" ate "duas
bases prontas: a que treina o modelo e a que o cobra".

Nao desenha modelo nem acuracia: isso e a outra aba.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D
from mp import config
from mp.classificacao import ESTATISTICAS


# O que cada estatistica responde. Fica aqui, e nao no modulo, porque e texto de
# tela — o modulo guarda a lista, a tela guarda a explicacao.
SENTIDO_DA_ESTATISTICA = [
    ("mediana", "Em que patamar o trecho ficou",
     "Robusta ao pico isolado — e em kurtosis e crest factor o pico isolado e "
     "frequente, entao a media enganaria."),
    ("desvio", "O quanto oscilou em torno desse patamar",
     "Separa o trecho estavel do trecho agitado, mesmo quando os dois tem a "
     "mesma mediana."),
    ("inclinacao", "Se estava subindo ou caindo ao longo do trecho",
     "E a UNICA que enxerga a ordem das leituras. Sem ela, embaralhar o trecho "
     "nao mudaria nenhum numero — e um defeito que se agrava seria igual a um "
     "que estabilizou."),
    ("amplitude", "A distancia entre o maior e o menor valor",
     "O extremo do trecho. Sensivel a um unico impacto, o que as vezes e "
     "exatamente o sinal procurado."),
    ("p90_p10", "A mesma ideia, sem os 10% das pontas",
     "Ao lado da amplitude, separa 'oscilou o tempo todo' de 'teve um pico e "
     "voltou': quando a amplitude e grande e esta e pequena, foi pico."),
]


def render() -> None:
    """Desenha a aba de preparacao."""

    try:
        leituras = D.r_clf_leituras()
    except FileNotFoundError as e:
        D.aviso_csv_ausente(e)

    n_leituras = len(leituras)
    n_eventos = leituras["evento"].nunique()
    n_familias = leituras["familia"].nunique()
    tamanho = config.CLF_JANELA_TAMANHO

    st.markdown(
        """
Esta aba responde uma pergunta so: **o que e um exemplo para o modelo?**

A resposta obvia — "cada linha do arquivo" — esta errada, e vale entender por
que. Uma linha diz *"neste instante a vibracao vertical foi 3,2 mm/s"*. Isso
nao caracteriza defeito nenhum: a mesma leitura aparece numa maquina sadia
acelerando e numa com rolamento gasto. O que caracteriza e **como um trecho se
comporta** — em que patamar ficou, o quanto oscilou, se estava subindo.

Entao a leitura precisa virar trecho, e o trecho precisa virar numero. Sao
seis passos, nesta ordem.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leituras no arquivo", f"{n_leituras:,}".replace(",", "."))
    c2.metric("Eventos (os grupos)", n_eventos)
    c3.metric("Familias (as classes)", n_familias)
    c4.metric("Janela escolhida", f"{tamanho} leituras")

    st.divider()

    # ======================================================================
    # PASSO 1 — o rotulo vira familia
    # ======================================================================
    st.header("Passo 1 — O rotulo vira familia", divider="gray")

    st.markdown(
        """
A coluna `fault` tem **151 valores distintos**, e a maior parte e o mesmo
defeito escrito de outro jeito: `desabalanceado`, `desbanlanceado`,
`ddesbalanceado`, `new_desbalanceado_2`. O modelo nao pode tratar cada grafia
como uma classe — ele aprenderia a distinguir a digitacao do operador.

**Aqui esta a primeira adaptacao em relacao ao projeto de origem.** La, a
normalizacao era uma funcao Python com listas de typos, prefixos e radicais
escritos no codigo:
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Como era — regra no codigo**")
        st.code(
            'replacements = {\n'
            '    "desabalanceado": "desbalanceado",\n'
            '    "desbanlanceado": "desbalanceado",\n'
            '    "ddesbalanceado": "desbalanceado",\n'
            '    "normla": "normal",\n'
            '    ...\n'
            '}\n'
            'label = label.removeprefix("new_")\n'
            'for tipo in tipos:\n'
            '    if label.startswith(tipo):\n'
            '        return tipo',
            language="python",
        )
        st.caption("`prep.py`, funcao `classe_base()`.")
    with c2:
        st.markdown("**Como ficou — lookup no catalogo curado**")
        st.code(
            "from mp.retrieval.catalog import familia_de\n\n"
            'familia_de("ddesbalanceado")\n'
            '# -> "desbalanceamento"',
            language="python",
        )
        st.caption("`data/fault_map.yaml`, versionado, 151 aliases.")

    st.success(
        """
**Por que a troca.** Este projeto ja tem esse mapa, e ele nao e um detalhe do
classificador — e o **principio 1**: numero nunca e comparado com texto; o
evento resolve para rotulo, o rotulo resolve para familia, e a familia aponta o
manual. Se o classificador normalizasse por conta propria, existiriam duas
respostas para "que familia e esta?", e nada garantiria que concordassem. O dia
em que discordassem, o modelo diria uma coisa e o manual aberto seria de outra.
        """
    )

    tabela_fam = (
        leituras.groupby("familia", observed=True)
        .agg(leituras=("familia", "size"), eventos=("evento", "nunique"))
        .reset_index()
        .sort_values("leituras", ascending=False)
    )
    tabela_fam["pct"] = 100 * tabela_fam["leituras"] / n_leituras

    c1, c2 = st.columns([3, 2])
    with c1:
        st.altair_chart(
            alt.Chart(tabela_fam)
            .mark_bar()
            .encode(
                x=alt.X("leituras:Q", title="leituras"),
                y=alt.Y("familia:N", sort="-x", title=None),
                tooltip=["familia", "leituras", "eventos",
                         alt.Tooltip("pct:Q", format=".1f", title="% do arquivo")],
            )
            .properties(height=max(240, 22 * len(tabela_fam))),
            width="stretch",
        )
    with c2:
        st.dataframe(tabela_fam, width="stretch", hide_index=True)
        st.caption(
            f"**151 rotulos crus viraram {n_familias} familias.** Nenhuma leitura "
            "foi perdida: o catalogo cobre todos os rotulos observados."
        )

    st.info(
        """
**Os estados continuam no conjunto, e isso e proposital.** `normal`,
`baseline`, `teste` e `motor_desligado` nao sao defeitos — o guardrail **G2**
encerra o fluxo prescritivo neles. Mas para o classificador eles sao classes
legitimas: reconhecer que a maquina esta **bem** e uma resposta util, e sem
elas o modelo seria obrigado a nomear um defeito para toda leitura que
recebesse. Quem quiser o recorte so-defeitos passa `so_defeitos=True`.
        """
    )

    st.divider()

    # ======================================================================
    # PASSO 2 — as leituras viram eventos
    # ======================================================================
    st.header("Passo 2 — As leituras viram eventos", divider="gray")

    st.markdown(
        """
O trecho tem de sair de dentro de **uma** medicao. Juntar o fim de um ensaio
com o comeco do seguinte produziria um exemplo que nunca existiu.

**Segunda adaptacao.** O projeto de origem abria um grupo novo quando o texto de
`fault` mudava, ou quando havia mais de uma hora de pausa. Este projeto testou
essa regra na Parte 1 e a **rejeitou**: a bancada rodava 500, 1000 e 2000 rpm em
sequencia **sem trocar o nome da falha**, entao tres ensaios viravam um grupo
so. Medido na epoca: 136 dos 205 grupos assim formados misturavam rotacoes — 95%
das leituras. Num caso a velocidade RMS ia de 3,5 a 21,1 dentro do "mesmo" grupo.

Aqui o grupo e o **evento** de `ingestion.construir_eventos`, que quebra em
`fault` **e** `rpm`.
        """
    )

    c1, c2 = st.columns(2)
    c1.error(
        "**Regra do projeto de origem**\n\n"
        "`fault` mudou, ou parou por 1 hora.\n\n"
        "Tres rotacoes do mesmo defeito = **um** grupo."
    )
    c2.success(
        "**Regra daqui** (`config.COLUNAS_QUEBRA_EVENTO`)\n\n"
        "`fault` mudou **ou** `rpm` mudou.\n\n"
        "Tres rotacoes do mesmo defeito = **tres** grupos."
    )

    st.markdown(
        """
**Isso importa duas vezes, e a segunda e a que quase ninguem ve.**

1. **Ao resumir** — a mediana de um trecho que mistura tres regimes nao
   descreve nenhum dos tres.
2. **Ao validar** — o evento e o que a validacao por grupo segura fora do
   treino. Grupo mal formado deixa metade de um ensaio no treino e a outra
   metade no teste, que e exatamente o vazamento que a validacao por grupo
   existe para impedir. Um grupo errado nao produz erro: produz uma **nota alta
   e falsa**. A outra aba mede o tamanho disso.
        """
    )

    st.divider()

    # ======================================================================
    # PASSO 3 — quais colunas entram
    # ======================================================================
    st.header("Passo 3 — Quais medidas entram", divider="gray")

    sem_regime = D.r_clf_colunas(False)
    com_regime = D.r_clf_colunas(True)
    regime = [c for c in com_regime if c not in sem_regime]

    st.markdown(
        f"""
O arquivo tem 23 colunas numericas de medida. Nem todas devem entrar, e as
exclusoes moram num lugar so (`classificacao/colunas.py`), com o motivo de cada
uma escrito ao lado.
        """
    )

    descartes = pd.DataFrame(
        [
            {"o que sai": "id, created_at",
             "por que":
             "Identificadores que crescem com o tempo. O arquivo foi gravado em "
             "campanhas, uma falha por campanha — o modelo acertaria pela posicao "
             "no arquivo, sem aprender nada sobre vibracao."},
            {"o que sai": "*_in_s (4 colunas)",
             "por que":
             "A mesma velocidade em polegada por segundo (x 25,4). Manter as duas "
             "conta a mesma informacao duas vezes."},
            {"o que sai": "temperature_f",
             "por que": "A mesma temperatura em Fahrenheit."},
            {"o que sai": ", ".join(regime),
             "por que":
             "Regime de operacao, nao sintoma. Sao as duas de que a proxima aba "
             "mede o efeito, com um botao que liga e desliga."},
        ]
    )
    st.dataframe(descartes, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Colunas de medida", 23)
    c2.metric("Entram no modelo", len(sem_regime))
    c3.metric("Com o regime ligado", len(com_regime))

    st.warning(
        f"""
**A exclusao do regime e uma aposta, e a proxima aba a testa.** O projeto de
origem usava `{'` e `'.join(regime)}` como feature e depois registrou como
limitacao numero 1 que *"o modelo aprendeu o ensaio, nao o defeito — a
temperatura ambiente do dia, a rotacao exata"*. Aqui elas saem por padrao pelo
mesmo motivo que saem do kNN. Mas afirmar nao basta: o experimento de regime
mede as duas configuracoes lado a lado.
        """
    )

    st.caption("As colunas que entram, na ordem em que compoem a matriz:")
    st.code("\n".join(sem_regime), language="text")

    st.divider()

    # ======================================================================
    # PASSO 4 — o recorte em janelas
    # ======================================================================
    st.header("Passo 4 — O trecho: janelas dentro do evento", divider="gray")

    st.markdown(
        f"""
Dentro de cada evento, blocos de **{tamanho} leituras** consecutivas, andando de
{tamanho // 2} em {tamanho // 2} — ou seja, com 50% de sobreposicao entre
janelas vizinhas. A {config.INTERVALO_ESPERADO_S:.0f} s por leitura, cada janela
cobre cerca de **{tamanho * config.INTERVALO_ESPERADO_S / 60:.1f} minuto de
motor**.

Evento mais curto que a janela e **descartado inteiro**. E aqui que a escolha do
tamanho para de ser gosto e vira medida.
        """
    )

    tamanhos_evento = leituras.groupby("evento").size()
    hist = (
        tamanhos_evento.value_counts()
        .rename_axis("leituras_no_evento")
        .reset_index(name="quantos_eventos")
        .sort_values("quantos_eventos", ascending=False)
        .head(10)
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**O tamanho dos eventos e quase todo discreto**")
        st.dataframe(hist, width="stretch", hide_index=True)
        st.caption(
            "A bancada gravava aquisicoes de comprimento fixo. Nao ha uma "
            "distribuicao continua a cortar — ha um degrau."
        )
    with c2:
        st.markdown("**E por isso o descarte tem um degrau, nao uma rampa**")
        st.altair_chart(
            alt.Chart(_curva_de_descarte(tamanhos_evento))
            .mark_line(point=True)
            .encode(
                x=alt.X("janela:Q", title="tamanho da janela (leituras)",
                        scale=alt.Scale(type="log")),
                y=alt.Y("pct_descartado:Q", title="% dos eventos descartados"),
                tooltip=["janela", alt.Tooltip("eventos_usados:Q"),
                         alt.Tooltip("pct_descartado:Q", format=".1f")],
            )
            .properties(height=260),
            width="stretch",
        )

    n_50 = int((tamanhos_evento >= 50).sum())
    n_60 = int((tamanhos_evento >= 60).sum())
    st.success(
        f"""
**Por que {tamanho}.** {int((tamanhos_evento == 50).sum())} dos {n_eventos}
eventos tem exatamente 50 leituras. Uma janela de 50 cabe em
**{n_50} eventos**; uma de 60 cabe em **{n_60}**. Um passo acima de 50 elimina
dois tercos do conjunto — e leva uma familia inteira junto, sem que metrica
nenhuma caia por isso, porque a familia simplesmente deixa de existir para o
modelo.

{tamanho} e a maior janela que ainda cabe no evento tipico. O numero mora em
`config.CLF_JANELA_TAMANHO`, com esta medicao no comentario.
        """
    )

    st.markdown("**Quem sobrevive a janela, por familia**")
    cobertura = D.r_clf_cobertura(tamanho)
    st.dataframe(
        cobertura,
        width="stretch",
        hide_index=True,
        column_config={
            "pct_descartado": st.column_config.ProgressColumn(
                "% eventos descartados", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )
    perdidas = cobertura[cobertura["eventos_aproveitados"] == 0]["familia"].tolist()
    if perdidas:
        st.error(
            f"**Familias que somem com esta janela:** `{'`, `'.join(perdidas)}`. "
            "Todos os eventos delas sao mais curtos que a janela. O modelo nunca "
            "vera um exemplo, e portanto nunca respondera esse nome."
        )
    else:
        st.caption("Nenhuma familia desaparece com esta janela.")

    st.divider()

    # ======================================================================
    # PASSO 5 — a janela vira numero
    # ======================================================================
    st.header("Passo 5 — Cada janela vira uma linha de numeros", divider="gray")

    amostras = D.r_clf_amostras("janela", tamanho, False)
    n_features = len(sem_regime) * len(ESTATISTICAS)

    st.markdown(
        f"""
A janela ainda e uma tabela de {tamanho} x {len(sem_regime)}. O modelo precisa de
**uma linha**. Cada coluna e resumida em {len(ESTATISTICAS)} numeros, e
{len(sem_regime)} x {len(ESTATISTICAS)} da as **{n_features} colunas** da matriz
final.

As cinco nao sao intercambiaveis — cada uma responde uma pergunta que as outras
nao respondem:
        """
    )

    st.dataframe(
        pd.DataFrame(
            [{"estatistica": e, "responde": r, "por que ela": p}
             for e, r, p in SENTIDO_DA_ESTATISTICA]
        ),
        width="stretch",
        hide_index=True,
    )

    # ----------------------------------------------------------------------
    # A transformacao, num caso concreto
    # ----------------------------------------------------------------------
    st.subheader("A transformacao, num caso concreto")

    st.markdown(
        f"""
A tabela acima diz o que cada estatistica significa. Aqui ela **acontece**:
escolha um evento e veja as {tamanho} leituras cruas de um lado e, do outro, os
{len(ESTATISTICAS)} numeros que saem de cada coluna. Os dois quadros sao o mesmo
trecho — o da esquerda tem {tamanho} linhas, o da direita tem {len(sem_regime)}.

Os numeros da direita nao sao recalculados para a tela: saem de `resumir_bloco`,
a mesma funcao que monta o conjunto de treino. E literalmente o que entra na
floresta.
        """
    )

    consultaveis = D.r_clf_eventos_consultaveis(tamanho)
    c1, c2 = st.columns(2)
    familia_demo = c1.selectbox(
        "Familia", sorted(consultaveis["familia"].unique()), key="clf_demo_familia"
    )
    do_demo = consultaveis[consultaveis["familia"] == familia_demo]
    evento_demo = c2.selectbox(
        "Evento",
        do_demo["evento"].tolist(),
        format_func=lambda e: f"evento {e} — {int(do_demo.loc[do_demo['evento'] == e, 'n_leituras'].iloc[0])} leituras",
        key="clf_demo_evento",
    )

    cruas = D.r_clf_leituras_do_evento(int(evento_demo), limite=tamanho)
    estatisticas = D.r_clf_estatisticas(int(evento_demo), tamanho, False)

    esq, meio, dir = st.columns([5, 1, 5])

    with esq:
        st.markdown(f"**ANTES — {len(cruas)} leituras cruas**")
        st.caption(
            f"Uma linha por leitura do sensor, a cada "
            f"{config.INTERVALO_ESPERADO_S:.0f} s. Role para o lado."
        )
        st.dataframe(
            cruas[sem_regime].round(4),
            width="stretch",
            hide_index=True,
            height=300,
        )
        st.caption(f"`{len(cruas)} x {len(sem_regime)}` — nao cabe numa linha de matriz.")

    with meio:
        st.markdown("&nbsp;")
        st.markdown("&nbsp;")
        st.markdown(
            "<div style='text-align:center;font-size:2.4rem;line-height:3rem'>→</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "<div style='text-align:center'>resumir_bloco</div>",
            unsafe_allow_html=True,
        )

    with dir:
        st.markdown(f"**DEPOIS — {len(sem_regime)} colunas x {len(ESTATISTICAS)} numeros**")
        st.caption(
            "Uma linha por coluna de medida. Achatada em sequencia, vira a linha "
            "unica que o modelo recebe."
        )
        st.dataframe(
            estatisticas.round(4).reset_index(),
            width="stretch",
            hide_index=True,
            height=300,
        )
        st.caption(
            f"`{len(sem_regime)} x {len(ESTATISTICAS)}` = **{n_features} numeros**, "
            "numa linha so."
        )

    st.info(
        f"""
**O que se perde e o que se ganha.** As {tamanho} leituras viram
{n_features} numeros: perde-se a leitura individual, e nao da para voltar atras.
Ganha-se o que uma leitura sozinha nunca teve — **variacao, tendencia e
extremo**, que e onde o defeito se manifesta. Uma vibracao de 3,2 mm/s nao
distingue maquina nenhuma; 3,2 mm/s **subindo, com picos de 5,1**, distingue.
        """
    )

    st.markdown("**A matriz final, do jeito que o modelo a recebe**")

    nomes = D.r_clf_nomes_features(False)
    previa = pd.DataFrame(
        np.vstack(amostras["features"].head(8).to_list()), columns=nomes
    ).iloc[:, :6]
    previa.insert(0, "→ classe", amostras["familia"].head(8).to_numpy())
    previa.insert(1, "evento (grupo)", amostras["evento"].head(8).to_numpy())
    st.dataframe(previa, width="stretch", hide_index=True)
    st.caption(
        f"8 das {len(amostras):,} amostras, e 6 das {n_features} colunas de numeros."
        .replace(",", ".")
        + " A coluna `evento` **nao** entra no modelo: ela e o grupo que a "
          "validacao usa para separar treino de teste."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", f"{len(amostras):,}".replace(",", "."))
    c2.metric("Numeros por amostra", n_features)
    c3.metric("Eventos representados", amostras["evento"].nunique())
    c4.metric("Familias a distinguir", amostras["familia"].nunique())

    st.download_button(
        "Baixar a matriz completa (CSV)",
        data=D.r_clf_csv(tamanho, False),
        file_name=f"amostras_janela_{tamanho}.csv",
        mime="text/csv",
    )

    st.divider()

    # ======================================================================
    # PASSO 6 — as duas bases
    # ======================================================================
    st.header("Passo 6 — As duas bases: treino e teste", divider="gray")

    _secao_bases(tamanho, n_features)

    st.divider()

    # ======================================================================
    # O caminho inteiro
    # ======================================================================
    st.header("O caminho inteiro, numa tabela", divider="gray")

    st.dataframe(
        pd.DataFrame(
            [
                {"etapa": "Arquivo bruto", "o que e uma linha": "uma leitura do sensor",
                 "linhas": f"{n_leituras:,}".replace(",", "."), "colunas": "26"},
                {"etapa": "1 — Rotulo vira familia",
                 "o que e uma linha": "uma leitura, com a familia resolvida",
                 "linhas": f"{n_leituras:,}".replace(",", "."), "colunas": "27"},
                {"etapa": "2 — Leituras viram eventos",
                 "o que e uma linha": "uma leitura, com o evento marcado",
                 "linhas": f"{n_leituras:,}".replace(",", "."), "colunas": "28"},
                {"etapa": "3 — Medidas escolhidas",
                 "o que e uma linha": "uma leitura, so com o que e sintoma",
                 "linhas": f"{n_leituras:,}".replace(",", "."),
                 "colunas": str(len(sem_regime))},
                {"etapa": f"4 — Recorte em janelas de {tamanho}",
                 "o que e uma linha": "ainda uma leitura, agrupada em blocos",
                 "linhas": f"{len(amostras):,} blocos".replace(",", "."),
                 "colunas": f"{tamanho} x {len(sem_regime)} cada"},
                {"etapa": "5 — Cada janela resumida",
                 "o que e uma linha": "**uma janela inteira**",
                 "linhas": f"{len(amostras):,}".replace(",", "."),
                 "colunas": f"{n_features} + classe + grupo"},
                {"etapa": "6 — Cortada em duas bases",
                 "o que e uma linha": "a mesma janela, de um lado ou do outro",
                 "linhas": "~80% treino / ~20% teste",
                 "colunas": f"{n_features} + classe + grupo"},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Repare onde o significado de *linha* muda: ate a etapa 3 e uma leitura "
        "do sensor; na etapa 5 e uma janela inteira. E essa troca que faz o "
        "problema virar aprendivel. A etapa 6 nao transforma nada — so reparte, "
        "e o criterio da reparticao e o assunto da proxima aba."
    )

    st.info(
        """
**O que foi adaptado, em resumo.** O algoritmo do projeto de origem esta
inteiro aqui — o mesmo recorte em janelas com sobreposicao, as mesmas cinco
estatisticas, a mesma floresta na proxima aba. O que trocou foi de onde vem
cada **decisao**: o rotulo agora vem do `fault_map.yaml` em vez de regras no
codigo, o grupo vem do evento com `rpm` em vez da troca de rotulo, e as colunas
vem da mesma funcao que o kNN usa. Sao tres pecas que este projeto ja tinha e
que agora tem um consumidor a mais, em vez de um segundo dono.
        """
    )


def _secao_bases(tamanho: int, n_features: int) -> None:
    """As duas bases de pe: a que treina o modelo e a que o cobra.

    A validacao cruzada faz este corte cinco vezes e descarta as tabelas — o que
    e certo para medir e inutil para conferir. Aqui o corte e feito uma vez, e as
    duas bases ficam visiveis e baixaveis.
    """
    st.markdown(
        """
A matriz do passo 5 e uma so. O modelo nao pode estudar nela inteira e depois
ser cobrado nela inteira — seria dar a prova com o gabarito junto: ele tira nota
alta sem ter aprendido nada, e ninguem descobre se sabe a materia.

Entao ela e cortada em duas:

- **Treino** — o que o modelo estuda, com a resposta a vista. **E esta a base
  que da entrada no modelo.**
- **Teste** — guardada no cofre. A resposta e escondida, o modelo palpita, e so
  entao comparamos.
        """
    )

    c1, c2 = st.columns([2, 3])
    por_evento = c1.radio(
        "Como cortar",
        [True, False],
        format_func=lambda v: (
            "Sorteando eventos inteiros (correto)" if v
            else "Sorteando amostras soltas (errado)"
        ),
        key="clf_base_corte",
    )
    fracao = c2.slider(
        "Fatia que vai para o teste", 0.10, 0.40, 0.20, step=0.05,
        format="%.0f%%", key="clf_base_fracao",
    )

    divisao = D.r_clf_divisao(tamanho, False, bool(por_evento), float(fracao))
    treino, teste = divisao["treino"], divisao["teste"]

    esq, dir = st.columns(2)

    with esq:
        st.markdown("#### TREINO — o que entra no modelo")
        st.caption("O modelo estuda estas linhas, com a coluna `familia` a vista.")
        st.dataframe(
            treino.head(8).iloc[:, :6].round(4), width="stretch", hide_index=True
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Amostras", f"{len(treino):,}".replace(",", "."))
        m2.metric("Eventos", divisao["eventos_treino"])
        m3.metric("Familias", divisao["familias_treino"])
        st.download_button(
            "Baixar a base de treino (CSV)",
            data=D.r_clf_csv_base("treino", tamanho, False, bool(por_evento)),
            file_name=f"treino_janela_{tamanho}.csv",
            mime="text/csv",
            width="stretch",
        )

    with dir:
        st.markdown("#### TESTE — o que cobra o modelo")
        st.caption("A coluna `familia` e escondida na hora de perguntar.")
        st.dataframe(
            teste.head(8).iloc[:, :6].round(4), width="stretch", hide_index=True
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Amostras", f"{len(teste):,}".replace(",", "."))
        m2.metric("Eventos", divisao["eventos_teste"])
        m3.metric("Familias", divisao["familias_teste"])
        st.download_button(
            "Baixar a base de teste (CSV)",
            data=D.r_clf_csv_base("teste", tamanho, False, bool(por_evento)),
            file_name=f"teste_janela_{tamanho}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.caption(
        f"Mostrando 8 linhas e as 6 primeiras de {n_features + 2} colunas. "
        f"Proporcao real: {100 * (1 - divisao['fracao_real']):.0f}% treino / "
        f"{100 * divisao['fracao_real']:.0f}% teste. Cada linha e uma janela "
        "resumida; `familia` e a resposta e `evento` e o grupo — nenhuma das "
        "duas entra como feature."
    )

    # ----------------------------------------------------------------------
    # A auditoria do proprio corte
    # ----------------------------------------------------------------------
    st.subheader("A pergunta que decide se a prova vale")

    st.markdown(
        "**Algum evento aparece dos dois lados ao mesmo tempo?** Se aparecer, o "
        "modelo reencontra na prova uma medicao que estudou — e como janelas do "
        "mesmo evento sao quase identicas (mesma bancada, mesma montagem, e "
        "ainda metade do conteudo repetido pela sobreposicao), isso e cola."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Eventos no treino", divisao["eventos_treino"])
    m2.metric("Eventos no teste", divisao["eventos_teste"])
    m3.metric("Eventos nos DOIS lados", divisao["eventos_vazados"])

    if divisao["eventos_vazados"]:
        st.error(
            f"**{divisao['eventos_vazados']} dos {divisao['eventos_teste']} "
            f"eventos do teste tambem estao no treino** — "
            f"{100 * divisao['eventos_vazados'] / divisao['eventos_teste']:.0f}% "
            "deles. A nota que sair daqui sai alta e nao significa nada. E o "
            "corte que a aba 2 mede: 92% contra 44%."
        )
        if divisao["lista_vazados"]:
            st.caption(
                "Alguns dos eventos que vazaram: "
                + ", ".join(str(e) for e in divisao["lista_vazados"])
            )
    else:
        st.success(
            "**Nenhum evento aparece nos dois lados.** Toda medicao testada e "
            "uma que o modelo nunca viu. A nota que sair daqui e honesta — e a "
            "resposta para a pergunta que importa: *vai funcionar numa maquina "
            "nova?*"
        )

    # ----------------------------------------------------------------------
    # O preco
    # ----------------------------------------------------------------------
    st.markdown("**O preco de cortar certo**")

    problemas = []
    if abs(divisao["fracao_real"] - divisao["fracao_pedida"]) > 0.01:
        problemas.append(
            f"O corte nao sai exatamente em {100 * divisao['fracao_pedida']:.0f}%: "
            f"deu {100 * divisao['fracao_real']:.1f}%. Os eventos tem tamanhos "
            "muito diferentes — 50 ou 1.000 leituras —, e sortea-los inteiros "
            "faz a proporcao variar."
        )
    if divisao["familias_so_no_treino"]:
        problemas.append(
            "Familias que ficaram **so no treino** e nao serao avaliadas: "
            f"`{'`, `'.join(divisao['familias_so_no_treino'])}`."
        )
    if divisao["familias_so_no_teste"]:
        problemas.append(
            "Familias que ficaram **so no teste**: "
            f"`{'`, `'.join(divisao['familias_so_no_teste'])}`. O modelo sera "
            "cobrado por um nome que nunca viu, e errara todas."
        )

    if problemas:
        st.warning("\n\n".join(f"- {p}" for p in problemas))
    else:
        st.caption("Neste sorteio, as duas bases ficaram equilibradas.")

    st.info(
        """
**Este corte serve para ver a base, nao para medir o modelo.** Um sorteio unico
depende da sorte: outra semente da outro numero, e as familias que sobraram de
um lado so mudam. Quem mede repete o corte 5 vezes e tira a media — e a
validacao cruzada da aba 2, que tambem e onde as duas maneiras de cortar sao
comparadas de verdade.
        """
    )


# --------------------------------------------------------------------------
# Auxiliares de desenho
# --------------------------------------------------------------------------


def _curva_de_descarte(tamanhos_evento: pd.Series) -> pd.DataFrame:
    """Quantos eventos cabem em cada tamanho de janela.

    Conta direto sobre o comprimento dos eventos, sem montar amostra nenhuma —
    a curva sai instantanea, e montar as janelas para descobrir isso levaria
    dezenas de segundos por ponto.
    """
    total = len(tamanhos_evento)
    grade = [10, 20, 30, 40, 50, 60, 90, 120, 180, 360, 500, 1000]
    linhas = []
    for janela in grade:
        usados = int((tamanhos_evento >= janela).sum())
        linhas.append(
            {
                "janela": janela,
                "eventos_usados": usados,
                "pct_descartado": 100 * (1 - usados / total),
            }
        )
    return pd.DataFrame(linhas)
