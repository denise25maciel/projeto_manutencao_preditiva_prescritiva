"""Aba 1 da tela de classificacao — **como a leitura vira um exemplo**.

E a adaptacao do `prep.py` do projeto irmao de classificacao, contada na ordem
em que o dado atravessa o pipeline. Cada passo mostra o que entrou, o que saiu e
por que a regra e essa e nao outra.

O fio da aba e uma frase so: **uma leitura de sensor nao e um exemplo de
defeito.** Uma linha do arquivo diz "a vibracao neste instante foi 3,2 mm/s", e
isso nao caracteriza falha nenhuma — o que caracteriza e o comportamento de um
trecho. Os cinco passos abaixo sao o caminho de "linha do arquivo" ate "exemplo
que o modelo pode aprender".

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
cinco passos, nesta ordem.
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
exclusoes sao as mesmas que o **motor de similaridade** ja aplica — a funcao e
literalmente a mesma (`similarity.features.colunas_de_similaridade`), para que
uma revisao de coluna nao precise ser feita em dois lugares.
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
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Repare onde o significado de *linha* muda: ate a etapa 3 e uma leitura "
        "do sensor; na etapa 5 e uma janela inteira. E essa troca que faz o "
        "problema virar aprendivel."
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
