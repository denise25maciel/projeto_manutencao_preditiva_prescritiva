"""Aba 2 da tela de classificacao — **a floresta, e o quanto ela vale**.

E a adaptacao do `sistema.py` e do `avaliacao.py` do projeto irmao. O modelo em
si ocupa pouco espaco aqui, e de proposito: treinar uma floresta e uma linha de
codigo, e o trabalho todo esta em descobrir **se o numero que ela devolve
significa alguma coisa**.

O achado que organiza a aba: a mesma floresta, sobre os mesmos dados, tira 92%
ou 44% conforme a forma de cortar o conjunto de teste — e so uma das duas notas
responde a pergunta "vai funcionar numa maquina que o modelo nunca viu?". A
distancia entre elas nao e ruido de medicao: e a medida de quanto o modelo
esta reconhecendo o ensaio em vez do defeito.

Nenhum numero desta tela e digitado no codigo. Todos saem de `mp.classificacao`
no momento em que a aba abre, e podem ser conferidos rodando o modulo direto.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import _dados as D
from mp import config


def render() -> None:
    """Desenha a aba do modelo."""

    tamanho = config.CLF_JANELA_TAMANHO

    st.markdown(
        """
A aba anterior terminou com uma tabela de numeros e uma coluna de resposta.
Esta pergunta o obvio seguinte: **da para aprender a ligacao entre as duas?**

A resposta curta e "em parte, e bem menos do que parece a primeira vista". A aba
inteira e a construcao desse "menos do que parece".
        """
    )

    st.warning(
        """
**Onde este modelo entra no sistema — e onde nao entra.**

Ele **nao** substitui o motor de similaridade, e a diferenca nao e de qualidade:
e de pergunta. O kNN responde *"a quais ensaios do historico este evento se
parece"* e mostra os vizinhos, que sao evidencia conferivel. A floresta responde
*"que familia e esta"* e devolve um nome, sem mostrar de onde veio.

Ele tambem **nao** escolhe manual, nao decide se prescreve e nao entra no
caminho do LLM. A familia que autoriza o manual continua saindo do
`fault_map.yaml` pelo rotulo — principio 1 —, e os guardrails continuam sendo
`SELECT` e comparacao numerica. Um classificador com a acuracia medida abaixo
nao tem autoridade para fixar o manual de uma conversa inteira.

O lugar dele e o que o projeto ja reservou em `[R2]`: **sinal auxiliar de
confianca**. Quando concorda com o kNN, ha duas leituras independentes
apontando o mesmo lugar. Quando discorda, e alerta — e nao ha criterio para
saber qual das duas tem razao, o que e precisamente o motivo de isso ir para a
tela e nao para dentro de um `if`.
        """
    )

    st.divider()

    # ======================================================================
    # 1. O modelo
    # ======================================================================
    st.header("1. O modelo", divider="gray")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"""
**Random Forest** — {config.CLF_N_ARVORES} arvores de decisao votando.

Uma arvore sozinha e uma sequencia de perguntas de sim/nao (*"o desvio da
kurtosis passa de 0,4?"*), que ela monta olhando os exemplos. Sozinha, ela
decora: aprende os exemplos que viu e erra feio nos novos. A floresta monta
{config.CLF_N_ARVORES} arvores, cada uma vendo um pedaco sorteado dos dados e um
subconjunto sorteado das perguntas possiveis, e a resposta mais votada vence. Os
erros individuais tendem a se cancelar.

**Por que este modelo, e nao outro.** Nao se incomoda com escalas diferentes —
e as nossas colunas vao de 2,5 (kurtosis) a 2000 (rpm). Aguenta classe com
poucos exemplos. E sabe dizer em que se baseou, o que a secao 5 usa.

Este e um dos poucos pontos em que **nada** foi adaptado: e o mesmo modelo, com
a mesma configuracao do projeto de origem.
            """
        )
    with c2:
        st.dataframe(
            pd.DataFrame(
                [
                    {"ajuste": "n_estimators", "valor": config.CLF_N_ARVORES,
                     "o que faz": "Arvores na floresta"},
                    {"ajuste": "random_state", "valor": config.CLF_SEMENTE,
                     "o que faz": "Fixa o sorteio — a tela da o mesmo numero toda vez"},
                    {"ajuste": "n_jobs", "valor": -1,
                     "o que faz": "Usa todos os nucleos"},
                    {"ajuste": "janela", "valor": tamanho,
                     "o que faz": "Leituras por amostra (secao 6)"},
                    {"ajuste": "folds", "valor": config.CLF_N_FOLDS,
                     "o que faz": "Partes da validacao cruzada"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Tudo em `config.py` — nenhum limiar escondido no meio do codigo.")

    st.divider()

    # ======================================================================
    # 2. As duas maneiras de cortar
    # ======================================================================
    st.header("2. As duas maneiras de cortar treino e teste", divider="gray")

    st.markdown(
        """
Nunca se testa o modelo nos mesmos exemplos com que ele estudou — seria dar a
prova com o gabarito. Entao o conjunto e cortado em 5 partes: treina com 4,
testa na que sobrou, repete 5 vezes trocando qual fica de fora, e tira a media.

**O corte pode ser feito de duas maneiras, e e aqui que mora o assunto.**
        """
    )

    c1, c2 = st.columns(2)
    c1.error(
        "**Jeito 1 — sortear amostras soltas**\n\n"
        "`StratifiedKFold`. Embaralha as janelas e reparte.\n\n"
        "Janelas do mesmo evento sao quase identicas: mesma bancada, mesma "
        "montagem, mesma rotacao, minutos de diferenca — e ainda **metade do "
        "conteudo repetido**, porque as janelas se sobrepoem. Ao embaralhar, uma "
        "cai no treino e a outra no teste."
    )
    c2.success(
        "**Jeito 2 — sortear eventos inteiros**\n\n"
        "`StratifiedGroupKFold`, agrupando por `evento`.\n\n"
        "Se o evento 12 caiu no teste, **todas** as janelas dele vao para o "
        "teste, e nenhuma sobra no treino. O modelo e obrigado a opinar sobre "
        "uma medicao que nunca viu — que e a pergunta real: *vai funcionar numa "
        "maquina nova?*"
    )

    st.markdown(
        """
Esta e a exigencia que o projeto ja tinha escrito para a Parte 3 — *"validacao
por grupo: segurar a sessao inteira fora; leave-one-out ingenuo infla a acuracia
por autocorrelacao temporal"*. A tabela abaixo e onde ela vira numero: a
distancia entre as duas notas **mede** a autocorrelacao, em vez de so afirmar
que ela existe.
        """
    )

    with st.spinner("Treinando 10 florestas (5 folds x 2 estrategias)..."):
        resultado = D.r_clf_validacao("janela", tamanho, False)

    aleatoria = resultado["estrategias"]["aleatoria"]
    por_evento = resultado["estrategias"]["por_evento"]
    base = resultado["linha_de_base"]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Jeito 1 — sorteando amostras",
                  f"{aleatoria['acuracia']:.1%}",
                  delta=f"+{resultado['inflacao']:.1%} de inflacao",
                  delta_color="inverse")
        st.error(
            f"**Este numero nao vale.** Nos 5 folds, "
            f"**{aleatoria['eventos_vazados']} eventos** apareceram no treino e no "
            "teste ao mesmo tempo."
        )
    with c2:
        st.metric("Jeito 2 — sorteando eventos",
                  f"{por_evento['acuracia']:.1%}",
                  delta=f"±{por_evento['desvio']:.1%} entre os folds",
                  delta_color="off")
        st.success(
            f"**Este e o numero honesto.** Eventos nos dois lados: "
            f"**{por_evento['eventos_vazados']}**. E o que se deve reportar."
        )

    st.markdown(
        f"""
### O que a distancia de {resultado['inflacao']:.0f} pontos significa

Nao e imprecisao de medida — e a medida exata do quanto o modelo **cola**. Ele
aprendeu a reconhecer **a medicao**, e nao **o defeito**: decorou o que era
especifico daquela montagem, e num ensaio novo nada disso se repete.

E o mesmo achado do projeto de origem, que mediu 92,1% contra 43,8% sobre uma
segmentacao diferente. Reproduzir o efeito com outra regra de agrupamento, outro
mapa de rotulos e outro conjunto de colunas e um resultado por si so: mostra que
o problema esta **nos dados**, e nao numa escolha infeliz de quem preparou.
        """
    )

    st.markdown(f"### Colocando {por_evento['acuracia']:.0%} em perspectiva")

    c1, c2, c3 = st.columns(3)
    c1.metric("Acuracia honesta", f"{por_evento['acuracia']:.1%}")
    c2.metric("Chutar no acaso", f"{base['aleatorio']:.1%}",
              help=f"1 dividido por {base['n_familias']} familias.")
    c3.metric("Responder sempre a mais comum", f"{base['maioria']:.1%}",
              help=f"A familia `{base['familia_mais_comum']}` sozinha.")

    st.markdown(
        f"""
As duas referencias estao ali porque uma so engana. Contra o chute uniforme
({base['aleatorio']:.1%}) qualquer coisa parece boa. A barra de verdade e a
segunda: responder **sempre** `{base['familia_mais_comum']}`, sem olhar dado
nenhum, acerta {base['maioria']:.1%}. O modelo acerta
{por_evento['acuracia'] / base['maioria']:.1f} vezes isso — entao **ha sinal
real sendo capturado**. So que muito menos do que os
{aleatoria['acuracia']:.0%} sugeriam.

Repare tambem no desvio entre os folds: **±{por_evento['desvio']:.1%}**. E
bastante, e tem causa conhecida — com {resultado['n_eventos']} eventos para
{resultado['n_familias']} familias, cada fold testa em poucos ensaios e o
resultado balanca. Diferenca pequena entre configuracoes nao e confiavel, e vale
lembrar disso ao ler os experimentos das secoes 6 e 7.
        """
    )

    if resultado["classes_raras"]:
        st.info(
            "**Familias deixadas de fora da validacao:** "
            f"`{'`, `'.join(resultado['classes_raras'])}`. Cada uma tem menos de "
            f"{resultado['n_folds']} eventos, e nao da para reparti-las em "
            f"{resultado['n_folds']} partes sem deixar alguma vazia. Isso e um "
            "resultado, nao um detalhe: significa que a bancada mediu esses "
            "estados poucas vezes."
        )

    st.markdown("**O detalhe de cada fold**")
    st.dataframe(_tabela_de_folds(resultado), width="stretch", hide_index=True)
    st.caption(
        "A coluna `eventos nos dois lados` e a prova do vazamento. Na estrategia "
        "por evento ela e zero por construcao; na aleatoria ela e a explicacao "
        "inteira da diferenca de acuracia."
    )

    st.divider()
