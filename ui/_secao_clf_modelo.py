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

    # ======================================================================
    # 3. Onde o modelo erra
    # ======================================================================
    st.header("3. Onde o modelo erra", divider="gray")

    st.markdown(
        f"""
{por_evento['acuracia']:.0%} de media nao diz **em que** ele erra, e essa e a
informacao que decide se o modelo serve para alguma coisa. Errar entre duas
familias que compartilham manual custa pouco; errar entre um defeito e o estado
normal custa caro.

Tudo abaixo vem da estrategia honesta. F1 macro:
**{por_evento['f1_macro']:.1%}** — proximo da acuracia, o que indica que o
acerto nao esta concentrado so nas familias grandes.
        """
    )

    por_familia = D.r_clf_por_familia("janela", tamanho, False, "por_evento")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.altair_chart(
            alt.Chart(por_familia)
            .mark_bar()
            .encode(
                x=alt.X("acuracia:Q", title="acuracia honesta",
                        axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("familia:N", sort="x", title=None),
                color=alt.Color(
                    "acuracia:Q",
                    scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                    legend=None,
                ),
                tooltip=["familia", alt.Tooltip("acuracia:Q", format=".1%"),
                         "n_amostras", "confundida_com", "n_confusoes"],
            )
            .properties(height=max(240, 24 * len(por_familia))),
            width="stretch",
        )
    with c2:
        st.dataframe(
            por_familia,
            width="stretch",
            hide_index=True,
            column_config={
                "acuracia": st.column_config.ProgressColumn(
                    "acuracia", format="%.1f%%", min_value=0, max_value=1
                ),
            },
        )
        st.caption(
            "`n_amostras` fica ao lado de proposito: 100% em 6 amostras nao e a "
            "mesma coisa que 100% em 600. `confundida_com` e o erro mais "
            "frequente de cada familia."
        )

    st.markdown("**A matriz de confusao — o que era, contra o que o modelo disse**")

    confusao = D.r_clf_confusao("janela", tamanho, False, "por_evento", True)
    st.altair_chart(heatmap_confusao(confusao), width="stretch")
    st.caption(
        "Cada linha soma 100%: e a fatia das amostras daquela familia que foi "
        "para cada resposta. A diagonal e o acerto. Somar os 5 folds e legitimo "
        "porque cada amostra e testada exatamente uma vez."
    )

    st.divider()

    # ======================================================================
    # 4. Experimentar com um evento
    # ======================================================================
    st.header("4. Perguntar ao modelo sobre um evento", divider="gray")

    st.markdown(
        """
Escolha um evento do historico. O modelo e treinado **sem ele** e so entao
recebe a pergunta — o mesmo criterio da estrategia honesta.

Perguntar a um modelo treinado no conjunto inteiro sobre um evento que estava
nesse conjunto nao demonstraria nada: ele reconheceria o que decorou, e a tela
mostraria uma certeza que a secao 2 ja provou nao existir.
        """
    )

    consultaveis = D.r_clf_eventos_consultaveis(tamanho)

    c1, c2 = st.columns([2, 3])
    with c1:
        familia_alvo = st.selectbox(
            "Familia",
            sorted(consultaveis["familia"].unique()),
            key="clf_familia_alvo",
        )
    do_alvo = consultaveis[consultaveis["familia"] == familia_alvo]
    with c2:
        evento = st.selectbox(
            "Evento",
            do_alvo["evento"].tolist(),
            format_func=lambda e: (
                f"evento {e} — rotulo `{do_alvo.loc[do_alvo['evento'] == e, config.COLUNA_ROTULO].iloc[0]}`"
                f" — {int(do_alvo.loc[do_alvo['evento'] == e, 'n_leituras'].iloc[0])} leituras"
            ),
            key="clf_evento_alvo",
        )

    # O resultado fica no `session_state`, e nao no retorno do botao. Um
    # `st.button` devolve `True` so no rerun do clique; sem guardar, qualquer
    # outra interacao da tela apagaria a resposta que a pessoa acabou de pedir.
    if st.button("Segurar este evento fora e perguntar", type="primary"):
        st.session_state["clf_evento_consultado"] = int(evento)

    consultado = st.session_state.get("clf_evento_consultado")
    if consultado is not None:
        r = D.r_clf_previsao_honesta(consultado, tamanho, False)

        st.caption(f"Resultado para o **evento {consultado}**.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Familia verdadeira", r["familia_verdadeira"])
        c2.metric("O modelo respondeu", r["previsto"])
        c3.metric("Confianca", f"{r['ranking'].iloc[0]['probabilidade']:.0%}")

        if r["acertou"]:
            st.success(
                f"**Acertou.** Treinou em {r['n_amostras_treino']:,} amostras sem "
                f"ver nenhuma das {r['n_janelas']} janelas deste evento."
                .replace(",", ".")
            )
        else:
            st.error(
                f"**Errou.** Disse `{r['previsto']}` para um evento de "
                f"`{r['familia_verdadeira']}`. Com "
                f"{por_evento['acuracia']:.0%} de acuracia honesta, isto acontece "
                "na maioria das vezes — a tela mostra o caso real, nao o caso "
                "escolhido."
            )

        st.altair_chart(
            alt.Chart(r["ranking"].head(8))
            .mark_bar()
            .encode(
                x=alt.X("probabilidade:Q", title="probabilidade media",
                        axis=alt.Axis(format="%")),
                y=alt.Y("familia:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.familia == r["familia_verdadeira"],
                    alt.value("#2a9d8f"),
                    alt.value("#adb5bd"),
                ),
                tooltip=["familia", alt.Tooltip("probabilidade:Q", format=".1%")],
            )
            .properties(height=240),
            width="stretch",
        )
        st.caption(
            f"Em verde, a familia verdadeira. As {r['n_janelas']} janelas do "
            "evento votam por **media de probabilidade**, nao por maioria: a "
            "media preserva a duvida, enquanto a maioria devolveria a vencedora "
            "com cara de certeza."
        )

    st.divider()

    # ======================================================================
    # 5. Em que o modelo se baseia
    # ======================================================================
    st.header("5. Em que o modelo se baseia", divider="gray")

    st.markdown(
        """
A floresta sabe dizer quais colunas mais pesaram nas decisoes. Vale como
sanidade: se o peso estivesse concentrado em algo que nao e vibracao, seria
sinal de atalho.
        """
    )

    if st.button("Treinar a floresta e ver os pesos"):
        st.session_state["clf_mostrar_pesos"] = True

    if st.session_state.get("clf_mostrar_pesos"):
        modelo = D.clf_modelo(tamanho, False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Por coluna de origem** (soma das 5 estatisticas)")
            por_coluna = modelo.importancia_por_coluna()
            st.altair_chart(
                alt.Chart(por_coluna)
                .mark_bar()
                .encode(
                    x=alt.X("importancia:Q", title="importancia somada"),
                    y=alt.Y("coluna:N", sort="-x", title=None),
                    tooltip=["coluna", alt.Tooltip("importancia:Q", format=".3f")],
                )
                .properties(height=max(240, 22 * len(por_coluna))),
                width="stretch",
            )
        with c2:
            st.markdown("**Por feature individual** (as 20 maiores)")
            st.dataframe(modelo.importancia(20), width="stretch", hide_index=True)

        st.caption(
            "As duas visoes juntas porque a de baixo sozinha engana: uma coluna "
            "forte espalhada entre as 5 estatisticas parece fraca ao lado de uma "
            "que concentra tudo numa so."
        )

    st.divider()

    # ======================================================================
    # 6. Experimento — tamanho da janela
    # ======================================================================
    st.header("6. Experimento — o tamanho da janela", divider="gray")

    st.markdown(
        f"""
A janela de **{tamanho} leituras** foi escolhida na aba anterior por caber no
evento tipico. Mas cabe perguntar se um tamanho diferente daria um modelo melhor
— e a resposta tem duas partes que costumam ser confundidas.
        """
    )

    st.info(
        f"Este experimento treina 10 florestas para **cada** tamanho testado "
        f"({len(config.CLF_JANELAS_TESTADAS)} tamanhos, "
        f"{10 * len(config.CLF_JANELAS_TESTADAS)} florestas). **Leva alguns "
        "minutos.** O resultado fica guardado depois da primeira vez."
    )

    if st.button("Rodar o experimento de janela"):
        exp = D.r_clf_experimento_janela(config.CLF_JANELAS_TESTADAS)
        st.session_state["clf_exp_janela"] = exp

    exp = st.session_state.get("clf_exp_janela")
    if exp is not None:
        st.dataframe(
            exp,
            width="stretch",
            hide_index=True,
            column_config={
                "minutos": st.column_config.NumberColumn(
                    "minutos de motor", format="%.1f"),
                "acuracia_honesta": st.column_config.NumberColumn(
                    "acuracia honesta", format="%.1f%%"),
                "acuracia_inflada": st.column_config.NumberColumn(
                    "acuracia inflada", format="%.1f%%"),
                "inflacao": st.column_config.NumberColumn(
                    "inflacao", format="%.1f%%"),
                "pct_eventos_descartados": st.column_config.ProgressColumn(
                    "% eventos descartados", format="%.1f%%",
                    min_value=0, max_value=100),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A acuracia honesta quase nao se mexe**")
            st.altair_chart(
                alt.Chart(exp)
                .mark_line(point=True)
                .encode(
                    x=alt.X("janela:O", title="tamanho da janela"),
                    y=alt.Y("acuracia_honesta:Q", title="acuracia honesta",
                            axis=alt.Axis(format="%")),
                    tooltip=["janela",
                             alt.Tooltip("acuracia_honesta:Q", format=".1%")],
                )
                .properties(height=240),
                width="stretch",
            )
        with c2:
            st.markdown("**Mas o descarte de eventos dispara**")
            st.altair_chart(
                alt.Chart(exp)
                .mark_bar()
                .encode(
                    x=alt.X("janela:O", title="tamanho da janela"),
                    y=alt.Y("pct_eventos_descartados:Q",
                            title="% dos eventos jogados fora"),
                    tooltip=["janela", "eventos_usados",
                             alt.Tooltip("pct_eventos_descartados:Q",
                                         format=".1f")],
                )
                .properties(height=240),
                width="stretch",
            )

        melhor = exp.loc[exp["acuracia_honesta"].idxmax()]
        faixa = exp["acuracia_honesta"].max() - exp["acuracia_honesta"].min()
        st.markdown(
            f"""
### O que o experimento diz

A melhor acuracia honesta medida foi **{melhor['acuracia_honesta']:.1%}**, com
janela de **{int(melhor['janela'])}**. Mas a diferenca entre o melhor e o pior
tamanho e de apenas **{faixa:.1%}** — menor que o desvio de
**±{por_evento['desvio']:.1%}** entre os folds de uma unica configuracao. Ou
seja, **esta dentro do proprio ruido da medicao**, e nao sustenta escolher um
tamanho pelo outro.

O que **de fato** muda com o tamanho e a coluna de descarte, e ali a diferenca e
enorme. E por isso que o criterio da escolha e o descarte, e nao a acuracia: o
evento descartado nao aparece em metrica nenhuma, mas leva familias inteiras
embora do conjunto.

**A licao maior:** mexer no tamanho da janela nao resolve o problema deste
sistema. O que trava o resultado e outra coisa — o modelo aprender a medicao em
vez do defeito.
            """
        )

    st.divider()

    # ======================================================================
    # 7. Experimento — o regime de operacao
    # ======================================================================
    st.header("7. Experimento — rpm e temperatura como feature", divider="gray")

    regime = [c for c in D.r_clf_colunas(True) if c not in D.r_clf_colunas(False)]

    st.markdown(
        f"""
`{'` e `'.join(regime)}` ficam de fora por padrao, com o argumento de que sao
**regime de operacao, nao sintoma**: o mesmo defeito a 500 e a 2000 rpm tem
assinaturas diferentes, e deixar o regime entrar faria o modelo reconhecer o
ensaio.

O projeto de origem usava as duas e depois registrou como limitacao numero 1 que
*"o modelo aprendeu o ensaio, nao o defeito — a temperatura ambiente do dia, a
rotacao exata"*. Argumento plausivel nao e medida. Este experimento roda as duas
configuracoes lado a lado.

**O sinal a procurar** nao e a acuracia honesta subir ou descer: e a **inflacao**
crescer. Feature que ajuda a reconhecer o ensaio, e nao a falha, aparece assim —
melhora a nota falsa sem melhorar a verdadeira.
        """
    )

    if st.button("Rodar o experimento de regime"):
        st.session_state["clf_exp_regime"] = D.r_clf_experimento_regime(tamanho)

    exp_r = st.session_state.get("clf_exp_regime")
    if exp_r is not None:
        st.dataframe(
            exp_r,
            width="stretch",
            hide_index=True,
            column_config={
                "regime_como_feature": st.column_config.CheckboxColumn(
                    "regime entra?"),
                "acuracia_honesta": st.column_config.NumberColumn(
                    "acuracia honesta", format="%.1f%%"),
                "acuracia_inflada": st.column_config.NumberColumn(
                    "acuracia inflada", format="%.1f%%"),
                "inflacao": st.column_config.NumberColumn(
                    "inflacao", format="%.1f%%"),
            },
        )

        sem = exp_r[~exp_r["regime_como_feature"]].iloc[0]
        com = exp_r[exp_r["regime_como_feature"]].iloc[0]
        d_honesta = com["acuracia_honesta"] - sem["acuracia_honesta"]
        d_inflacao = com["inflacao"] - sem["inflacao"]

        c1, c2 = st.columns(2)
        c1.metric("Efeito na acuracia honesta", f"{d_honesta:+.1%}",
                  help="Incluindo o regime, em relacao a deixa-lo de fora.")
        c2.metric("Efeito na inflacao", f"{d_inflacao:+.1%}",
                  delta_color="inverse")

        st.markdown(
            f"""
### Leitura honesta do resultado

Incluir o regime muda a acuracia honesta em **{d_honesta:+.1%}** e a inflacao em
**{d_inflacao:+.1%}**. As duas mudancas sao **pequenas** — bem menores que o
desvio de ±{por_evento['desvio']:.1%} entre folds —, entao o mais correto a
dizer e que o experimento **nao decide a questao**: nao ha ganho em incluir o
regime, e o sinal de que ele piora a inflacao existe mas e fraco demais para se
apoiar nele.

O que sustenta a exclusao continua sendo o **argumento**, nao esta medida: rpm
faz parte da identidade do evento (`config.COLUNAS_QUEBRA_EVENTO`), e uma
variavel que define o grupo nao deveria tambem servir para prever a classe
dentro dele. O experimento fica registrado pelo que ele e — a verificacao de que
excluir o regime nao esta custando acuracia.
            """
        )

    st.divider()

    # ======================================================================
    # 8. Limitacoes
    # ======================================================================
    st.header("8. As limitacoes, sem maquiagem", divider="gray")

    st.markdown(
        f"""
**1. O modelo aprende a medicao, nao o defeito.** E a limitacao principal, e as
outras derivam dela. A distancia de {resultado['inflacao']:.0f} pontos entre a
nota inflada e a honesta mostra que boa parte do acerto vem de reconhecer
condicoes especificas de cada montagem. Numa maquina nova, isso nao se
transfere.

**2. Ha poucos eventos.** {resultado['n_eventos']} para
{resultado['n_familias']} familias, e o desvio de ±{por_evento['desvio']:.1%}
entre folds e consequencia direta disso. Toda comparacao entre configuracoes
carrega essa margem — e por isso as secoes 6 e 7 concluem "nao decide" em vez de
apontar um vencedor.

**3. Algumas familias nao entram na validacao.**
{'`' + '`, `'.join(resultado['classes_raras']) + '`' if resultado['classes_raras'] else 'Nenhuma, nesta configuracao.'}
{' Se detectar alguma delas for requisito, esta configuracao nao atende.' if resultado['classes_raras'] else ''}

**4. Ajustar nao adianta.** Os dois experimentos acima mexem no tamanho da
janela e no conjunto de colunas, e nenhum move a acuracia honesta para fora do
proprio ruido. O gargalo nao e configuracao.
        """
    )

    st.markdown("### O que teria chance de melhorar de verdade")

    st.markdown(
        """
Todos os caminhos abaixo atacam a limitacao numero 1 — fazer o modelo enxergar o
defeito, e nao a montagem:

**Comparar cada leitura com a linha de base da propria maquina**, em vez do
valor absoluto. Hoje o modelo ve "vibracao de 3,2 mm/s". Se visse "vibracao 40%
acima do normal desta maquina", o numero deixaria de carregar a assinatura da
montagem. Este projeto tem a familia `normal` medida justamente para servir de
referencia — e a mesma ideia que a **Parte 7** usa ao comparar assinatura contra
`normal` em vez da mediana global.

**Coletar mais ensaios**, principalmente das familias raras. E o caminho mais
lento e o mais garantido.

**Usar o modelo como o que ele e:** sinal auxiliar. Concordancia entre floresta
e kNN vale mais que qualquer uma das duas sozinha, e discordancia e informacao
util — sem que nenhuma das duas precise decidir nada por conta propria.
        """
    )

    st.info(
        """
**Onde cada coisa mora no codigo.** `mp/classificacao/amostras.py` faz a
preparacao da aba anterior; `mp/classificacao/modelo.py` e a floresta e a
consulta; `mp/classificacao/validacao.py` e tudo desta aba. A interface nao
calcula nada — so chama e desenha, como as demais telas do projeto.
        """
    )


# --------------------------------------------------------------------------
# Auxiliares de desenho
# --------------------------------------------------------------------------


def _tabela_de_folds(resultado: dict) -> pd.DataFrame:
    """Os 5 folds das duas estrategias, lado a lado."""
    linhas = []
    for nome, dados in resultado["estrategias"].items():
        rotulo = "sorteando eventos" if nome == "por_evento" else "sorteando amostras"
        for f in dados["folds"]:
            linhas.append(
                {
                    "estrategia": rotulo,
                    "fold": f["fold"],
                    "amostras treino": f["n_treino"],
                    "amostras teste": f["n_teste"],
                    "eventos treino": f["eventos_treino"],
                    "eventos teste": f["eventos_teste"],
                    "eventos nos dois lados": f["eventos_vazados"],
                    "acuracia": f["acuracia"],
                }
            )
    return pd.DataFrame(linhas)


def heatmap_confusao(confusao: pd.DataFrame) -> alt.Chart:
    """Matriz de confusao normalizada por linha.

    Publica porque a aba de execucao desenha a mesma matriz no laudo dela. Duas
    copias divergiriam na primeira vez que uma delas fosse ajustada.
    """
    longo = (
        confusao.reset_index(names="verdadeira")
        .melt(id_vars="verdadeira", var_name="prevista", value_name="fracao")
    )
    ordem = list(confusao.index)

    base = alt.Chart(longo).encode(
        x=alt.X("prevista:N", sort=ordem, title="o modelo disse",
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("verdadeira:N", sort=ordem, title="era"),
    )

    celulas = base.mark_rect().encode(
        color=alt.Color("fracao:Q", title="fracao",
                        scale=alt.Scale(scheme="blues", domain=[0, 1])),
        tooltip=["verdadeira", "prevista", alt.Tooltip("fracao:Q", format=".1%")],
    )

    # So os valores relevantes viram texto: a matriz tem 169 celulas e a maioria
    # e zero, que escrito por extenso vira ruido visual.
    rotulos = base.mark_text(fontSize=9).encode(
        text=alt.condition(
            alt.datum.fracao >= 0.05,
            alt.Text("fracao:Q", format=".0%"),
            alt.value(""),
        ),
        color=alt.condition(
            alt.datum.fracao > 0.5, alt.value("white"), alt.value("#333")
        ),
    )

    return (celulas + rotulos).properties(height=380)
