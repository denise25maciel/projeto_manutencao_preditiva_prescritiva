"""O fluxo inteiro numa tela so. Duas portas de entrada, o mesmo destino.

**Por texto (o caminho comum).** O tecnico descreve o problema. O sistema busca
nos manuais, escolhe o que trata daquilo e fixa. Nao ha lista de falhas para
escolher — descobrir qual e a falha e o que o sistema faz.

**Por evento de sensor (opcional).** Quando ha um JSON de leitura, o kNN o
compara com o historico e indica a familia pelos numeros. O campo `fault`, se
vier, e anotacao a conferir: a familia usada e a que os vizinhos indicam, e a
divergencia vira alerta.

Nos dois casos, uma vez aberto, o manual fica travado e a conversa segue igual.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import _dados as D


D.configurar_pagina("Diagnostico", "🩺")

st.title("🩺 Diagnostico e Conversa")
st.caption("Descreva o problema. O sistema acha o procedimento e conversa sobre ele.")

# ==========================================================================
# Barra lateral
# ==========================================================================
with st.sidebar:
    st.header("Modelo")
    cfg = st.session_state.get("llm_config")
    if cfg:
        st.success(f"`{cfg['provedor']}`\n\n`{cfg['modelo']}`")
    else:
        st.warning("Nenhum modelo escolhido. Abra a tela **Modelo de Linguagem**.")

    usar_llm = st.toggle(
        "Deixar o modelo redigir", value=bool(cfg), disabled=not cfg,
        help="Desligado, a resposta e o texto cru do manual — sem modelo nenhum.",
    )

    st.divider()
    st.header("Busca")
    k_vizinhos = st.slider("Vizinhos consultados", 5, 100, 25, step=5)
    k_trechos = st.slider("Trechos por resposta", 1, 10, 5)
    #analisar
    if st.session_state.get("sessao"):
        st.divider()
        if st.button("Encerrar sessao", width="stretch"):
            for chave in ("sessao", "diagnostico", "evento"):
                st.session_state.pop(chave, None)
            st.rerun()

sessao = st.session_state.get("sessao")

# ==========================================================================
# 1. A entrada
# ==========================================================================
if sessao is None:
    st.subheader("1. O que esta acontecendo")

    aba_texto, aba_sensor = st.tabs(
        ["📝 Descrever o problema", "📈 Evento do sensor (opcional)"]
    )

    # ---------------------------------------------------------------- texto
    with aba_texto:
        st.markdown(
            """
Escreva o que voce esta vendo na maquina. O sistema procura nas documentações da empresa.


Nao precisa saber o nome da falha — descobrir isso e o trabalho do sistema.
Quanto mais concreto o sintoma, melhor: **onde**, **quando** e **o que mudou**.
"""
        )

        descricao = st.text_area(
            "Descricao do problema",
            placeholder="Ex.: o motor esta vibrando e o mancal do lado do acoplamento "
                        "esquentou depois da ultima manutencao...",
            height=140,
        )

        if st.button("Procurar o procedimento", type="primary", key="btn_texto"):
            if not descricao.strip():
                st.warning("Escreva a descricao do problema.")
                st.stop()
            with st.spinner("Procurando nos seis manuais..."):
                nova = D.abrir_conversa_por_texto(
                    descricao, k=8, usar_llm=usar_llm, config_llm=cfg
                )
            st.session_state["sessao"] = nova
            st.session_state.pop("diagnostico", None)
            st.rerun()

    # --------------------------------------------------------------- sensor
    with aba_sensor:
        st.markdown(
            """
Quando existe a leitura do sensor, ela entra aqui. O kNN compara os numeros com
as 166 mil leituras do historico e indica a falha — sem passar pelo texto.

O campo `fault` e opcional: se vier preenchido, e a anotacao do operador, que
sera **confrontada** com o que os vizinhos indicarem.
"""
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            semente = st.number_input(
                "Semente do sorteio", 0, 9999, 0, step=1,
                help="Muda qual leitura real do historico e sorteada.",
            )
        with c2:
            manter_fault = st.checkbox(
                "Incluir a anotacao do operador (`fault`)", value=True,
                help="Desmarque para ver o sistema decidir sem nenhuma pista.",
            )

        exemplo = D.r_evento_de_exemplo(None, int(semente))
        if not manter_fault:
            exemplo = {k: v for k, v in exemplo.items() if k != "fault"}

        texto_json = st.text_area(
            "JSON do evento", value=json.dumps(exemplo, indent=2, ensure_ascii=False),
            height=300,
        )

        if st.button("Diagnosticar pelo sensor", key="btn_sensor"):
            try:
                evento = json.loads(texto_json)
            except json.JSONDecodeError as e:
                st.error(f"JSON invalido: {e}")
                st.stop()

            with st.spinner("Comparando com o historico..."):
                diag = D.diagnosticar(evento, k=k_vizinhos)
                nova = D.abrir_conversa(evento=evento, diagnostico=diag)

            st.session_state["evento"] = evento
            st.session_state["diagnostico"] = diag
            st.session_state["sessao"] = nova
            st.rerun()

    st.stop()

# ==========================================================================
# 2. O que os numeros disseram
# ==========================================================================
diag = st.session_state.get("diagnostico")

if sessao.origem == "texto":
    # ---------------------------------------------------------------------
    # 2a. Aberta por descricao escrita
    # ---------------------------------------------------------------------
    st.subheader("A conversa ate aqui")

    # ---------------------------------------------------------------------
    # A escolha do manual como CONVERSA, nao como painel.
    #
    # O que o tecnico escreveu tem de pesar mais na tela do que o que o sistema
    # respondeu — e a fala dele que conduz. Por isso o sintoma vai em texto
    # normal na mensagem do usuario, e tudo que e maquinaria do sistema (margem,
    # share, pesos) desce para `st.caption`, em cinza pequeno.
    #
    # Nada de `st.error` ou `st.success` aqui: caixa colorida de largura inteira
    # rouba a atencao da mensagem que importa.
    # ---------------------------------------------------------------------
    for i, sintoma in enumerate(sessao.sintomas):
        with st.chat_message("user"):
            st.markdown(sintoma)
            st.caption("o que voce descreveu" if i == 0 else f"sintoma {i + 1}")

    if sessao.situacao == "escolhendo":
        veredito = next(
            (v for v in reversed(sessao.vereditos_abertura) if v.id == "G1T"), None
        )
        d = veredito.detalhe if veredito else {}
        resumo_evidencia = (
            f"margem {d.get('margem', 0):.0%} "
            f"(min {D.config.MARGEM_MINIMA_DOCUMENTO:.0%}) · "
            f"evidencia no 1o {d.get('share', 0):.0%} "
            f"(min {D.config.SHARE_MINIMO_DOCUMENTO:.0%})"
        )

        with st.chat_message("assistant"):
            st.markdown(sessao.aviso_de_pouca_informacao)
            st.caption(resumo_evidencia)

        # -----------------------------------------------------------------
        # A lista inteira, nao os tres primeiros.
        #
        # O peso e a SOMA dos scores dos trechos daquele manual entre os `k`
        # recuperados, entao um documento com quatro trechos medios passa a
        # frente de um com um trecho otimo. Cortar a lista em tres esconderia
        # justamente esse caso; por isso todos aparecem, e ao lado do peso vem
        # o **melhor trecho**, que e o numero que nao sofre desse efeito.
        #
        # Cada candidato mostra o pedaco de manual que o fez aparecer. E o que
        # torna a escolha informada em vez de um sorteio entre codigos: o
        # tecnico le o texto e reconhece — ou nao — a propria maquina.
        # -----------------------------------------------------------------
        st.caption("**Qual destes descreve a sua maquina?**")
        melhores = sessao.melhor_trecho_por_documento

        for doc, peso in sessao.candidatos:
            trecho = melhores.get(doc)
            with st.container(border=True):
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                with c1:
                    st.markdown(f"**{sessao.nomes_candidatos.get(doc, doc)}**")
                    legenda = f"`{doc}` · peso {peso:.2f}"
                    if trecho is not None:
                        legenda += (
                            f" · melhor trecho {trecho.score:.2f} "
                            f"({trecho.referencia})"
                        )
                    st.caption(legenda)
                with c2:
                    if st.button("Seguir", key=f"seguir_{doc}", width="stretch"):
                        st.session_state["sessao"] = D.escolher_documento(sessao, doc)
                        st.rerun()

                if trecho is not None:
                    texto = " ".join(trecho.texto.split())
                    if len(texto) > 320:
                        texto = texto[:320].rsplit(" ", 1)[0] + "..."
                    st.caption(texto)

        # O campo de texto vem DEPOIS da lista: detalhar e a alternativa a
        # escolher, nao o caminho obrigatorio. Sem teto de vezes — quem decide
        # quando parar de detalhar e o tecnico.
        resposta = st.chat_input(
            "Ou conte mais — quanto mais detalhe, maior a chance de eu decidir sozinho"
        )
        if resposta:
            with st.spinner("Refazendo a busca com todos os sintomas..."):
                st.session_state["sessao"] = D.detalhar_sintoma(
                    sessao, resposta, k=8, usar_llm=usar_llm, config_llm=cfg
                )
            st.rerun()

        with st.expander("Por que a lista em vez de escolher o melhor"):
            st.markdown(
                """
`documento_predominante` e um `max`: **sempre** ha um vencedor, mesmo com 1,43
contra 1,39. Era assim que uma descricao ampla travava num manual por acaso — e,
uma vez travado, o manual nao muda mais durante a conversa.

O **G1T** exige duas coisas antes de travar: **margem** (ganhou do 2o) e
**share** (concentra a evidencia). A segunda pega o caso que a primeira nao ve —
pesos [1,0; 0,5; 0,5; 0,5; 0,5] tem margem folgada e ainda deixam quatro manuais
de pe. Reprovar aqui nao e recusa: e passar a decisao para voce.

**Nao ha rodada de investigacao.** O sistema nao gasta perguntas antes de
mostrar a lista, porque quem esta na maquina costuma reconhecer o defeito assim
que le os trechos. Detalhar continua valendo, quantas vezes quiser — se a margem
abrir, o sistema trava sozinho e a lista some.

Nenhum modelo participa desta tela: o ranking e busca vetorial, o empate e um
`if`, os nomes vem do `fault_map.yaml` e a decisao final e sua.
"""
            )

        st.stop()

    # A confirmacao entra como mais uma FALA do sistema, no mesmo tom das
    # outras. Antes eram tres metricas grandes, uma tabela e uma caixa amarela
    # de largura inteira — tudo isso vinha antes da conversa e a soterrava.
    # Agora o detalhe fica em expansor: quem quiser auditar abre.
    if sessao.aberta:
        with st.chat_message("assistant"):
            # Cobrindo varias familias, a frase diz o CONJUNTO e avisa que o
            # tipo nao foi apurado. Escrever um dos nomes mandaria o tecnico
            # olhar uma peca especifica que ninguem determinou — o que a busca
            # por texto identificou foi o procedimento.
            familias_extra = (
                " Ele cobre "
                + ", ".join(f"`{f}`" for f in sessao.familias_do_documento)
                + " — o procedimento e o mesmo para todas, e qual delas e o seu"
                " caso nao foi apurado."
                if len(sessao.familias_do_documento) > 1 else ""
            )
            st.markdown(
                f"Encontrei o procedimento: **{sessao.manual}** "
                f"(`{sessao.assunto}`).{familias_extra}"
            )
            st.caption(
                f"🔒 travado ate o fim da conversa · melhor trecho "
                f"{max(t.score for t in sessao.trechos_de_abertura):.3f}"
                if sessao.trechos_de_abertura else "🔒 travado ate o fim da conversa"
            )

    # A separacao da fala em sintomas e a unica coisa que o modelo fez ate aqui,
    # e ela muda o resultado da busca — entao fica a vista, com o antes e o
    # depois. Se ele picotou errado, o tecnico ve na hora, em vez de descobrir
    # pelo manual estranho que apareceu.
    if len(sessao.sintomas) > 1 or sessao.nota_sintomas:
        with st.expander(
            f"Como a sua descricao foi lida — {len(sessao.sintomas)} sintoma(s)"
        ):
            st.markdown("**Voce escreveu**")
            st.code(sessao.descricao or "—", language="text")
            st.markdown("**O sistema procurou por**")
            for s in sessao.sintomas:
                st.markdown(f"- {s}")
            if sessao.nota_sintomas:
                st.caption(sessao.nota_sintomas)
            st.caption(
                "Cada sintoma vira uma busca, e cada trecho fica com o **maior** "
                "score entre elas. Numa frase so, o vetor cai perto da media dos "
                "sintomas — e a media nao e nenhum deles: a secao 'Mancal "
                "Aquecido' seria derrubada por falar pouco de vibracao. Quem "
                "separou foi o modelo, com saida estruturada; o que ele devolveu "
                "passou por uma conferencia de codigo, que descarta sintoma que "
                "nao aparece no que voce escreveu."
            )

    with st.expander("Ver a evidencia que apontou este procedimento"):
        if sessao.trechos_de_abertura:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"documento": t.documento_id, "secao": t.numero,
                         "pagina": t.pagina or "—", "titulo": t.titulo,
                         "tipo": t.campo or "—", "similaridade": round(t.score, 4)}
                        for t in sessao.trechos_de_abertura
                    ]
                ),
                width="stretch", hide_index=True,
            )
            # Numero solto nao se interpreta: 0,42 e bom ou ruim? A regua vem
            # junto — o teto teorico, o minimo do G4 e o que esta tabela
            # alcancou. Sem isso o tecnico so consegue comparar as linhas entre
            # si, e nao sabe se a melhor delas ja e fraca.
            scores = [t.score for t in sessao.trechos_de_abertura]
            st.caption(
                f"**Escala:** a similaridade e o cosseno entre a sua descricao e "
                f"o trecho, de −1 a +1 · **+1** = mesmo texto, **0** = nada em "
                f"comum · minimo exigido pelo G4: **{D.SCORE_MINIMO_CHUNK:.2f}**"
            )
            st.caption(
                f"**Nesta busca:** melhor {max(scores):.3f} · pior "
                f"{min(scores):.3f} · {sum(s >= D.SCORE_MINIMO_CHUNK for s in scores)} "
                f"de {len(scores)} acima do minimo. Pergunta curta contra secao "
                f"inteira de manual raramente passa de 0,7 — nao espere valores "
                f"perto de 1."
            )
        st.markdown(
            """
**Este caminho e mais fraco que o do sensor, e vale saber por que.**

O desenho do projeto manda filtrar primeiro pela familia e so depois buscar por
semelhanca — assim, falha sem manual nao tem onde ser procurada e a recusa e
estrutural. Entrando por texto **nao ha familia ainda**, entao a ordem se inverte:
busca-se em todos os manuais e o documento aparece do resultado.

Busca por semelhanca **nunca volta vazia**. A unica trava aqui e o score minimo
(G4) — por isso ele esta a vista na tabela. Score baixo em todas as linhas
significa que nenhum procedimento trata do que voce descreveu, ainda que algo
tenha sido devolvido.

Da segunda pergunta em diante a conversa volta ao caminho normal, filtrada
dentro do manual fixado.
"""
        )

elif diag is None:
    st.subheader("2. Sessao aberta")
    st.caption("Sem diagnostico de sensor nesta sessao.")

else:
    # ---------------------------------------------------------------------
    # 2b. Aberta por evento de sensor
    # ---------------------------------------------------------------------
    st.subheader("2. O que os numeros dizem")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Falha indicada", diag.familia or "—")
    c2.metric("Concordancia", f"{diag.confianca:.0%}",
              delta=f"{diag.votos} de {diag.k} vizinhos", delta_color="off")
    c3.metric("Episodios distintos", diag.n_episodios)
    c4.metric("Rotacao dos vizinhos",
              f"{diag.rpm_predominante:.0f} rpm" if diag.rpm_predominante else "—")

    st.caption(
        "**Episodios distintos e o numero que importa**, nao a contagem de "
        "vizinhos. Leituras seguidas do mesmo ensaio sao quase identicas: 25 "
        "vizinhos de um episodio so seriam uma confirmacao, nao 25. Aqui sao "
        f"{diag.n_episodios}."
    )

    e, d = st.columns([1, 1])
    with e:
        st.markdown("**Como os vizinhos se distribuem**")
        st.dataframe(diag.distribuicao, width="stretch", hide_index=True)
    with d:
        st.markdown("**Distancia**")
        st.markdown(
            f"""
- ao vizinho mais proximo: **{diag.distancia_min:.4f}**
- media dos {diag.k} vizinhos: **{diag.distancia_media:.4f}**
- limiar do G1: **{diag.limiar_g1:.4f}**

O limiar sai do proprio historico: a distancia tipica entre uma leitura e sua
vizinha mais parecida, no percentil 99. Ele barra o **absurdo** — outra maquina,
sensor trocado —, nao decide entre falhas parecidas.
"""
        )

    # --- confronto com a anotacao do operador ----------------------------
    if diag.rotulo_do_operador:
        if diag.divergiu:
            st.error(
                f"**Divergencia.** O operador anotou `{diag.rotulo_do_operador}` "
                f"(familia `{diag.familia_do_operador}`), mas os vizinhos indicam "
                f"`{diag.familia}`.\n\n"
                "O sistema seguiu os **numeros**. A anotacao e informacao, nao "
                "ordem — e a divergencia e justamente o que vale reportar."
            )
        else:
            st.success(
                f"**Confere.** O operador anotou `{diag.rotulo_do_operador}` e a "
                f"similaridade tambem indica `{diag.familia}`."
            )
    else:
        st.info(
            f"**O JSON veio sem `fault`.** A familia `{diag.familia}` foi inferida "
            "so pelos numeros — nenhuma pista humana foi usada."
        )

    with st.expander("Os vizinhos, um a um"):
        st.dataframe(
            diag.vizinhos[["distancia", "rotulo", "familia", "evento", "rpm"]]
            .round({"distancia": 4}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "`evento` e o episodio a que aquela leitura pertence. Vizinhos "
            "repetindo o mesmo numero de episodio sao a mesma ocorrencia vista "
            "varias vezes."
        )

    st.warning(
        """
**O quanto confiar nisto.** Com validacao por grupo — segurando o rotulo inteiro
fora do treino — a acuracia e baixa. Nao e defeito da implementacao: o
`banner.csv` traz medidas agregadas (RMS, pico, kurtosis), e o manual de
rolamentos diagnostica por frequencias de defeito (BPFO, BPFI, BSF), que exigem
**espectro**. O espectro nao esta no arquivo.

Por isso o resultado vem com a distribuicao inteira e o numero de episodios a
vista, em vez de um rotulo sozinho com ar de certeza.
"""
    )

# ==========================================================================
# Os guardrails da abertura — em expansor, nao em destaque
# ==========================================================================
#
# Eram seis caixas verdes/vermelhas de largura inteira no meio da tela. Sao
# informacao de auditoria, nao fala da conversa: quem quiser conferir abre.
selo = " ".join(
    f"{'✓' if v.passou else '✗'} {v.id}" for v in sessao.vereditos_abertura
)
with st.expander(f"Guardrails da abertura — {selo}"):
    for v in sessao.vereditos_abertura:
        st.markdown(f"**{v.id}** {'✓' if v.passou else '✗'} — {v.mensagem}")

if not sessao.aberta:
    with st.chat_message("assistant"):
        st.markdown(f"**Nao posso seguir com esta conversa.** {sessao.motivo}")
        st.caption(
            "O modelo de linguagem nao foi chamado — e nao seria. Quem barrou "
            "foi um `SELECT` no catalogo, nao um julgamento do modelo. Busca "
            "por semelhanca nunca volta vazia, entao ela nao pode responder "
            "'existe documento?'. Use **Encerrar sessao** na barra lateral "
            "para comecar outra."
        )
    st.stop()

# ==========================================================================
# A conversa
# ==========================================================================

# Achar o manual nao e resposta. Assim que ele e fixado, o sistema ja responde
# sozinho o que o tecnico ia perguntar de qualquer jeito: qual e o problema,
# quais os sintomas e como corrigir.
#
# Roda uma vez so — depois `sessao.turnos` deixa de estar vazio e a condicao
# nao volta a valer, mesmo com os reruns do Streamlit.
if not sessao.turnos:
    with st.spinner("Lendo o procedimento..."):
        D.resumo_de_abertura(sessao, usar_llm=usar_llm, config_llm=cfg,
                             k=k_trechos + 1)
    st.rerun()

for turno in sessao.turnos:
    # O turno de abertura nao teve pergunta do tecnico — desenhar um balao de
    # usuario com a pergunta que o sistema fez a si mesmo seria mentira visual.
    if not turno.abertura:
        with st.chat_message("user"):
            st.markdown(turno.pergunta)

    with st.chat_message("assistant"):
        # Recusa e degradacao entram como texto e legenda, nao como caixa
        # colorida: a fala do tecnico e que conduz a conversa, e um bloco
        # amarelo de largura inteira gritaria mais alto que ela.
        if turno.recusado:
            st.markdown(turno.resposta)
            st.caption("🚫 recusado por codigo · o modelo nao foi chamado")
            continue

        # `para_exibir` so rebaixa titulo e tira marcador de conversao. O texto
        # gravado no turno continua intacto — a aba de auditoria o mostra cru.
        if turno.degradou:
            st.markdown(D.para_exibir(turno.texto_para_historico))
            st.caption(
                "⚠️ o texto acima e o do manual, sem prosa — a redacao do modelo "
                "foi reprovada no G5 nas duas tentativas"
            )
        else:
            st.markdown(D.para_exibir(turno.resposta))

        if turno.trechos:
            st.caption("**Fontes:** " + " · ".join(f"`{r}`" for r in turno.referencias))

            with st.expander("Ver o texto de cada fonte"):
                # Mesma regua da tela de evidencia: cosseno de −1 a +1, com o
                # minimo do G4 ao lado. Score 0,00 aparece quando os trechos
                # vieram de `SELECT` por tipo de secao — ali nao houve
                # semelhanca calculada, e inventar um numero seria mentir.
                st.caption(
                    f"Similaridade = cosseno da pergunta com o trecho, de −1 a "
                    f"+1 · minimo do G4: **{D.SCORE_MINIMO_CHUNK:.2f}** · "
                    f"0,000 = secao escolhida por tipo, sem calculo de semelhanca"
                )
                for t in turno.trechos:
                    pag = f"pagina {t.pagina}" if t.pagina else "pagina nao disponivel"
                    st.markdown(
                        f"**{t.documento_id}, secao {t.numero} — {t.titulo}**  \n"
                        f"*{pag} · tipo: {t.campo or '—'} · "
                        f"similaridade {t.score:.3f}*"
                    )
                    # O `>` vai em CADA linha, nao so na primeira: um `>` solto
                    # no inicio nao alcanca o titulo que vem depois, e o bloco
                    # saia metade citado, metade nao.
                    st.markdown(
                        "\n".join(
                            f"> {linha}"
                            for linha in D.para_exibir(t.texto).splitlines()
                        )
                    )
                    st.divider()

        selos = [f"{'✓' if v.passou else '✗'} {v.id}" for v in turno.vereditos]
        if turno.usou_llm:
            selos += [
                f"{turno.provedor}/{turno.modelo}",
                f"{turno.tokens_entrada}+{turno.tokens_saida} tokens",
            ]
            if turno.tentativas > 1:
                selos.append(f"{turno.tentativas} tentativas")
        else:
            selos.append("sem modelo de linguagem")
        selos.append(f"{turno.segundos}s")
        st.caption(" · ".join(selos))

pergunta = st.chat_input("Pergunte sobre o procedimento...")

if pergunta:
    with st.chat_message("user"):
        st.markdown(pergunta)
    with st.chat_message("assistant"):
        with st.spinner("Buscando no manual..."):
            D.responder_turno(sessao, pergunta, usar_llm=usar_llm,
                              config_llm=cfg, k=k_trechos)
    st.rerun()

# ==========================================================================
# 5. Auditoria
# ==========================================================================
if sessao.turnos:
    st.divider()
    st.subheader("5. Auditoria")

    t1, t2 = st.tabs(["Historico verificado", "O que foi enviado ao modelo"])

    with t1:
        st.markdown(
            """
O historico enviado ao modelo **nao e o que esta na tela** — e o que foi
verificado:

- resposta que passou no G5 entra como `AIMessage`, isto e, como fala do modelo
- resposta reprovada **nao entra**; no lugar dela vai o trecho do manual, e como
  `SystemMessage` — porque aquele texto e do procedimento, e o modelo nunca o
  escreveu. Marca-lo como fala dele seria dar-lhe uma afirmacao emprestada
- recusa nao entra de forma alguma

Sem isso, um deslize no turno 2 viraria contexto no turno 3, e o modelo o
trataria como fato ja estabelecido.
"""
        )
        historico = sessao.historico_para_prompt()
        if not historico:
            st.caption("Nada ainda — nenhum turno produziu conteudo verificado.")

        # O tipo de cada mensagem na frente do texto: e ele que mostra que a
        # conversa chega estruturada, e nao achatada num paragrafo so.
        SELO = {
            "human": ("🧑‍🔧", "HumanMessage", "pergunta do tecnico"),
            "ai": ("🤖", "AIMessage", "resposta verificada no G5"),
            "system": ("📄", "SystemMessage", "texto do manual, nao fala do modelo"),
        }
        for m in historico:
            icone, classe, papel = SELO.get(m.type, ("•", m.type, ""))
            st.markdown(f"{icone} **`{classe}`** — {papel}")
            texto = str(m.content)
            st.code(texto[:800] + ("..." if len(texto) > 800 else ""), language="text")

    with t2:
        if sessao.turnos[-1].prompt:
            st.caption(
                "O texto exato do ultimo turno. Nada alem disto: sem internet, "
                "sem os outros manuais, sem memoria de conversas anteriores."
            )
            st.code(sessao.turnos[-1].prompt, language="text")
        else:
            st.caption("O ultimo turno parou antes de chegar ao modelo.")
