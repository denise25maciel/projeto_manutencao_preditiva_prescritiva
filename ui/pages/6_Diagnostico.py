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
Escreva o que voce esta vendo na maquina. O sistema procura nos seis
procedimentos, escolhe o que trata daquilo e abre a conversa nele.

Nao precisa saber o nome da falha — descobrir isso e o trabalho do sistema.
Quanto mais concreto o sintoma, melhor: **onde**, **quando** e **o que mudou**.
"""
        )

        exemplos = [
            "O motor esta vibrando muito na direcao radial e o mancal esquentou.",
            "Depois que trocamos o acoplamento, apareceu vibracao no sentido axial.",
            "A correia esta escorregando e faz um chiado quando parte.",
            "Tem um ruido de impacto no rolamento, parece batida a cada volta.",
            "A polia parece torta, balanca quando gira devagar.",
        ]
        escolha = st.selectbox("Exemplos", ["(escrever a minha)"] + exemplos)

        descricao = st.text_area(
            "Descricao do problema",
            value="" if escolha.startswith("(") else escolha,
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
    # A investigacao como CONVERSA, nao como painel.
    #
    # O que o tecnico escreveu tem de pesar mais na tela do que o que o sistema
    # respondeu — e a fala dele que conduz. Por isso o sintoma vai em texto
    # normal na mensagem do usuario, e tudo que e maquinaria do sistema (margem,
    # share, candidatos, rodada) desce para `st.caption`, em cinza pequeno.
    #
    # Nada de `st.error` ou `st.success` aqui: caixa colorida de largura inteira
    # rouba a atencao da mensagem que importa.
    # ---------------------------------------------------------------------
    for i, sintoma in enumerate(sessao.sintomas):
        with st.chat_message("user"):
            st.markdown(sintoma)
            st.caption("o que voce descreveu" if i == 0 else f"sintoma {i + 1}")

        if i < len(sessao.perguntas_investigacao):
            with st.chat_message("assistant"):
                st.markdown(sessao.perguntas_investigacao[i])
                st.caption(
                    "Ainda ha mais de um procedimento possivel — perguntei para "
                    "estreitar."
                )

    if sessao.situacao == "investigando":
        veredito = next(
            (v for v in reversed(sessao.vereditos_abertura) if v.id == "G1T"), None
        )
        d = veredito.detalhe if veredito else {}
        resumo_evidencia = (
            f"margem {d.get('margem', 0):.0%} "
            f"(min {D.config.MARGEM_MINIMA_DOCUMENTO:.0%}) · "
            f"evidencia no 1o {d.get('share', 0):.0%} "
            f"(min {D.config.SHARE_MINIMO_DOCUMENTO:.0%}) · "
            f"rodada {sessao.rodadas}/{D.config.MAX_RODADAS_INVESTIGACAO} · "
            + " ".join(f"`{doc}` {p:.2f}" for doc, p in sessao.candidatos[:4])
        )

        if sessao.aguardando_escolha:
            with st.chat_message("assistant"):
                st.markdown(
                    "Nao consegui separar os candidatos com o que voce contou. "
                    "**A escolha e sua** — nao vou decidir isso no chute."
                )
                st.caption(resumo_evidencia)

            escolha = st.radio(
                "Qual procedimento seguir?",
                [doc for doc, _ in sessao.candidatos[:3]],
                format_func=lambda doc: f"{doc} — peso {dict(sessao.candidatos)[doc]:.2f}",
                key="escolha_manual",
            )
            if st.button("Seguir com este", type="primary"):
                st.session_state["sessao"] = D.escolher_documento(sessao, escolha)
                st.rerun()
        else:
            st.caption(resumo_evidencia)
            resposta = st.chat_input("Responda aqui — quanto mais detalhe, melhor")
            if resposta:
                with st.spinner("Refazendo a busca com todos os sintomas..."):
                    st.session_state["sessao"] = D.continuar_conversa_investigacao(
                        sessao, resposta, k=8, usar_llm=usar_llm, config_llm=cfg,
                    )
                st.rerun()

        with st.expander("Por que perguntar em vez de escolher o melhor"):
            st.markdown(
                """
`documento_predominante` e um `max`: **sempre** ha um vencedor, mesmo com 1,43
contra 1,39. Era assim que uma pergunta ampla travava num manual por acaso.

O **G1T** exige duas coisas: **margem** (ganhou do 2o) e **share** (concentra a
evidencia). A segunda pega o caso que a primeira nao ve — pesos
[1,0; 0,5; 0,5; 0,5; 0,5] tem margem folgada e ainda deixam quatro manuais de pe.

O que continua sendo codigo, nao modelo: **quem decide se ha empate** (G1T),
**sobre o que perguntar** (`SELECT` nas secoes de sintomas) e **quem decide no
fim**, se a duvida persistir — voce.
"""
            )

        st.stop()

    # A confirmacao entra como mais uma FALA do sistema, no mesmo tom das
    # outras. Antes eram tres metricas grandes, uma tabela e uma caixa amarela
    # de largura inteira — tudo isso vinha antes da conversa e a soterrava.
    # Agora o detalhe fica em expansor: quem quiser auditar abre.
    if sessao.aberta:
        with st.chat_message("assistant"):
            familias_extra = (
                " Ele cobre "
                + ", ".join(f"`{f}`" for f in sessao.familias_do_documento)
                + ", e a conversa vale para todas."
                if len(sessao.familias_do_documento) > 1 else ""
            )
            st.markdown(
                f"Encontrei o procedimento: **{sessao.manual}** "
                f"(`{sessao.familia}`).{familias_extra} "
                "Pode perguntar o que quiser sobre ele."
            )
            st.caption(
                f"🔒 travado ate o fim da conversa · melhor trecho "
                f"{max(t.score for t in sessao.trechos_de_abertura):.3f}"
                if sessao.trechos_de_abertura else "🔒 travado ate o fim da conversa"
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

if not sessao.turnos:
    st.markdown(
        f"""
Pergunte o que fazer. Cada resposta cita **documento, secao e pagina**.

Sugestoes para esta falha (`{sessao.familia}`):
- *Como corrigir? Quais passos devo seguir?*
- *Que ferramentas eu preciso?*
- *Como sei que o conserto ficou bom?*
- *E se eu nao tiver a ferramenta certa?* — para ver a recusa por falta de base
"""
    )

for turno in sessao.turnos:
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

        if turno.degradou:
            st.markdown(turno.texto_para_historico)
            st.caption(
                "⚠️ o texto acima e o do manual, sem prosa — a redacao do modelo "
                "foi reprovada no G5 nas duas tentativas"
            )
        else:
            st.markdown(turno.resposta)

        if turno.trechos:
            st.caption("**Fontes:** " + " · ".join(f"`{r}`" for r in turno.referencias))

            with st.expander("Ver o texto de cada fonte"):
                for t in turno.trechos:
                    pag = f"pagina {t.pagina}" if t.pagina else "pagina nao disponivel"
                    st.markdown(
                        f"**{t.documento_id}, secao {t.numero} — {t.titulo}**  \n"
                        f"*{pag} · tipo: {t.campo or '—'} · "
                        f"similaridade {t.score:.3f}*"
                    )
                    st.markdown(f"> {t.texto.strip()}")
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

- resposta que passou no G5 entra como esta
- resposta reprovada **nao entra**; no lugar dela vai o trecho do manual
- recusa nao entra de forma alguma

Sem isso, um deslize no turno 2 viraria contexto no turno 3, e o modelo o
trataria como fato ja estabelecido.
"""
        )
        historico = sessao.historico_para_prompt()
        if not historico:
            st.caption("Nada ainda — nenhum turno produziu conteudo verificado.")
        for i, (q, r) in enumerate(historico, 1):
            st.markdown(f"**{i}. {q}**")
            st.code(r[:800] + ("..." if len(r) > 800 else ""), language="text")

    with t2:
        if sessao.turnos[-1].prompt:
            st.caption(
                "O texto exato do ultimo turno. Nada alem disto: sem internet, "
                "sem os outros manuais, sem memoria de conversas anteriores."
            )
            st.code(sessao.turnos[-1].prompt, language="text")
        else:
            st.caption("O ultimo turno parou antes de chegar ao modelo.")
