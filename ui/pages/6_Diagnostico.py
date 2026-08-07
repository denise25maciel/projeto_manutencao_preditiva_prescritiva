"""O fluxo inteiro numa tela so. Duas portas de entrada, o mesmo destino.

**Por texto (o caminho comum).** O tecnico descreve o problema. O sistema busca
nos manuais, escolhe o que trata daquilo e fixa. Nao ha lista de falhas para
escolher — descobrir qual e a falha e o que o sistema faz.

**Por evento de sensor (opcional).** Quando ha um JSON de leitura, o kNN o
compara com o historico e indica a familia pelos numeros. O campo `fault`, se
vier, e anotacao a conferir: a familia usada e a que os vizinhos indicam, e a
divergencia vira alerta.

Nos dois casos, uma vez aberto, o manual fica travado e a conversa segue igual.

A tela tem **duas abas**: a conversa e tudo sobre o modelo de linguagem —
provedores disponiveis, configuracao, regras do prompt, teste de conexao. A
segunda mora em `_secao_modelo.py` e era uma tela propria, apagada: quem esta no
meio de uma conversa nao pode precisar sair dela para trocar de provedor.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import _dados as D
import _secao_modelo


D.configurar_pagina("Diagnostico", "🩺")
# Teto de tamanho para os titulos que vierem dentro dos baloes. O saneamento do
# texto continua sendo `para_exibir` / `em_uma_linha`; isto e a rede embaixo.
D.estilo_do_chat()

st.title("🩺 Diagnostico e Conversa")
st.caption("Descreva o problema. O sistema acha o procedimento e conversa sobre ele.")

# ==========================================================================
# As duas abas
# ==========================================================================
#
# A configuracao do modelo tem de ser alcancavel DE DENTRO da conversa: trocar
# de provedor ou mexer nas regras do prompt nao pode custar sair da tela e
# perder o fio. Por isso a tela Modelo de Linguagem foi apagada e virou esta
# aba — o conteudo inteiro dela esta em `_secao_modelo.py`.
#
# A aba do modelo e desenhada PRIMEIRO, embora apareca em segundo: o fluxo do
# diagnostico chama `st.stop()` em varios pontos, e `st.stop()` interrompe o
# script inteiro, nao so a aba. Desenhada depois, ela ficaria vazia sempre que a
# conversa parasse antes do fim — que e quase sempre.
aba_diagnostico, aba_modelo = st.tabs(
    ["🩺 Diagnostico e conversa", "🤖 Modelo de linguagem"]
)

with aba_modelo:
    st.markdown(
        """
        O que estiver escolhido aqui e o que redige a resposta da aba ao lado, da
        proxima pergunta em diante.

        Com o modelo desligado na barra lateral, nada disto e usado: a resposta
        passa a ser o texto cru do manual, sem prosa.
        """
    )
    cfg = _secao_modelo.render()


# ==========================================================================
# Barra lateral
# ==========================================================================
with st.sidebar:
    st.header("Modelo")
    # `cfg` vem da aba ao lado, desenhada logo acima — nao de uma leitura solta
    # da sessao. Assim a barra lateral mostra o que esta escolhido AGORA, e nao o
    # que sobrou da rodada anterior.
    if cfg:
        st.success(f"`{cfg['provedor']}`\n\n`{cfg['modelo']}`")
    else:
        st.warning(
            "Nenhum provedor disponivel — veja a aba **Modelo de linguagem**. "
            "A conversa segue funcionando com o texto cru do manual."
        )

    usar_llm = st.toggle(
        "Deixar o modelo redigir", value=bool(cfg), disabled=not cfg,
        help="Desligado, a resposta e o texto cru do manual — sem modelo nenhum.",
    )

    st.divider()
    st.header("Busca")
    k_trechos = st.slider("Trechos por resposta", 1, 10, 5)
    #analisar
    if st.session_state.get("sessao"):
        st.divider()
        if st.button("Encerrar sessao", width="stretch"):
            for chave in ("sessao", "diagnostico", "evento"):
                st.session_state.pop(chave, None)
            st.rerun()

with aba_diagnostico:
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

            # ------------------------------------------------------------------
            # A leitura do sensor, no mesmo formulario
            # ------------------------------------------------------------------
            #
            # Existe uma aba so para o sensor, ao lado, para quem so tem o JSON.
            # Aqui ele e **complemento da descricao**: o tecnico conta o que ve e,
            # se tiver o numero, cola junto — que e como a coisa acontece na
            # pratica, e nao em duas telas separadas.
            #
            # Quando vem preenchido, quem decide a familia sao os numeros. Nao e
            # preferencia: a floresta mede o comportamento do trecho, enquanto a
            # descricao passa por uma busca semantica que sempre volta com
            # alguma coisa.
            with st.expander("📈 Tenho as leituras do sensor (opcional)"):
                st.caption(
                    "Cole um **trecho** de leituras em CSV. Com ele preenchido, "
                    "**quem decide a familia sao os numeros**, e o resultado entra "
                    "na conversa como a primeira fala. Vazio, o sistema decide "
                    "pela descricao."
                )

                c_semente, c_botao = st.columns([1, 2])
                with c_semente:
                    semente_txt = st.number_input(
                        "Sorteio", 0, 9999, 0, step=1, key="txt_semente",
                        help="Muda qual trecho real do historico e sorteado.",
                    )
                with c_botao:
                    st.markdown("&nbsp;")
                    if st.button("Preencher com um trecho real do historico",
                                 key="btn_preencher", width="stretch"):
                        exemplo_txt = D.r_trecho_de_exemplo(50, int(semente_txt))
                        st.session_state["sensor_json"] = exemplo_txt[
                            D.r_clf_colunas(False) + [D.config.COLUNA_ROTULO]
                        ].to_csv(index=False)
                        st.rerun()

                st.text_area(
                    "Leituras do sensor (CSV)",
                    key="sensor_json",
                    height=220,
                    placeholder="z_rms_velocity_mm_s,x_rms_velocity_mm_s,...\n"
                                "1.09,1.57,...\n1.10,1.56,...",
                    help="Varias linhas: uma leitura isolada zera quatro das cinco "
                         "estatisticas. A coluna `fault`, se vier, e a anotacao do "
                         "operador — sera confrontada, nunca obedecida.",
                )

            if st.button("Procurar o procedimento", type="primary", key="btn_texto"):
                sensor_txt = (st.session_state.get("sensor_json") or "").strip()

                if not descricao.strip() and not sensor_txt:
                    st.warning(
                        "Escreva a descricao do problema, ou cole a leitura do "
                        "sensor no campo acima."
                    )
                    st.stop()

                bloco = None
                if sensor_txt:
                    try:
                        bloco = D.ler_bloco_de_sensor(sensor_txt)
                    except Exception as e:  # noqa: BLE001 — a mensagem vai para a tela
                        st.error(f"Nao consegui ler as leituras: {e}")
                        st.stop()

                if bloco is not None:
                    # --- caminho do sensor -------------------------------------
                    with st.spinner("Classificando o trecho..."):
                        clf = D.classificar_trecho(bloco)
                        nova = D.abrir_conversa(classificacao=clf)

                    # O anuncio entra como FALA, e nao como painel de metricas: o
                    # tecnico esta lendo uma conversa. Os numeros sao os da
                    # floresta — o modelo, quando ligado, so os redige, e o G5N
                    # confere se cada numero escrito foi um numero apurado.
                    with st.spinner("Escrevendo o que os numeros indicam..."):
                        D.classificar_na_conversa(
                            nova, clf, usar_llm=usar_llm, config_llm=cfg
                        )

                    st.session_state["diagnostico"] = clf
                else:
                    # --- caminho por texto -------------------------------------
                    with st.spinner("Procurando nos seis manuais..."):
                        nova = D.abrir_conversa_por_texto(
                            descricao, k=8, usar_llm=usar_llm, config_llm=cfg
                        )
                    st.session_state.pop("diagnostico", None)

                st.session_state["sessao"] = nova
                st.rerun()

        # --------------------------------------------------------------- sensor
        with aba_sensor:
            st.markdown(
                """
    Um **trecho** de leituras do sensor, em CSV. A floresta o resume e indica a
    falha — sem passar pelo texto.

    Precisa de varias linhas: uma leitura isolada zera quatro das cinco
    estatisticas, e zero variacao e a assinatura de `motor_desligado`.

    A coluna `fault`, se vier, e a anotacao do operador — sera **confrontada**
    com o que o modelo indicar, nunca obedecida.
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

            exemplo = D.r_trecho_de_exemplo(50, int(semente))
            colunas_exemplo = D.r_clf_colunas(False) + (
                [D.config.COLUNA_ROTULO] if manter_fault else []
            )

            texto_csv = st.text_area(
                "Leituras do sensor (CSV)",
                value=exemplo[colunas_exemplo].to_csv(index=False),
                height=300,
            )

            if st.button("Diagnosticar pelo sensor", key="btn_sensor"):
                try:
                    bloco = D.ler_bloco_de_sensor(texto_csv)
                except Exception as e:  # noqa: BLE001 — a mensagem vai para a tela
                    st.error(f"Nao consegui ler as leituras: {e}")
                    st.stop()

                with st.spinner("Classificando o trecho..."):
                    clf = D.classificar_trecho(bloco)
                    nova = D.abrir_conversa(classificacao=clf)

                with st.spinner("Escrevendo o que os numeros indicam..."):
                    D.classificar_na_conversa(nova, clf, usar_llm=usar_llm, config_llm=cfg)

                st.session_state["diagnostico"] = clf
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
                        # `em_uma_linha` e nao `para_exibir`: numa previa de uma
                        # linha o `##` do inicio do trecho nao e hierarquia, e
                        # o Markdown o leria como cabecalho dentro da legenda.
                        st.caption(D.em_uma_linha(trecho.texto))

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

    # ==========================================================================
    # O que os numeros disseram — antes do veredito da abertura
    # ==========================================================================
    #
    # Desenhado aqui, e nao no laco da conversa la embaixo, porque este turno
    # **vale mesmo quando o fluxo para**. Saber que os numeros apontam `normal`,
    # ou que apontam uma familia sem manual, e a resposta — nao a falta dela. Se
    # ficasse depois do `st.stop()`, sumiria justamente nos casos em que e a
    # unica coisa que o sistema tem a dizer.
    for _turno in sessao.turnos:
        if not _turno.classificacao:
            continue
        with st.chat_message("assistant"):
            st.markdown(D.para_exibir(_turno.resposta))
            _selo = "🔢 numeros do kNN"
            if _turno.usou_llm:
                _v = next((v for v in _turno.vereditos if v.id == "G5N"), None)
                _selo += (
                    f" · redigido pelo modelo · {'✓' if _v and _v.passou else '✗'} G5N"
                    if _v else " · redigido pelo modelo"
                )
                if _turno.degradou:
                    _selo += " · texto do modelo descartado, numeros preservados"
            else:
                _selo += " · sem modelo, texto do proprio sistema"
            st.caption(_selo)

            with st.expander("Contra o que esta resposta foi conferida"):
                st.caption(
                    "O **G5N** so aceita, na prosa, numero que esteja nesta "
                    "lista. Ela foi apurada pelo kNN antes de o modelo ser "
                    "chamado — ele redige, nunca produz."
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"fato apurado": k, "valor": str(v)}
                         for k, v in _turno.fatos.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                if _turno.prompt:
                    st.caption("O prompt exato que foi enviado:")
                    st.code(_turno.prompt, language="text")

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
    # O turno de classificacao NAO conta como "a conversa ja comecou": ele
    # anuncia o que os numeros disseram, e o resumo do manual e a continuacao
    # natural dele. Sem esta distincao, colar a leitura do sensor faria o
    # sistema pular o procedimento inteiro.
    if not any(not t.classificacao for t in sessao.turnos):
        with st.spinner("Lendo o procedimento..."):
            D.resumo_de_abertura(sessao, usar_llm=usar_llm, config_llm=cfg,
                                 k=k_trechos + 1)
        st.rerun()

    for turno in sessao.turnos:
        # Ja desenhado la em cima, antes do veredito da abertura.
        if turno.classificacao:
            continue

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

    # Dentro de uma aba, o `chat_input` aparece **no fim da conversa**, e nao
    # grudado na base da janela — o Streamlit so fixa quando ele esta solto no
    # corpo principal. E o preco de ter as duas abas, e o preco certo: uma barra
    # fixa na base apareceria tambem na aba de configuracao, onde nao ha conversa
    # nenhuma para responder.
    pergunta = st.chat_input("Pergunte sobre o procedimento...")

    if pergunta:
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            with st.spinner("Buscando no manual..."):
                D.responder_turno(sessao, pergunta, usar_llm=usar_llm,
                                  config_llm=cfg, k=k_trechos)
        st.rerun()

