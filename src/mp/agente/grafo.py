"""Os nos do fluxo e a ordem entre eles.

    TURNO 1 — chega o evento do sensor
      1. validar entrada             G0   codigo
      2. buscar semelhantes          G1   kNN        (Parte 3, ainda stub)
      3. e defeito?                  G2   catalogo
      4. tem manual?                 G3   catalogo
             |
         FIXA NA SESSAO: familia + manual autorizado

    TURNO 1 e seguintes — a pergunta em texto
      5. a pergunta cabe no manual?       codigo
      6. buscar trechos                   vetorial, dentro do manual fixado
      7. os trechos servem?          G4   codigo
             |
      8. REDIGIR                     <-   o modelo, unico no com LLM
             |
      9. conferir ancoragem          G5   codigo
           +- passou  -> mostra e grava no historico
           +- falhou  -> volta ao 8 (max 2 tentativas)
           +- falhou  -> mostra o trecho cru, sem prosa

**A regra que governa tudo: um unico no tem modelo de linguagem, e ele so
redige.** Nao escolhe ferramenta, nao decide se busca, nao decide se responde.
Recebe fatos prontos e devolve portugues. Todos os outros sao `if`, `SELECT` e
multiplicacao de matriz.

Isto nao e um agente autonomo. Um agente decide *"vou consultar... nao achei...
vou tentar outra coisa"* — e nesse momento quem controla e o modelo, exatamente
o que os guardrails evitam.

**Sobre o LangGraph.** As funcoes aqui sao os nos; a ordem entre elas esta escrita
em Python simples. Trocar por LangGraph e substituir este arquivo por um grafo
declarativo com as mesmas funcoes — nenhuma delas muda. O que o framework
acrescenta e o desenho explicito das arestas e o checkpoint de estado; o que ele
nao pode acrescentar e decisao, que continua sendo codigo nosso.
"""

from __future__ import annotations

import time

from mp import config
from mp.agente.estado import Sessao, Turno
from mp.guardrails import rules as g
from mp.llm import prompts
from mp.retrieval import rag
from mp.retrieval.catalog import familias_do_documento, resolver

MAX_TENTATIVAS = 2


# --------------------------------------------------------------------------
# Nos 1 a 4 — abertura da sessao
# --------------------------------------------------------------------------


def abrir_sessao(
    rotulo: str | None = None,
    evento: dict | None = None,
    diagnostico=None,
) -> Sessao:
    """Roda G0 a G3 e fixa o manual autorizado.

    Acontece **uma vez**. Depois disso o documento esta travado: nenhuma pergunta
    dos turnos seguintes pode busca-lo em outro lugar.

    **Quem decide a familia e o `diagnostico`**, quando ele vem — ou seja, o kNN
    sobre os numeros do sensor. O `rotulo` avulso existe so para inspecao manual
    e para os testes; no fluxo real, o operador nao escolhe a familia numa lista.

    Quando o JSON traz `fault` preenchido, ele e **anotacao a conferir**, nunca
    ordem: a familia usada e a que os vizinhos indicam, e a divergencia entre as
    duas vira alerta em vez de sumir.
    """
    s = Sessao(rotulo=rotulo or "")

    if evento is not None:
        v = g.g0_entrada(evento)
        s.vereditos_abertura.append(v)
        if not v:
            s.motivo = v.mensagem
            return s
        s.rotulo = rotulo or evento.get("fault") or ""

    # --- G1: os vizinhos sao proximos o bastante? --------------------------
    if diagnostico is not None:
        s.diagnostico = diagnostico
        s.rotulo = diagnostico.rotulo or s.rotulo
        v = g.g1_similaridade(diagnostico.distancia_min, diagnostico.limiar_g1)
    else:
        v = g.g1_similaridade(None)
    s.vereditos_abertura.append(v)
    if not v:
        s.motivo = v.mensagem
        return s

    # --- G2 e G3: catalogo, nao similaridade -------------------------------
    # A familia vem do kNN quando ha diagnostico; do catalogo quando a sessao
    # foi aberta por rotulo.
    if diagnostico is not None and diagnostico.familia:
        s.familia = diagnostico.familia
    else:
        s.familia = resolver(s.rotulo)["familia"]

    v = g.g2_e_problema(s.familia or s.rotulo)
    s.vereditos_abertura.append(v)
    if not v:
        s.motivo = v.mensagem
        return s

    v = g.g3_tem_documento(s.familia)
    s.vereditos_abertura.append(v)
    if not v:
        s.motivo = v.mensagem
        return s

    s.documentos = v.detalhe.get("documentos", [])
    s.aberta = True
    s.motivo = f"Manual autorizado: {s.manual}. Ele nao muda durante a conversa."
    return s


def _pergunta_de_investigacao(sessao: Sessao, cliente=None, motor=None) -> str:
    """O que perguntar ao tecnico para separar os candidatos empatados.

    **O conteudo vem de um `SELECT`**, nunca da imaginacao do modelo: as secoes
    de sintoma dos documentos que empataram. O modelo so redige — e quando nao
    ha modelo, o sistema mostra os proprios trechos e pergunta de forma generica.
    Sem LLM a pergunta fica pior; o fluxo continua inteiro.
    """
    docs = [d for d, _ in sessao.candidatos[:3]]
    trechos = rag.secoes_de_sintomas(docs, motor=motor)

    if cliente is None or not trechos:
        return (
            "Descreva mais um sintoma: onde a vibracao e mais forte, se houve "
            "ruido, calor ou folga, e o que muda quando a rotacao sobe."
        )

    mensagens = prompts.investigacao.montar(sessao.sintomas, trechos, docs)
    return cliente.gerar(mensagens).texto.strip()


def _avaliar_candidatos(sessao: Sessao, k: int, cliente=None, motor=None) -> Sessao:
    """Busca com os sintomas acumulados e decide: travar, investigar ou perguntar.

    O coracao do loop. Roda na abertura e a cada nova informacao, sempre com
    **todos** os sintomas — e por isso que a conversa converge: a evidencia
    cresce a cada volta em vez de ser substituida.
    """
    resultado = rag.buscar_por_sintomas(sessao.sintomas, k=k, motor=motor)
    sessao.trechos_de_abertura = resultado.trechos

    # --- G4: ha evidencia de qualquer especie? ------------------------------
    v = g.g4_trechos_relevantes(resultado.trechos)
    sessao.vereditos_abertura.append(v)
    if not v:
        sessao.investigando = False
        sessao.motivo = (
            "Nenhum procedimento trata do que voce descreveu. "
            + v.mensagem
            + " Descreva o sintoma com mais detalhe, ou registre um documento "
            "para essa falha."
        )
        return sessao

    sessao.candidatos = rag.ranking_documentos(resultado)

    # --- G1T: a evidencia aponta UM manual? ---------------------------------
    v = g.g1t_evidencia_decide(sessao.candidatos)
    sessao.vereditos_abertura.append(v)

    if not v:
        if sessao.rodadas < config.MAX_RODADAS_INVESTIGACAO:
            sessao.rodadas += 1
            sessao.investigando = True
            sessao.pergunta_investigacao = _pergunta_de_investigacao(
                sessao, cliente=cliente, motor=motor
            )
            sessao.perguntas_investigacao.append(sessao.pergunta_investigacao)
            sessao.motivo = v.mensagem
            return sessao

        # Teto esgotado. NAO se escolhe o topo em silencio e NAO se pergunta ao
        # modelo: mostram-se os candidatos e o tecnico decide. Pessoa escolhendo
        # nao fere os guardrails — quem eles tiram da decisao e o modelo.
        sessao.investigando = False
        sessao.aguardando_escolha = True
        sessao.pergunta_investigacao = ""
        sessao.motivo = (
            f"Depois de {sessao.rodadas} tentativas, a descricao ainda nao separa "
            f"os candidatos ({v.mensagem}). Escolha qual procedimento seguir — o "
            "sistema nao vai decidir isso no chute."
        )
        return sessao

    return fixar_documento(sessao, sessao.candidatos[0][0])


def fixar_documento(sessao: Sessao, documento: str) -> Sessao:
    """Trava o manual e abre a sessao. Ponto unico de travamento.

    Chamado pelo `_avaliar_candidatos` quando o G1T aprova, e pela interface
    quando o **tecnico** escolhe depois do teto de rodadas. Os dois caminhos
    passam pelo mesmo G2 e produzem a mesma sessao travada — a origem da decisao
    muda, a garantia nao.

    **Travar e irreversivel dentro da sessao.** Chamar de novo numa sessao ja
    aberta nao troca o manual: seria a unica porta capaz de furar a regra 1 do
    `estado.py` — o manual e fixado no turno 1 e nao muda. A investigacao existe
    justamente para nao precisar destravar depois; quem quiser outro manual
    encerra a sessao e abre outra.
    """
    if sessao.aberta:
        return sessao

    familias = familias_do_documento(documento)

    sessao.documentos = [documento]
    sessao.familia = familias[0] if familias else None
    sessao.familias_do_documento = familias
    sessao.peso_documento = dict(sessao.candidatos).get(documento, 0.0)
    sessao.investigando = False
    sessao.aguardando_escolha = False
    sessao.pergunta_investigacao = ""

    if sessao.familia is None:
        sessao.aberta = False
        sessao.motivo = (
            f"O documento {documento} nao esta ligado a nenhuma familia no "
            "fault_map.yaml — corrija o catalogo."
        )
        return sessao

    # G2 continua valendo: documento de estado nao gera prescricao.
    v = g.g2_e_problema(sessao.familia)
    sessao.vereditos_abertura.append(v)
    if not v:
        sessao.aberta = False
        sessao.motivo = v.mensagem
        return sessao

    sessao.rotulo = sessao.familia
    sessao.aberta = True
    sessao.motivo = (
        f"Os sintomas apontam {documento} ({sessao.familia}). "
        "Ele fica fixado ate o fim da conversa."
    )
    return sessao


def continuar_investigacao(
    sessao: Sessao, sintoma: str, cliente=None, k: int = 8, motor=None
) -> Sessao:
    """Mais um sintoma entra e a busca recomeca — com todos, nao so com o novo.

    Este e o passo que faz o loop valer a pena. O problema original era falta de
    texto: duas palavras dao um vetor fraco e o vencedor sai quase por sorteio.
    Cada volta acrescenta evidencia; a margem tende a abrir.

    Nao ha garantia de que abra. Desalinhamento e desbalanceamento descrevem
    sintomas parecidos de verdade, e nenhuma pergunta separa os dois — por isso
    existe o teto de rodadas e a escolha humana no fim.
    """
    if sessao.aberta or not sintoma.strip():
        return sessao

    sessao.sintomas.append(sintoma.strip())
    return _avaliar_candidatos(sessao, k=k, cliente=cliente, motor=motor)


def abrir_sessao_por_texto(
    descricao: str, k: int = 8, motor=None, cliente=None
) -> Sessao:
    """Abre a conversa a partir da **descricao escrita** do problema.

    O caminho mais comum na pratica: o tecnico chega dizendo "o motor esta
    vibrando muito e esquentando o mancal". Nao ha JSON, nao ha rotulo, e a
    pergunta a responder e justamente *qual manual serve aqui*.

    Sem familia, o filtro exato do estagio 1 nao tem como rodar. Entao a ordem
    se inverte **uma unica vez**: busca em todos os manuais, escolhe o documento
    que os trechos apontam, e **fixa esse documento**. Da segunda pergunta em
    diante a conversa volta ao caminho normal, filtrada dentro dele.

    **O que se perde nessa inversao.** Com o filtro por familia, uma falha sem
    manual nao tinha onde ser buscada — a recusa era estrutural. Aqui a busca
    sempre devolve alguma coisa, e a unica trava e o G4: se nem o melhor trecho
    passa do score minimo, a sessao nao abre. E uma garantia mais fraca, e a
    interface mostra o score para que a diferenca fique visivel.

    **A sessao pode nao abrir de primeira, e isso e deliberado.** Antes, o
    documento saia de um `max` sobre os pesos — que sempre devolve um vencedor,
    mesmo com 1,43 contra 1,39. Agora o G1T exige margem; sem ela a sessao entra
    em **investigacao** e pede mais sintomas, em vez de travar no acaso. Como o
    manual nao muda depois de travado, travar errado contamina a conversa
    inteira: e mais barato perguntar.
    """
    s = Sessao(rotulo="")
    s.descricao = descricao
    s.sintomas = [descricao.strip()] if descricao.strip() else []

    if not s.sintomas:
        s.motivo = "Descreva o problema para o sistema procurar o procedimento."
        return s

    return _avaliar_candidatos(s, k=k, cliente=cliente, motor=motor)


# --------------------------------------------------------------------------
# No 5 — trava de escopo
# --------------------------------------------------------------------------


def no_escopo(pergunta: str, sessao: Sessao, motor=None) -> tuple[bool, float, str]:
    """A pergunta trata do assunto do manual fixado?

    Mede a melhor semelhanca da pergunta contra o manual **inteiro** — todas as
    secoes, nao so as de correcao. Se nem a melhor passa de `LIMIAR_ESCOPO`, a
    pergunta e de outro assunto e o modelo nao e chamado.

    E diferente do G4, que vem depois: aqui perguntamos *"isto tem a ver com este
    manual?"*; la, *"estes trechos sustentam uma resposta?"*. Passar aqui e
    falhar la e o caso comum e util — a pergunta e do assunto, mas o manual nao
    a responde.
    """
    resultado = rag.buscar(pergunta, sessao.familia, k=1, motor=motor)
    if resultado.vazio:
        return False, 0.0, resultado.motivo or "Nada a comparar no manual."

    melhor = resultado.trechos[0].score
    if melhor < config.LIMIAR_ESCOPO:
        return False, melhor, (
            f"O procedimento nao cobre isso. A pergunta nao tem relacao com "
            f"{sessao.manual}, que e o manual desta conversa "
            f"(semelhanca {melhor:.3f}, minimo {config.LIMIAR_ESCOPO})."
        )
    return True, melhor, ""


# --------------------------------------------------------------------------
# Nos 6 a 9 — a volta da conversa
# --------------------------------------------------------------------------


def responder(
    sessao: Sessao,
    pergunta: str,
    cliente=None,
    k: int = 5,
    so_prescritivos: bool = True,
    motor=None,
) -> Turno:
    """Uma volta completa: escopo, busca, G4, redacao, G5.

    Acrescenta o turno a sessao e o devolve. `cliente=None` para no no 7 e
    entrega o trecho cru — util para provar que o conteudo nao depende do modelo.
    """
    t0 = time.time()
    turno = Turno(pergunta=pergunta)

    if not sessao.aberta:
        turno.recusado = True
        turno.motivo = sessao.motivo
        turno.resposta = sessao.motivo
        sessao.turnos.append(turno)
        return turno

    # --- no 5: trava de escopo ---------------------------------------------
    dentro, score, motivo = no_escopo(pergunta, sessao, motor=motor)
    if not dentro:
        turno.recusado = True
        turno.motivo = motivo
        turno.resposta = motivo
        turno.segundos = round(time.time() - t0, 2)
        sessao.turnos.append(turno)
        return turno

    # --- no 6: busca dentro do manual fixado -------------------------------
    busca = (rag.buscar_prescritivo if so_prescritivos else rag.buscar)(
        pergunta, sessao.familia, k=k, motor=motor
    )
    turno.trechos = busca.trechos

    # --- no 7: G4 -----------------------------------------------------------
    v = g.g4_trechos_relevantes(turno.trechos)
    turno.vereditos.append(v)
    if not v:
        turno.recusado = True
        turno.motivo = v.mensagem
        turno.resposta = (
            "O manual existe, mas nao responde a esta pergunta. " + v.mensagem
        )
        turno.segundos = round(time.time() - t0, 2)
        sessao.turnos.append(turno)
        return turno

    historico = sessao.historico_para_prompt(config.MAX_TURNOS_NO_PROMPT)
    mensagens = prompts.montar(pergunta, sessao.familia, turno.trechos,
                               historico=historico)
    turno.prompt = prompts.texto_enviado(mensagens)

    if cliente is None:
        turno.resposta = "\n\n".join(
            f"**{t.referencia} — {t.titulo}**\n\n{t.texto.strip()}"
            for t in turno.trechos
        )
        turno.segundos = round(time.time() - t0, 2)
        sessao.turnos.append(turno)
        return turno

    # --- no 8 e 9: redigir e conferir, com ate duas tentativas -------------
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        turno.tentativas = tentativa
        r = cliente.gerar(mensagens)

        turno.resposta = r.texto.strip()
        turno.usou_llm = True
        turno.provedor, turno.modelo = r.provedor, r.modelo
        turno.tokens_entrada += r.tokens_entrada
        turno.tokens_saida += r.tokens_saida

        v = g.g5_citacoes_existem(turno.resposta, turno.trechos)
        if v.passou or tentativa == MAX_TENTATIVAS:
            turno.vereditos.append(v)
            turno.verificada = v.passou
            turno.degradou = not v.passou
            break

        # Reprovou e ainda ha tentativa: reforca a regra violada e repete.
        # A tentativa ruim NAO entra no historico — senao o proprio erro vira
        # contexto e o modelo passa a trata-lo como aceito.
        disponiveis = "; ".join(t.citacao for t in turno.trechos)
        mensagens = prompts.montar(
            f"{pergunta}\n\n[Aviso do sistema: a resposta anterior foi rejeitada — "
            f"{v.mensagem} As unicas fontes citaveis sao: {disponiveis}.]",
            sessao.familia, turno.trechos, historico=historico,
        )

    turno.segundos = round(time.time() - t0, 2)
    sessao.turnos.append(turno)
    return turno
