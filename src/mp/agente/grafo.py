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
from mp.retrieval.catalog import familias_do_documento, verificar_existencia_conserto

MAX_TENTATIVAS = 2


# --------------------------------------------------------------------------
# Nos 1 a 4 — abertura da sessao
# --------------------------------------------------------------------------


def abrir_sessao(
    rotulo: str | None = None,
    evento: dict | None = None,
    diagnostico=None,
) -> Sessao:
    """Roda G0 a G3 e fixa o manual. A entrada pelo **sensor**; a outra e
    `abrir_sessao_por_texto`.

    Quem decide a familia e o `diagnostico`, isto e, o kNN sobre os numeros. O
    `rotulo` avulso serve so a inspecao manual e aos testes.

    `fault` no JSON e anotacao a conferir, nunca ordem: vale a familia dos
    vizinhos, e a divergencia vira alerta.
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
    #Analise
    if diagnostico is not None and diagnostico.familia:
        s.familia = diagnostico.familia
    else:
        s.familia = verificar_existencia_conserto(s.rotulo).familia

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


def _nome_de_familias(familias: list[str], documento: str) -> str:
    """`["rolamento_inner", ...]` -> `rolamento ×4`. Vem do catalogo, nao do modelo.

    Um documento pode cobrir varias familias — o Doc1 atende quatro tipos de
    rolamento, e listar as quatro deixa a frase ilegivel. Quando ha prefixo
    comum ele ja nomeia o conjunto; senao, mostra a primeira e conta o resto.

    **O nome do conjunto, nunca um representante dele.** `rolamento ×4` diz o
    que se sabe; `rolamento_inner` diria um tipo que ninguem apurou.
    """
    if not familias:
        return documento
    if len(familias) == 1:
        return familias[0]

    # Sem parenteses: o nome entra dentro de uma lista que ja esta entre
    # parenteses, e aninhar dois niveis fica ilegivel.
    prefixo = familias[0].split("_")[0]
    if all(f.startswith(prefixo) for f in familias):
        return f"{prefixo} ×{len(familias)}"
    return f"{familias[0]} +{len(familias) - 1}"


def _nome_legivel(documento: str) -> str:
    """O nome do documento para a tela, resolvido pelo catalogo."""
    return _nome_de_familias(familias_do_documento(documento), documento)


def _avaliar_candidatos(sessao: Sessao, k: int, motor=None) -> Sessao:
    """Busca com os sintomas acumulados e decide: travar o manual ou pedir a escolha.

    Roda na abertura e a cada sintoma novo, sempre com todos os sintomas juntos.

    **A busca aqui e invertida:**

    - aqui: nao sei a familia -> busco em **todos** os manuais -> vence o
      documento mais apontado pelos trechos. O documento e o *resultado*.
    - normal: sei a familia -> filtro o manual dela -> busco dentro. O documento
      e a *entrada*.

    Custa uma garantia: com o filtro, falha sem manual nao tinha onde ser
    buscada e a recusa era estrutural. Buscando em tudo sempre volta algo, e so
    o **G4** barra — por isso a tela mostra o score.

    G1T com margem -> `fixar_documento`. Sem margem -> a lista vai para o
    tecnico, porque manual travado nao muda mais.

    **Nenhum modelo participa:** busca vetorial, `SELECT` e `if`.
    """
    resultado = rag.buscar_por_sintomas(sessao.sintomas, k=k, motor=motor)
    sessao.trechos_de_abertura = resultado.trechos

    # --- G4: ha evidencia de qualquer especie? ------------------------------
    v = g.g4_trechos_relevantes(resultado.trechos)
    sessao.vereditos_abertura.append(v)
    if not v:
        sessao.aguardando_escolha = False
        sessao.motivo = (
            "Nenhum procedimento trata do que voce descreveu. "
            + v.mensagem
            + " Descreva o sintoma com mais detalhe, ou registre um documento "
            "para essa falha."
        )
        return sessao

    sessao.candidatos = rag.ranking_documentos(resultado)
    sessao.nomes_candidatos = {
        doc: _nome_legivel(doc) for doc, _ in sessao.candidatos
    }

    # --- G1T: a evidencia aponta UM manual? ---------------------------------
    v = g.g1t_evidencia_decide(sessao.candidatos)
    sessao.vereditos_abertura.append(v)

    if not v:
        # NAO se escolhe o topo em silencio e NAO se pergunta ao modelo qual e:
        # mostram-se os candidatos, com o trecho que fez cada um aparecer, e o
        # tecnico decide. Pessoa escolhendo nao fere os guardrails — quem eles
        # tiram da decisao e o modelo.
        sessao.aguardando_escolha = True
        sessao.motivo = (
            f"A descricao nao separa os candidatos ({v.mensagem}). Escolha o "
            "procedimento que descreve a sua maquina, ou detalhe mais — o "
            "sistema nao vai decidir isso no chute."
        )
        return sessao

    return fixar_documento(sessao, sessao.candidatos[0][0])


def fixar_documento(
    sessao: Sessao, documento: str, por_escolha: bool = False
) -> Sessao:
    """Trava o manual e abre a sessao. **Ponto unico de travamento.**
    #analise - verificar possibilidade de remover G2 após a limpeza dos dados
    Duas origens: o G1T aprovando, ou o tecnico escolhendo na lista. Ambas
    passam pelo mesmo G2; `por_escolha` so muda o texto do `motivo`.

    **Irreversivel:** numa sessao ja aberta nao troca o manual — seria a unica
    porta capaz de furar a regra 1 do `estado.py`.
    """
    if sessao.aberta:
        return sessao

    familias = familias_do_documento(documento)

    sessao.documentos = [documento]
    sessao.familias_do_documento = familias
    # **O tipo so tem nome quando o documento o determina.** Um manual de uma
    # familia so a identifica; um que cobre varias, nao — o Doc1 atende quatro
    # tipos de rolamento, e a busca por texto travou o procedimento, nao o tipo.
    # Gravar `familias[0]` ali seria dar nome de defeito apurado a primeira
    # linha de uma lista, e o nome sairia trocado ao reordenar o YAML.
    sessao.familia = familias[0] if len(familias) == 1 else None
    sessao.nomes_candidatos.setdefault(
        documento, _nome_de_familias(familias, documento)
    )
    sessao.peso_documento = dict(sessao.candidatos).get(documento, 0.0)
    sessao.aguardando_escolha = False

    if not familias:
        sessao.aberta = False
        sessao.motivo = (
            f"O documento {documento} nao esta ligado a nenhuma familia no "
            "fault_map.yaml — corrija o catalogo."
        )
        return sessao
    #analise
    # G2 continua valendo: documento de estado nao gera prescricao. Julga
    # `familias[0]`, como sempre julgou — a fragilidade de decidir por uma so
    # quando o documento cobre varias esta registrada e sera tratada a parte.
    v = g.g2_e_problema(familias[0])
    sessao.vereditos_abertura.append(v)
    if not v:
        sessao.aberta = False
        sessao.motivo = v.mensagem
        return sessao

    # So recebe nome quando ha nome. Manual de varias familias deixa o rotulo
    # vazio: escrever um dos quatro tipos aqui registraria como falha observada
    # algo que ninguem observou — nem o operador anotou, nem o kNN apurou.
    sessao.rotulo = sessao.familia or ""
    sessao.aberta = True
    # Quem travou muda o que se pode afirmar. "Os sintomas apontam" e verdade
    # quando o G1T decidiu; quando quem decidiu foi o tecnico, escrever isso
    # seria atribuir a evidencia uma conclusao que ela justamente nao deu.
    sessao.motivo = (
        f"Voce escolheu {documento} ({sessao.assunto}). "
        "Ele fica fixado ate o fim da conversa."
        if por_escolha else
        f"Os sintomas apontam {documento} ({sessao.assunto}). "
        "Ele fica fixado ate o fim da conversa."
    )
    return sessao


def separar_sintomas(descricao: str, cliente=None) -> tuple[list[str], str]:
    """No 0 — a fala do tecnico vira uma lista de sintomas. Devolve `(sintomas, nota)`.

    **O unico ponto do projeto em que o modelo devolve estrutura, e nao prosa.**
    Por baixo e *tool calling*: `prompts.sintomas.Sintomas` e o formulario, e
    `cliente.estruturar` faz o provedor preenche-lo.

    Existe porque a busca precisa dos sintomas **separados**: ela codifica um a
    um e fica com o maior score por trecho, enquanto uma frase unica vira um
    vetor perto da media — e a media nao e nenhum dos sintomas. Ver
    `rag.buscar_por_sintomas`.

    **Degrada para o comportamento anterior.** Sem cliente, ou dando erro, a
    descricao inteira volta como sintoma unico, que e como era antes desta
    etapa. Extracao e melhoria da busca, nao requisito dela — e o modelo nao
    pode ser capaz de derrubar a conversa.
    """
    descricao = descricao.strip()
    if not descricao:
        return [], ""
    if cliente is None:
        return [descricao], ""

    try:
        resultado = cliente.estruturar(
            prompts.sintomas.montar(descricao), prompts.sintomas.Sintomas
        )
    except Exception as e:  # noqa: BLE001 — a nota vai para a tela
        return [descricao], (
            f"Nao deu para separar os sintomas ({type(e).__name__}); a busca usa "
            "a descricao inteira."
        )

    return prompts.sintomas.conferir(descricao, list(resultado.sintomas))


def acrescentar_sintoma(
    sessao: Sessao, sintoma: str, k: int = 8, motor=None, cliente=None
) -> Sessao:
    """Mais um sintoma entra e a busca recomeca — com todos, nao so com o novo.

    A indecisao vem de falta de texto: duas palavras dao um vetor fraco e o
    vencedor sai quase por sorteio. Mais sintoma, mais margem.

    **Sem teto de tentativas**, porque desalinhamento e desbalanceamento tem
    sintomas parecidos de verdade e podem nunca se separar. Por isso a lista
    fica sempre na tela, e nao no fim de um contador.

    O que o tecnico acrescenta passa pela mesma separacao da abertura: uma
    frase com duas observacoes entra como duas, aqui tambem.
    """
    if sessao.aberta or not sintoma.strip():
        return sessao

    novos, nota = separar_sintomas(sintoma, cliente)
    sessao.sintomas.extend(novos)
    sessao.nota_sintomas = nota
    return _avaliar_candidatos(sessao, k=k, motor=motor)


def abrir_sessao_por_texto(
    descricao: str, k: int = 8, motor=None, cliente=None
) -> Sessao:
    """Abre a conversa pela **descricao escrita** do problema.

    "O motor esta vibrando e esquentando o mancal": sem JSON nem rotulo, a
    pergunta e qual manual serve. Aqui so se monta a `Sessao`; o trabalho e do
    `_avaliar_candidatos`.

    Com `cliente`, a fala e separada em sintomas antes da busca — ver
    `separar_sintomas`. Sem ele, entra inteira, e a busca segue funcionando.

    Pode voltar **sem abrir**, com a lista de candidatos. E desfecho normal.
    """
    s = Sessao(rotulo="")
    s.descricao = descricao

    if not descricao.strip():
        s.motivo = "Descreva o problema para o sistema procurar o procedimento."
        return s

    s.sintomas, s.nota_sintomas = separar_sintomas(descricao, cliente)

    return _avaliar_candidatos(s, k=k, motor=motor)


# --------------------------------------------------------------------------
# No 5 — trava de escopo
# --------------------------------------------------------------------------


def no_escopo(pergunta: str, sessao: Sessao, motor=None) -> tuple[bool, float, str]:
    """A pergunta trata do assunto do manual fixado?

    Compara contra o manual **inteiro**, nao so as secoes de correcao. Abaixo de
    `LIMIAR_ESCOPO`, o modelo nao e chamado.

    Nao e o G4: aqui, "isto tem a ver com este manual?"; la, "estes trechos
    sustentam uma resposta?".

    Busca por `documentos`, nao por familia: o manual travado e o proprio
    escopo, e e ele que a pergunta tem de caber.
    """
    resultado = rag.buscar(pergunta, sessao.familia, k=1, motor=motor,
                           documentos=sessao.documentos)
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
    trechos=None,
    sistema: str | None = None,
) -> Turno:
    """Uma volta completa: escopo, busca, G4, redacao, G5.

    `cliente=None` para no no 7 e entrega o trecho cru — prova que o conteudo
    nao vem do modelo.

    `trechos` pronto pula os nos 5 e 6. Serve ao resumo de abertura, cuja
    pergunta e do proprio sistema e ja nasce dentro do manual travado. Do G4 em
    diante o caminho e o mesmo.

    `sistema` troca as regras do prompt, e a tela deixa edita-las. Os nos 5, 7 e
    9 nao leem esse texto: escopo, G4 e G5 continuam valendo qualquer coisa que
    seja escrita ali.
    """
    t0 = time.time()
    turno = Turno(pergunta=pergunta)

    if not sessao.aberta:
        turno.recusado = True
        turno.motivo = sessao.motivo
        turno.resposta = sessao.motivo
        sessao.turnos.append(turno)
        return turno

    if trechos is not None:
        turno.trechos = list(trechos)
    else:
        # --- no 5: trava de escopo -----------------------------------------
        dentro, score, motivo = no_escopo(pergunta, sessao, motor=motor)
        if not dentro:
            turno.recusado = True
            turno.motivo = motivo
            turno.resposta = motivo
            turno.segundos = round(time.time() - t0, 2)
            sessao.turnos.append(turno)
            return turno

        # --- no 6: busca dentro do manual fixado ---------------------------
        # `documentos` e o que foi travado. Filtrar pela familia daria a volta
        # familia -> documento e poderia trazer um manual a mais.
        busca = (rag.buscar_prescritivo if so_prescritivos else rag.buscar)(
            pergunta, sessao.familia, k=k, motor=motor,
            documentos=sessao.documentos,
        )
        turno.trechos = busca.trechos

    # --- no 7: G4 -----------------------------------------------------------
    #
    # O G4 mede pertinencia pelo SCORE de similaridade. Quando os trechos vem de
    # `SELECT`, nao ha score — `secoes_por_campo` devolve 0.0 porque nenhuma
    # semelhanca foi calculada, e seria desonesto inventar um. Aplicar o limiar
    # ali reprovaria **sempre**, e reprovaria justamente o conteudo de
    # pertinencia mais garantida do fluxo: as secoes de sintoma e correcao do
    # manual que o G3 ja autorizou.
    #
    # Entao o no 7 continua existindo e continua registrando veredito — o que
    # muda e a pergunta que ele faz. Por semelhanca: "o score passa do minimo?".
    # Por selecao: "as secoes existem?". A trava nao some, muda de natureza.
    if trechos is not None:
        v = g.Veredito(
            "G4", bool(turno.trechos),
            f"{len(turno.trechos)} secao(oes) escolhidas por tipo dentro de "
            f"{sessao.manual}. Pertinencia por construcao, nao por score."
            if turno.trechos else
            f"{sessao.manual} nao tem secoes dos tipos necessarios.",
            {"por_selecao": True, "total": len(turno.trechos)},
        )
    else:
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
                               historico=historico, familias=sessao.familias,
                               sistema=sistema)
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
            familias=sessao.familias, sistema=sistema,
        )

    turno.segundos = round(time.time() - t0, 2)
    sessao.turnos.append(turno)
    return turno


# --------------------------------------------------------------------------
# Abertura: o resumo que o sistema da sozinho ao fixar o manual
# --------------------------------------------------------------------------

# A pergunta que o sistema faz a si mesmo assim que trava o documento. Fixa e
# versionada aqui, nao montada na tela: e ela que define o que toda conversa
# comeca respondendo.
PERGUNTA_DE_ABERTURA = (
    "Qual e o problema, quais sao os sintomas dele e como corrigir?"
)

# As secoes que a abertura precisa ter, na ordem em que a resposta as usa. Uma
# pergunta so — "o problema, os sintomas e a correcao" — precisa de tipos de
# secao diferentes, e e por tipo que elas sao buscadas.
CAMPOS_DE_ABERTURA = ("sintomas", "causas", "correcao", "validacao")


def _trechos_de_abertura(sessao: Sessao, k: int, motor=None) -> list:
    """As secoes do manual travado que respondem sintomas **e** correcao.

    **Por `SELECT`, nao por semelhanca:** na busca vetorial, "qual o problema" e
    "como corrigir" disputam as mesmas vagas do top-k e a correcao fica de fora.
    Por tipo de secao, a cobertura e garantida por construcao.
    """
    achados = rag.secoes_por_campo(sessao.documentos, motor=motor,
                                   campos=CAMPOS_DE_ABERTURA)

    # Agrupa por campo para o corte nao comer um tipo inteiro: Doc3 tem 6 secoes
    # de correcao e 1 de sintomas — cortar a lista crua deixaria os sintomas de
    # fora dependendo da ordem.
    por_campo: dict[str, list] = {}
    for t in achados:
        por_campo.setdefault(t.campo, []).append(t)

    por_vez = max(1, k // max(1, len(por_campo)))
    escolhidos = []
    for campo in CAMPOS_DE_ABERTURA:
        escolhidos.extend(por_campo.get(campo, [])[:por_vez])
    return escolhidos[:k]


# --------------------------------------------------------------------------
# O anuncio do que os numeros disseram
# --------------------------------------------------------------------------


def fatos_do_diagnostico(diagnostico) -> dict:
    """O que o kNN apurou, com as chaves escritas por extenso.

    As chaves viram o vocabulario da frase que o modelo escreve: com
    `n_episodios` ele escreve "n episodios"; com "ocorrencias parecidas no
    historico" ele escreve portugues. E o mesmo dicionario que o **G5N** usa
    como lista de numeros permitidos, entao o que nao estiver aqui nao pode
    aparecer na resposta.
    """
    if diagnostico is None:
        return {}

    fatos: dict = {
        "familia indicada": diagnostico.familia or "indefinida",
        "vizinhos consultados": diagnostico.k,
        "vizinhos que votaram nessa familia": diagnostico.votos,
        "confianca": diagnostico.confianca,
        "ocorrencias parecidas no historico": diagnostico.n_episodios,
        "distancia do vizinho mais proximo": round(diagnostico.distancia_min, 3),
    }

    if (rpm := diagnostico.rpm_predominante) is not None:
        fatos["rotacao predominante dos vizinhos (rpm)"] = rpm

    # A anotacao do operador entra por ultimo e so quando existe: ela e o unico
    # fato que pode CONTRARIAR os outros, e o prompt manda destacar isso.
    if diagnostico.familia_do_operador:
        fatos["familia anotada pelo operador"] = diagnostico.familia_do_operador

    return fatos


def turno_de_classificacao(
    sessao: Sessao, diagnostico=None, cliente=None, sistema: str | None = None
) -> Turno | None:
    """A classificacao do evento entra na conversa como fala, nao como painel.

    **O modelo nao classifica.** Quando este no roda, o kNN ja comparou o evento
    com o historico, ja votou por familia e ja mediu a distancia. O que se pede
    ao modelo e uma frase sobre fatos fechados — e o **G5N** confere depois se
    cada numero escrito foi um numero apurado.

    Sem cliente, ou quando o G5N reprova, o proprio codigo escreve a frase. Nos
    dois casos os numeros sao os mesmos: o que se perde e a prosa, nunca o
    conteudo.

    Vem **antes** do resumo do manual de proposito. A conversa passa a ler como
    o trabalho e feito: primeiro o que a maquina mostrou, depois o que o
    procedimento manda fazer.
    """
    if diagnostico is None:
        return None

    t0 = time.time()
    fatos = fatos_do_diagnostico(diagnostico)

    turno = Turno(pergunta="")
    turno.abertura = True
    turno.classificacao = True
    turno.fatos = fatos

    if cliente is None:
        turno.resposta = prompts.classificacao.texto_de_fallback(fatos)
        turno.segundos = round(time.time() - t0, 2)
        sessao.turnos.append(turno)
        return turno

    mensagens = prompts.classificacao.montar(fatos, sistema=sistema)
    turno.prompt = prompts.classificacao.texto_enviado(mensagens)

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        turno.tentativas = tentativa
        r = cliente.gerar(mensagens)

        turno.resposta = r.texto.strip()
        turno.usou_llm = True
        turno.provedor, turno.modelo = r.provedor, r.modelo
        turno.tokens_entrada += r.tokens_entrada
        turno.tokens_saida += r.tokens_saida

        v = g.g5n_numeros_apurados(turno.resposta, fatos)
        if v.passou or tentativa == MAX_TENTATIVAS:
            turno.vereditos.append(v)
            turno.verificada = v.passou
            turno.degradou = not v.passou
            break

        # Reprovou e ainda ha tentativa: repete dizendo qual numero sobrou.
        mensagens = prompts.classificacao.montar(fatos, sistema=sistema) + [
            prompts.classificacao.HumanMessage(
                f"[Aviso do sistema: a resposta anterior foi rejeitada — "
                f"{v.mensagem} Escreva de novo usando apenas os numeros do "
                "bloco de fatos, exatamente como estao.]"
            )
        ]

    # Degradou: o texto do modelo nao pode ficar, porque ele contem numero que
    # ninguem apurou. Entra a versao de codigo, com os numeros certos.
    if turno.degradou:
        turno.resposta = prompts.classificacao.texto_de_fallback(fatos)

    turno.segundos = round(time.time() - t0, 2)
    sessao.turnos.append(turno)
    return turno


def resumo_inicial(sessao: Sessao, cliente=None, k: int = 6, motor=None,
                   sistema: str | None = None) -> Turno | None:
    """Responde problema, sintomas e correcao assim que o manual e fixado.

    O tecnico ia perguntar isso de qualquer jeito.

    **E um turno normal, nao um texto montado a mao:** passa por redacao e G5
    como os outros. Montado direto do banco, seria a primeira coisa que ele le
    e a unica sem guardrail.

    Sem `cliente`, degrada para o texto cru das secoes.

    O turno de classificacao, quando existe, ja esta no histórico e **nao**
    conta como "a conversa ja comecou": ele anuncia o que os numeros disseram,
    e o resumo do manual e a continuacao natural dele.
    """
    if not sessao.aberta:
        return None
    if any(not t.classificacao for t in sessao.turnos):
        return None

    turno = responder(
        sessao, PERGUNTA_DE_ABERTURA, cliente=cliente, k=k, motor=motor,
        trechos=_trechos_de_abertura(sessao, k, motor=motor), sistema=sistema,
    )
    turno.abertura = True
    return turno
