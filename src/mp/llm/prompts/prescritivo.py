"""O prompt da resposta prescritiva.

O modelo entra no fluxo em um unico ponto e com uma unica tarefa: **redigir**.
Quando este prompt e montado, todas as decisoes ja foram tomadas por codigo —
qual a familia, se ela e defeito, se tem manual, quais trechos usar. Nada aqui
escolhe nada.

Por isso o prompt e curto. Ele nao precisa ensinar o modelo a decidir; precisa
impedi-lo de acrescentar.

**O que este texto NAO garante.** Pedir "nao invente" e uma sugestao, nao uma
trava — o modelo pode desobedecer e nao ha como saber pelo prompt. Quem garante
sao o G4 (antes) e o G5 (depois), que sao codigo. As instrucoes abaixo existem
para tornar a desobediencia rara, nao impossivel.
"""

from __future__ import annotations

from mp.llm.client import BaseMessage, HumanMessage, SystemMessage, como_texto

SISTEMA = """Voce redige respostas de manutencao industrial para um tecnico em campo.

Regras, todas obrigatorias:

1. Use SOMENTE o que estiver nos trechos fornecidos. Voce tem conhecimento
   proprio sobre manutencao; aqui ele esta proibido. Se o trecho nao diz, voce
   nao diz.
2. Toda afirmacao tecnica termina com a fonte, no formato exato: (Doc1, secao 4).
   Sem fonte, a afirmacao nao pode existir.
3. Nunca cite um documento ou secao que nao esteja nos trechos fornecidos.
4. Nunca invente numero. Torque, folga, tolerancia, temperatura, intervalo: se o
   valor nao aparece escrito no trecho, escreva "o procedimento nao especifica o
   valor" em vez de estimar.
5. Se os trechos nao respondem a pergunta, responda exatamente:
   "O procedimento disponivel nao cobre esta pergunta." e pare.
6. Escreva em portugues do Brasil, direto, em passos numerados quando for uma
   sequencia de acoes. Sem introducao, sem resumo final, sem oferecer ajuda.
7. Nao repita o enunciado da pergunta."""


def bloco_de_trechos(trechos) -> str:
    """Os trechos recuperados, com o endereco que a resposta tera de citar.

    O rotulo de cada bloco e exatamente o formato que o **G5** procura depois.
    Se aqui e ali divergirem, toda resposta valida sera reprovada.
    """
    if not trechos:
        return "(nenhum trecho disponivel)"

    partes = []
    for t in trechos:
        campo = f" [{t.campo}]" if t.campo else ""
        # A pagina do PDF NAO entra aqui de proposito. Ela e metadado do banco e
        # e a interface que a acrescenta; se estivesse no prompt, o modelo teria
        # de copia-la e poderia errar o numero. Fato apurado nao se pede ao
        # modelo — entrega-se pronto ou nao se usa.
        partes.append(
            f"--- {t.documento_id}, secao {t.numero}: {t.titulo}{campo}\n{t.texto.strip()}"
        )
    return "\n\n".join(partes)


def bloco_de_fatos(fatos: dict | None) -> str:
    """Os numeros vindos do banco, ja prontos.

    Fronteira deterministica/generativa: contagem, frequencia e janela temporal
    sao consulta, nunca geracao. O modelo recebe o numero calculado e no maximo
    o repete — se ele produzisse esses valores, seriam palpite.
    """
    if not fatos:
        return ""

    linhas = [f"- {chave}: {valor}" for chave, valor in fatos.items()]
    return "DADOS APURADOS NO HISTORICO (ja calculados, nao recalcule):\n" + "\n".join(linhas)


def bloco_de_assunto(familia: str | None, familias=None) -> str:
    """A primeira linha do prompt: sobre o que e esta conversa.

    **Nem toda sessao sabe o nome do defeito.** Quando ela nasce de um evento de
    sensor, o kNN apurou a familia e o bloco pode afirma-la. Quando nasce de uma
    descricao escrita, o que foi identificado e o *procedimento* — e um
    procedimento pode cobrir varias familias: o Doc1 atende quatro tipos de
    rolamento, porque o conserto e o mesmo para os quatro.

    Nesse caso nao ha tipo a afirmar. Escrever um deles aqui entregaria ao
    modelo, como fato ja apurado, a primeira linha de uma lista — e por essa
    porta o G5 nao olha: ele confere as citacoes da resposta, nao os fatos da
    pergunta.
    """
    familias = list(familias or ([familia] if familia else []))

    if familia:
        return f"DEFEITO IDENTIFICADO: {familia}"

    if len(familias) > 1:
        return (
            "PROCEDIMENTO DESTA CONVERSA: cobre " + ", ".join(familias) + ".\n"
            "Qual desses tipos e o caso NAO foi apurado. Nao afirme um deles; se "
            "os trechos ensinam a distinguir, apresente o criterio como criterio."
        )

    return "PROCEDIMENTO DESTA CONVERSA: o manual fixado para esta conversa."


def montar(
    pergunta: str,
    familia: str | None,
    trechos,
    fatos: dict | None = None,
    historico: list[BaseMessage] | None = None,
    familias=None,
    sistema: str | None = None,
) -> list[BaseMessage]:
    """Monta as mensagens da resposta prescritiva.

    A forma da lista:

        SystemMessage   as regras
        (historico)     HumanMessage / AIMessage alternados das voltas anteriores
        HumanMessage    assunto + fatos + trechos + a pergunta desta volta

    `historico` ja vem como **mensagens**, montado por
    `Sessao.historico_para_prompt`. Antes era um paragrafo dentro da mensagem do
    usuario, descrevendo a conversa; agora e a conversa. Nao muda o que o modelo
    pode afirmar — o historico continua sendo contexto, nunca fonte, e a regra 1
    do sistema continua valendo.

    `sistema` substitui as regras padrao. Existe para a tela poder edita-las **e
    mostrar que isso nao afrouxa nada**: apagar a regra da citacao nao faz o G5
    aceitar resposta sem fonte, porque o G5 e codigo. Guardrail que um campo de
    texto desliga nao era guardrail.

    `familia` vazia com `familias` preenchida e a sessao que travou o manual sem
    ter apurado o tipo — ver `bloco_de_assunto`.
    """
    blocos = [bloco_de_assunto(familia, familias)]

    if texto := bloco_de_fatos(fatos):
        blocos.append(texto)

    blocos.append(
        "TRECHOS DO PROCEDIMENTO (a unica fonte permitida):\n\n"
        + bloco_de_trechos(trechos)
    )
    blocos.append(f"PERGUNTA DO TECNICO:\n{pergunta.strip()}")

    return [
        SystemMessage(content=sistema if sistema is not None else SISTEMA),
        *(historico or []),
        HumanMessage(content="\n\n".join(blocos)),
    ]


def texto_enviado(mensagens: list[BaseMessage]) -> str:
    """As mensagens como um texto so — para mostrar na tela o que de fato foi enviado.

    Existe para auditoria: a tela precisa poder provar que o modelo nao recebeu
    nada alem disso. O rotulo de cada bloco (`[SYSTEM]`, `[HUMAN]`, `[AI]`) e o
    tipo real da mensagem, entao a tela mostra tambem **como** o historico foi
    enviado, nao so o que.
    """
    return como_texto(mensagens)
