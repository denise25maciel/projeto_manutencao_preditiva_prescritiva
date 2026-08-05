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

from mp.llm.client import Mensagem

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


def bloco_de_historico(historico) -> str:
    """As voltas anteriores da conversa, so com o que foi verificado.

    O que chega aqui ja passou pelo filtro do `Turno.texto_para_historico`:
    resposta reprovada no G5 nao entra: no lugar dela vem o trecho do manual.
    """
    if not historico:
        return ""

    partes = []
    for pergunta, resposta in historico:
        partes.append(f"Tecnico: {pergunta}\nVoce respondeu: {resposta}")
    return (
        "CONVERSA ATE AQUI (so o que foi verificado; use para entender o contexto "
        "da pergunta nova, nunca como fonte de fato tecnico):\n\n"
        + "\n\n".join(partes)
    )


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


def montar(
    pergunta: str,
    familia: str,
    trechos,
    fatos: dict | None = None,
    historico=None,
) -> list[Mensagem]:
    """Monta as mensagens da resposta prescritiva.

    `historico` e a lista `(pergunta, resposta_verificada)` das voltas
    anteriores. Ele entra como **contexto da conversa**, nunca como fonte: o
    prompt diz isso explicitamente, e a regra 1 do sistema continua valendo —
    so os trechos autorizam uma afirmacao tecnica.
    """
    blocos = [f"DEFEITO IDENTIFICADO: {familia}"]

    if texto := bloco_de_historico(historico):
        blocos.append(texto)
    if texto := bloco_de_fatos(fatos):
        blocos.append(texto)

    blocos.append(
        "TRECHOS DO PROCEDIMENTO (a unica fonte permitida):\n\n"
        + bloco_de_trechos(trechos)
    )
    blocos.append(f"PERGUNTA DO TECNICO:\n{pergunta.strip()}")

    return [Mensagem("system", SISTEMA), Mensagem("user", "\n\n".join(blocos))]


def texto_enviado(mensagens: list[Mensagem]) -> str:
    """As mensagens como um texto so — para mostrar na tela o que de fato foi enviado.

    Existe para auditoria: a tela precisa poder provar que o modelo nao recebeu
    nada alem disso.
    """
    return "\n\n".join(f"[{m.papel.upper()}]\n{m.conteudo}" for m in mensagens)
