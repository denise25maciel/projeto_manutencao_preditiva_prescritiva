"""O prompt que anuncia, em portugues, o que a similaridade apurou.

**O modelo nao classifica nada aqui.** Quando este prompt e montado, o kNN ja
comparou o evento com as 166 mil leituras do historico, ja votou por familia e
ja mediu a distancia. O que chega ao modelo e uma lista de fatos fechados; o que
se pede a ele e uma frase.

E a mesma divisao do `prescritivo`, aplicada a outra materia: la os fatos sao
trechos de manual e a trava e a citacao (**G5**); aqui os fatos sao numeros e a
trava e o **G5N**, que confere se cada numero escrito foi um numero apurado.

**Por que isto existe, se um `f-string` resolveria.** Porque a classificacao
tinha de entrar na conversa como fala, e nao como painel — o tecnico esta lendo
um chat, e um bloco de metricas no meio quebra a leitura. O que nao muda e de
onde vem o conteudo: com modelo ou sem, os numeros sao os mesmos, e sem modelo o
proprio codigo escreve a frase (`texto_de_fallback`).
"""

from __future__ import annotations

from mp.llm.client import BaseMessage, HumanMessage, SystemMessage, como_texto

SISTEMA = """Voce anuncia a um tecnico de manutencao o resultado de uma comparacao
que JA FOI FEITA por um algoritmo. Voce nao classifica, nao diagnostica e nao
opina sobre a maquina.

Regras, todas obrigatorias:

1. Use SOMENTE os numeros e nomes do bloco de fatos. Nao arredonde para um valor
   mais bonito, nao converta unidade, nao acrescente numero nenhum.
2. Nao invente causa, sintoma nem recomendacao. Isso vem do manual, na proxima
   mensagem, e nao e a sua tarefa aqui.
3. Nao afirme certeza. O resultado e o que os vizinhos mais parecidos indicam —
   escreva nesse tom.
4. Se houver divergencia entre a anotacao do operador e o que os numeros
   indicam, diga isso explicitamente. E a informacao mais importante do bloco.
5. Portugues do Brasil, 2 a 4 frases, direto. Sem saudacao, sem introducao, sem
   oferecer ajuda, sem repetir a pergunta.
6. Nao use lista nem titulo. E uma fala dentro de uma conversa."""


def bloco_de_fatos(fatos: dict) -> str:
    """Os fatos apurados, um por linha.

    A chave e escrita por extenso porque ela vira o vocabulario da frase: com
    `n_episodios` o modelo escreve "n episodios"; com "ocorrencias parecidas no
    historico" ele escreve portugues.
    """
    linhas = [f"- {rotulo}: {valor}" for rotulo, valor in fatos.items()]
    return "\n".join(linhas) if linhas else "(nenhum fato apurado)"


def montar(fatos: dict, sistema: str | None = None) -> list[BaseMessage]:
    """As mensagens da redacao. `fatos` e o dicionario que o G5N vai conferir."""
    return [
        SystemMessage(sistema or SISTEMA),
        HumanMessage(
            "Anuncie ao tecnico o resultado desta comparacao.\n\n"
            "FATOS APURADOS PELO ALGORITMO:\n"
            + bloco_de_fatos(fatos)
        ),
    ]


def texto_de_fallback(fatos: dict) -> str:
    """A mesma noticia, escrita por codigo. Sem modelo, ou quando o G5N reprova.

    Nao e mensagem de erro: e a versao verificavel da mesma coisa. Por isso
    carrega os mesmos numeros, na mesma ordem — o que se perde e a prosa, nunca
    o conteudo.
    """
    familia = fatos.get("familia indicada") or "indefinida"
    partes = [f"Os numeros do sensor apontam **{familia}**."]

    if (conf := fatos.get("confianca")) is not None:
        votos = fatos.get("vizinhos que votaram nessa familia")
        total = fatos.get("vizinhos consultados")
        if votos is not None and total is not None:
            partes.append(
                f"{votos} dos {total} eventos mais parecidos do historico sao "
                f"dessa familia — confianca de {conf}."
            )
        else:
            partes.append(f"Confianca de {conf}.")

    if (n_ep := fatos.get("ocorrencias parecidas no historico")) is not None:
        partes.append(f"Ha {n_ep} ocorrencia(s) parecida(s) registrada(s).")

    if (anotado := fatos.get("familia anotada pelo operador")) is not None:
        if anotado != familia:
            partes.append(
                f"⚠️ O operador anotou **{anotado}**, e os numeros indicam "
                f"**{familia}**. Vale conferir."
            )
        else:
            partes.append(f"A anotacao do operador ({anotado}) confere.")

    return " ".join(partes)


def texto_enviado(mensagens: list[BaseMessage]) -> str:
    """O prompt inteiro, para a aba de auditoria."""
    return como_texto(mensagens)
