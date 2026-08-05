"""O prompt da pergunta de investigacao.

O segundo — e ultimo — lugar em que o modelo escreve. A tarefa dele aqui e mais
estreita ainda que a da resposta prescritiva: **transformar uma lista de
sintomas em uma pergunta**. Ele nao escolhe sobre o que perguntar.

Quem escolhe e um `SELECT`: `rag.secoes_de_sintomas` traz as secoes
`sintomas`, `causas`, `indicadores` e `diagnostico` dos documentos que ficaram
empatados. O modelo recebe esse texto pronto e devolve uma pergunta curta.

**Por que isso nao viola a regra do no unico.** A regra e que o modelo nao
decide o fluxo. Aqui ele nao decide: quem decidiu que ha empate foi o G1T, quem
decidiu quais documentos comparar foi o ranking, e quem trouxe o texto foi o
banco. Se o modelo escrever uma pergunta ruim, o pior caso e uma rodada
desperdicada — o teto de rodadas e a escolha humana no fim continuam de pe.

**O que este prompt nao pode fazer.** Nao ha guardrail depois dele: a pergunta
nao e uma afirmacao tecnica, entao nao ha citacao a conferir. Por isso as regras
abaixo sao restritivas — o maior risco e o modelo perguntar sobre um sintoma que
nenhum dos manuais menciona e conduzir o tecnico para fora do catalogo.
"""

from __future__ import annotations

from mp.llm.client import Mensagem

SISTEMA = """Voce ajuda um tecnico de manutencao a descrever melhor um problema.

A situacao: a descricao dada ate agora nao permitiu distinguir entre dois ou mais
procedimentos possiveis. Sua tarefa e escrever UMA pergunta que ajude a separar.

Regras, todas obrigatorias:

1. Pergunte SOMENTE sobre sintomas que aparecem nos trechos fornecidos. Voce tem
   conhecimento proprio sobre manutencao; aqui ele esta proibido.
2. Pergunte sobre algo que DIFERENCIE os candidatos — um sintoma que um deles
   descreve e o outro nao, ou que os dois descrevem de formas diferentes.
   Perguntar sobre o que todos tem em comum nao separa nada.
3. Uma pergunta so, no maximo duas frases. Sem lista, sem preambulo.
4. Linguagem de chao de fabrica: o que se ve, ouve, sente ou mede. Nada de nome
   de defeito, de secao ou de documento — quem responde e quem opera a maquina,
   nao quem escreveu o manual.
5. Nao diga quais sao os candidatos nem sugira um deles. A pergunta nao pode
   induzir a resposta.
6. Portugues do Brasil, direto. Nao repita o que o tecnico ja contou."""


def bloco_de_sintomas(trechos) -> str:
    """As secoes de sintoma de cada candidato, agrupadas por documento.

    Agrupar importa: o modelo precisa ver o que e de um e o que e do outro para
    achar o que difere. Numa lista misturada, tudo parece do mesmo lugar.
    """
    if not trechos:
        return "(nenhuma secao de sintomas cadastrada para os candidatos)"

    por_documento: dict[str, list] = {}
    for t in trechos:
        por_documento.setdefault(t.documento_id, []).append(t)

    partes = []
    for doc, itens in por_documento.items():
        linhas = [f"--- CANDIDATO {doc} ---"]
        for t in itens:
            campo = f" [{t.campo}]" if t.campo else ""
            linhas.append(f"secao {t.numero} — {t.titulo}{campo}\n{t.texto.strip()}")
        partes.append("\n\n".join(linhas))

    return "\n\n".join(partes)


def montar(sintomas: list[str], trechos, candidatos: list[str]) -> list[Mensagem]:
    """Monta as mensagens da pergunta de investigacao.

    `sintomas` sao as falas do tecnico ate aqui, uma por linha — para o modelo
    nao repetir o que ja foi dito. `trechos` vem do `SELECT`; `candidatos` sao
    os documentos empatados, so para o modelo saber quantos lados existem.
    """
    ja_dito = "\n".join(f"- {s.strip()}" for s in sintomas if s.strip())

    blocos = [
        f"O TECNICO JA CONTOU:\n{ja_dito or '(nada ainda)'}",
        f"ISSO NAO SEPAROU {len(candidatos)} PROCEDIMENTOS POSSIVEIS.",
        "SECOES DE SINTOMAS DE CADA CANDIDATO (a unica fonte permitida):\n\n"
        + bloco_de_sintomas(trechos),
        "Escreva a pergunta que melhor separa estes candidatos.",
    ]

    return [Mensagem("system", SISTEMA), Mensagem("user", "\n\n".join(blocos))]
