"""Separar a fala do tecnico em sintomas independentes.

O problema que isto resolve
---------------------------
"O motor esta vibrando e o mancal esquentou" sao **dois** sintomas. Entrando como
um texto so, viram um vetor unico que fica perto da media dos dois — e a media
nao e nenhum deles. `rag.buscar_por_sintomas` foi escrito justamente para o
contrario: codifica cada sintoma separado e fica com o **maior** score por
trecho, para que a secao "Mancal Aquecido" nao seja derrubada por falar pouco de
vibracao.

Ou seja, a porta de entrada mais usada estava entregando ao motor de busca
exatamente o formato que o motor foi feito para evitar. Este modulo conserta
isso, e e o unico lugar do projeto onde o modelo devolve **estrutura** em vez de
prosa.

Por que isto nao fere a regra do no unico
-----------------------------------------
O modelo aqui nao escolhe manual, nao decide se busca e nao decide se responde.
Ele reescreve a frase do tecnico numa lista, e nada mais. Quem decide continua
sendo o G1T, sobre o resultado da busca.

E o pior caso e barato: se a separacao sair ruim, a busca fica como era antes —
com a frase inteira. Por isso `conferir` **descarta em silencio** e devolve a
descricao original, em vez de levantar erro.

O que o codigo confere depois
-----------------------------
Modelo separando texto pode acrescentar sintoma que ninguem relatou — e um
sintoma inventado vira uma consulta a mais, que pesa no ranking dos documentos.
`conferir` exige que cada item devolvido compartilhe uma palavra de conteudo com
o que o tecnico escreveu. Nao e prova de fidelidade; e o suficiente para barrar
"vibracao no eixo" aparecendo numa fala que so mencionava temperatura.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from mp.llm.client import BaseMessage, HumanMessage, SystemMessage

# Quantos sintomas fazem sentido tirar de uma fala. Acima disto, o modelo esta
# picotando a frase em pedacos sem conteudo proprio ("o motor", "esta ruim") —
# cada um vira uma consulta fraca que so adiciona ruido ao ranking.
MAXIMO_SINTOMAS = 6

# Palavras curtas ou de ligacao nao provam parentesco entre o item devolvido e a
# fala original: "o", "de", "que" aparecem em qualquer frase.
TAMANHO_MINIMO_PALAVRA = 4


class Sintomas(BaseModel):
    """O formulario que o modelo preenche. E o esquema do *tool call*."""

    sintomas: list[str] = Field(
        description=(
            "Cada sintoma observavel mencionado pelo tecnico, como frase curta e "
            "independente. Um item por observacao distinta."
        )
    )


SISTEMA = """Voce organiza o relato de um tecnico de manutencao industrial.

Sua unica tarefa e SEPARAR o que ele disse em sintomas independentes. Voce nao
diagnostica, nao sugere causa e nao sugere conserto.

Regras, todas obrigatorias:

1. Cada item e UMA observacao: um fenomeno, um lugar, um momento. "Vibrando e
   esquentando" sao dois itens.
2. Use as palavras do tecnico. Pode limpar a frase e completar o sujeito, mas
   nao troque o termo dele por um nome tecnico.
3. NAO acrescente sintoma que ele nao mencionou, nem o que voce esperaria
   encontrar junto. Se ele falou de temperatura, nao escreva nada sobre
   vibracao.
4. NAO escreva nome de defeito (desalinhamento, desbalanceamento, folga...),
   ainda que esteja obvio. Nomear o defeito e trabalho do sistema, com os dados.
5. Item curto, sem numeracao e sem marcador. De 1 a 6 itens.
6. Se a fala tiver uma observacao so, devolva uma lista de um item."""


def montar(descricao: str) -> list[BaseMessage]:
    """As mensagens da extracao. `SystemMessage` com as regras, `HumanMessage` com a fala."""
    return [
        SystemMessage(content=SISTEMA),
        HumanMessage(content=f"Relato do tecnico:\n{descricao.strip()}"),
    ]


def _palavras(texto: str) -> set[str]:
    """As palavras de conteudo, sem acento e em minuscula, para comparar.

    Sem acento porque o tecnico escreve "vibracao" e o modelo pode devolver
    "vibração" — a mesma palavra nao pode falhar na conferencia por causa da
    cedilha.
    """
    plano = unicodedata.normalize("NFKD", texto.lower())
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    return {p for p in re.findall(r"[a-z0-9]+", plano) if len(p) >= TAMANHO_MINIMO_PALAVRA}


def conferir(descricao: str, sintomas: list[str]) -> tuple[list[str], str]:
    """Filtra o que o modelo devolveu. Retorna `(sintomas, motivo)`.

    Guarda tres coisas, nesta ordem:

    - item sem nenhuma palavra de conteudo em comum com a fala original **sai**
      (regra 3 do prompt, agora como codigo);
    - a lista e cortada em `MAXIMO_SINTOMAS`;
    - sobrando nada, volta a descricao inteira como sintoma unico — que e
      exatamente o comportamento anterior a esta etapa.

    O `motivo` e para a tela: o tecnico ve o que o sistema registrou e percebe na
    hora se algo dele se perdeu.
    """
    originais = _palavras(descricao)
    limpos, descartados = [], []

    for bruto in sintomas or []:
        item = " ".join(str(bruto).split()).strip(" -•\t")
        if not item:
            continue
        if originais and not (_palavras(item) & originais):
            descartados.append(item)
            continue
        if item.lower() not in {s.lower() for s in limpos}:
            limpos.append(item)

    cortados = len(limpos) - MAXIMO_SINTOMAS
    limpos = limpos[:MAXIMO_SINTOMAS]

    if not limpos:
        return [descricao.strip()], (
            "O modelo nao devolveu nenhum sintoma aproveitavel; a busca usa a "
            "descricao inteira, como antes."
        )

    partes = [f"{len(limpos)} sintoma(s) separado(s) da sua descricao."]
    if descartados:
        partes.append(
            f"{len(descartados)} descartado(s) por nao aparecer(em) no que voce "
            f"escreveu: {'; '.join(descartados)}."
        )
    if cortados > 0:
        partes.append(f"{cortados} a mais foram cortados (limite de {MAXIMO_SINTOMAS}).")
    return limpos, " ".join(partes)
