"""Prompts do projeto.

Um arquivo por tarefa. Prompt e texto versionado, nao string solta no meio do
codigo — quando a resposta muda de qualidade, e preciso saber o que mudou.

Sao tres, e eles pedem coisas diferentes ao modelo:

- `prescritivo` — a redacao da resposta. Devolve **texto**, e o G5 confere.
- `sintomas` — a separacao da fala do tecnico. Devolve **estrutura**, via
  `with_structured_output`, e o `conferir` do proprio modulo filtra.
- `classificacao` — o anuncio do que o kNN apurou. Devolve **texto**, e o G5N
  confere se cada numero escrito foi um numero apurado.
"""

from mp.llm.prompts import classificacao, sintomas
from mp.llm.prompts.prescritivo import (
    SISTEMA,
    bloco_de_assunto,
    bloco_de_fatos,
    bloco_de_trechos,
    montar,
    texto_enviado,
)

__all__ = [
    "SISTEMA",
    "montar",
    "bloco_de_assunto",
    "bloco_de_trechos",
    "bloco_de_fatos",
    "texto_enviado",
    "sintomas",
    "classificacao",
]
