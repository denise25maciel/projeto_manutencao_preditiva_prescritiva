"""Prompts do projeto.

Um arquivo por tarefa. Prompt e texto versionado, nao string solta no meio do
codigo — quando a resposta muda de qualidade, e preciso saber o que mudou.

Sao dois, e eles pedem coisas diferentes ao modelo:

- `prescritivo` — a redacao da resposta. Devolve **texto**, e o G5 confere.
- `sintomas` — a separacao da fala do tecnico. Devolve **estrutura**, via
  `with_structured_output`, e o `conferir` do proprio modulo filtra.
"""

from mp.llm.prompts import sintomas
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
]
