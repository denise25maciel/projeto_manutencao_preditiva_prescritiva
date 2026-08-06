"""Prompts do projeto.

Um arquivo por tarefa. Prompt e texto versionado, nao string solta no meio do
codigo — quando a resposta muda de qualidade, e preciso saber o que mudou.
"""

from mp.llm.prompts.prescritivo import (
    SISTEMA,
    bloco_de_assunto,
    bloco_de_fatos,
    bloco_de_historico,
    bloco_de_trechos,
    montar,
    texto_enviado,
)

__all__ = [
    "SISTEMA",
    "montar",
    "bloco_de_assunto",
    "bloco_de_trechos",
    "bloco_de_historico",
    "bloco_de_fatos",
    "texto_enviado",
]
