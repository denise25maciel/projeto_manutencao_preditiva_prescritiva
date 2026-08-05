"""Clientes de modelo de linguagem, plugaveis.

Todo provedor cumpre o mesmo contrato. Trocar de um para outro nao muda uma
linha do pipeline — se mudar, a abstracao esta errada.
"""

from mp.llm.client import (
    Cliente,
    Mensagem,
    PROVEDORES,
    Resposta,
    criar,
    provedores_disponiveis,
)

__all__ = [
    "Cliente",
    "Mensagem",
    "Resposta",
    "criar",
    "provedores_disponiveis",
    "PROVEDORES",
]
