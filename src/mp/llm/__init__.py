"""Clientes de modelo de linguagem, plugaveis.

Todo provedor cumpre o mesmo contrato. Trocar de um para outro nao muda uma
linha do pipeline — se mudar, a abstracao esta errada.

As mensagens sao as do LangChain (`SystemMessage`, `HumanMessage`,
`AIMessage`), reexportadas daqui para que o resto do projeto tenha **um** lugar
de onde importa-las. Ver `client.py` para o porque de o framework parar aqui.
"""

from mp.llm.client import (
    AIMessage,
    BaseMessage,
    Cliente,
    HumanMessage,
    Mensagem,
    PROVEDORES,
    Resposta,
    SystemMessage,
    como_texto,
    criar,
    provedores_disponiveis,
)

__all__ = [
    "Cliente",
    "Mensagem",
    "Resposta",
    "SystemMessage",
    "HumanMessage",
    "AIMessage",
    "BaseMessage",
    "como_texto",
    "criar",
    "provedores_disponiveis",
    "PROVEDORES",
]
