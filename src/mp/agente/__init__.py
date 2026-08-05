"""O agente da Parte 5: os nos do fluxo e o estado da conversa.

Nome herdado do desenho, nao da tecnologia: **isto nao e um agente autonomo**.
Nenhum no deixa o modelo escolher o que fazer. Ver `grafo.py`.
"""

from mp.agente.estado import Sessao, Turno
from mp.agente.grafo import (
    abrir_sessao,
    abrir_sessao_por_texto,
    continuar_investigacao,
    fixar_documento,
    no_escopo,
    responder,
)

__all__ = [
    "Sessao",
    "Turno",
    "abrir_sessao",
    "abrir_sessao_por_texto",
    "continuar_investigacao",
    "fixar_documento",
    "responder",
    "no_escopo",
]
