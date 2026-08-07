"""Guardrails — as travas que impedem o sistema de responder quando nao deve.

Sao **codigo**, nao instrucao no prompt. Um prompt pede; um `if` garante.
"""

from mp.guardrails.rules import (
    Veredito,
    avaliar,
    g0_entrada,
    g1_similaridade,
    g2_e_problema,
    g3_tem_documento,
    g4_trechos_relevantes,
    g5_citacoes_existem,
    g5n_numeros_apurados,
)

__all__ = [
    "Veredito",
    "avaliar",
    "g0_entrada",
    "g1_similaridade",
    "g2_e_problema",
    "g3_tem_documento",
    "g4_trechos_relevantes",
    "g5_citacoes_existem",
    "g5n_numeros_apurados",
]
