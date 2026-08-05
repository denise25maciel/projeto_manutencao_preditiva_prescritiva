"""Parte 3 — o motor de similaridade.

O unico modulo que responde "que falha e essa?" a partir de numeros. Todo o
resto do sistema parte do rotulo ja resolvido.
"""

from mp.similarity.features import Preparador, colunas_de_similaridade
from mp.similarity.search import (
    K_PADRAO,
    Diagnostico,
    Indice,
    avaliar_por_grupo,
)

__all__ = [
    "Preparador",
    "colunas_de_similaridade",
    "Indice",
    "Diagnostico",
    "avaliar_por_grupo",
    "K_PADRAO",
]
