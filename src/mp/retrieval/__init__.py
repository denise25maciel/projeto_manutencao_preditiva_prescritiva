"""Recuperacao: catalogo de falhas e busca nos procedimentos.

- `catalog`    — o `fault_map.yaml`: rotulo -> familia -> documento
- `embeddings` — texto vira numeros, para a busca por significado
- `rag`        — busca em dois estagios: filtro exato, depois semelhanca
"""

from mp.retrieval import embeddings
from mp.retrieval.rag import (
    CAMPOS_PRESCRITIVOS,
    Resultado,
    Trecho,
    buscar,
    buscar_prescritivo,
    como_tabela,
    indexar,
    modelo_indexado,
)
from mp.retrieval.catalog import (
    carregar_fault_map,
    documentos_de,
    familia_de,
    is_problem,
    resolver,
    tabela_familias,
    validar_cobertura,
)

__all__ = [
    "embeddings",
    "indexar",
    "modelo_indexado",
    "buscar",
    "buscar_prescritivo",
    "como_tabela",
    "Trecho",
    "Resultado",
    "CAMPOS_PRESCRITIVOS",
    "carregar_fault_map",
    "familia_de",
    "documentos_de",
    "is_problem",
    "resolver",
    "tabela_familias",
    "validar_cobertura",
]
