"""Recuperacao: catalogo de falhas e, a partir da Parte 4, o RAG."""

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
    "carregar_fault_map",
    "familia_de",
    "documentos_de",
    "is_problem",
    "resolver",
    "tabela_familias",
    "validar_cobertura",
]
