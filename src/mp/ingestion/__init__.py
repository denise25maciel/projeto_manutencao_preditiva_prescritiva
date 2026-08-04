"""Ingestao de insumos brutos.

`documents` converte os PDFs de procedimento em Markdown com secoes numeradas —
o formato que a Parte 4 vai fatiar em chunks.
"""

from mp.ingestion.documents import (
    campos_pendentes,
    carregar_markdowns,
    converter_todos,
    cobertura_por_familia,
    extrair_texto,
    matriz_campos,
    pdf_para_markdown,
    separar_secoes,
)

__all__ = [
    "extrair_texto",
    "separar_secoes",
    "pdf_para_markdown",
    "converter_todos",
    "carregar_markdowns",
    "matriz_campos",
    "campos_pendentes",
    "cobertura_por_familia",
]
