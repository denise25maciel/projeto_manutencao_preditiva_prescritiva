"""Ingestao de insumos brutos — o que TRANSFORMA o dado.

- `sensors`   — agrupa as leituras de vibracao em eventos
- `documents` — converte os PDFs de procedimento em Markdown com secoes numeradas

Contraste com `analysis/`, que so descreve e nunca altera nada.
"""

from mp.ingestion.sensors import (
    analise_corte_interno,
    criterios_limiar,
    construir_eventos,
    diagnostico_eventos,
    diagnostico_ordenacao,
    exemplo_desordem,
    resumo_por_rotulo,
    validar_eventos,
)
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
    "analise_corte_interno",
    "criterios_limiar",
    "construir_eventos",
    "diagnostico_eventos",
    "diagnostico_ordenacao",
    "exemplo_desordem",
    "validar_eventos",
    "resumo_por_rotulo",
    "extrair_texto",
    "separar_secoes",
    "pdf_para_markdown",
    "converter_todos",
    "carregar_markdowns",
    "matriz_campos",
    "campos_pendentes",
    "cobertura_por_familia",
]
