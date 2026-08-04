"""Modulo de analise (Parte 0).

Le o CSV bruto e descreve o que veio, sem transformar nada. Nenhuma funcao
daqui remove linha, preenche nulo ou trata outlier — isso e a Parte 1.

As checagens de `quality` voltam a rodar na ingestao em producao, por isso a
analise e modulo e nao so notebook.
"""

from mp.analysis.loader import carregar, colunas_numericas
from mp.analysis.profiling import (
    e_estado,
    janela_temporal,
    nulos_por_coluna,
    perfil_rotulos,
    resumo_geral,
    sugerir_familias,
    taxa_amostragem,
)
from mp.analysis.quality import (
    colunas_constantes,
    colunas_redundantes,
    duplicatas_consecutivas,
    outliers_iqr,
    relatorio_qualidade,
)
from mp.analysis.signatures import (
    assinatura_de_rotulo,
    assinaturas_por_rotulo,
    colunas_a_descartar,
    comparar_com_global,
)

__all__ = [
    "carregar",
    "colunas_numericas",
    "resumo_geral",
    "perfil_rotulos",
    "nulos_por_coluna",
    "janela_temporal",
    "taxa_amostragem",
    "sugerir_familias",
    "e_estado",
    "colunas_constantes",
    "colunas_redundantes",
    "duplicatas_consecutivas",
    "outliers_iqr",
    "relatorio_qualidade",
    "assinaturas_por_rotulo",
    "assinatura_de_rotulo",
    "comparar_com_global",
    "colunas_a_descartar",
]
