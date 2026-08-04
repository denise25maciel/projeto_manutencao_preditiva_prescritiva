"""Ponte entre a UI e o modulo `mp.analysis`.

A UI nao calcula nada: so chama funcoes de `src/mp/` e desenha o resultado
(principio 5 do GUIA.md). Este arquivo existe apenas para (a) achar o pacote
e (b) guardar os resultados em cache, porque o Streamlit reexecuta o script
inteiro a cada clique.

Nota de arquitetura: a partir da Parte 5 a UI passa a falar com a API por HTTP,
para nao recarregar o LLM a cada rerun. Na Parte 0 nao ha modelo nenhum — e
tudo pandas, cacheavel — entao o import direto e o caminho simples e honesto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# O pacote ainda nao esta instalado (`pip install -e .` e opcional no MVP),
# entao apontamos para src/ na mao.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mp import config  # noqa: E402
from mp.analysis import (  # noqa: E402
    assinatura_de_rotulo,
    assinaturas_por_rotulo,
    carregar,
    colunas_a_descartar,
    colunas_constantes,
    colunas_numericas,
    colunas_redundantes,
    comparar_com_global,
    duplicatas_consecutivas,
    janela_temporal,
    nulos_por_coluna,
    outliers_iqr,
    perfil_rotulos,
    resumo_geral,
    sugerir_familias,
    taxa_amostragem,
)

# TTL infinito: o CSV bruto nao muda durante a sessao.
CACHE = dict(show_spinner=False)


@st.cache_data(**CACHE)
def dados():
    """DataFrame bruto. Carregado uma vez por sessao."""
    return carregar()


@st.cache_data(**CACHE)
def caminho_do_csv() -> str:
    return str(config.caminho_csv())


# --- perfil -----------------------------------------------------------------

@st.cache_data(**CACHE)
def r_resumo():
    return resumo_geral(dados())


@st.cache_data(**CACHE)
def r_nulos():
    return nulos_por_coluna(dados())


@st.cache_data(**CACHE)
def r_rotulos():
    return perfil_rotulos(dados())


@st.cache_data(**CACHE)
def r_janela():
    return janela_temporal(dados())


@st.cache_data(**CACHE)
def r_amostragem():
    return taxa_amostragem(dados())


@st.cache_data(**CACHE)
def r_familias():
    return sugerir_familias(dados()[config.COLUNA_ROTULO].dropna().unique())


# --- qualidade --------------------------------------------------------------

@st.cache_data(**CACHE)
def r_constantes():
    return colunas_constantes(dados())


@st.cache_data(**CACHE)
def r_redundantes():
    return colunas_redundantes(dados())


@st.cache_data(**CACHE)
def r_duplicatas():
    d = duplicatas_consecutivas(dados())
    # A mascara e uma Series de 166 mil booleanos que a UI nao usa; fora do cache.
    return {k: v for k, v in d.items() if k != "mascara"}


@st.cache_data(**CACHE)
def r_outliers_global():
    return outliers_iqr(dados())


@st.cache_data(**CACHE)
def r_outliers_do_rotulo(rotulo: str):
    df = dados()
    return outliers_iqr(df[df[config.COLUNA_ROTULO] == rotulo])


@st.cache_data(**CACHE)
def r_descartar():
    return colunas_a_descartar(dados())


# --- assinaturas ------------------------------------------------------------

@st.cache_data(**CACHE)
def r_assinaturas(min_leituras: int = 1):
    return assinaturas_por_rotulo(dados(), min_leituras=min_leituras)


@st.cache_data(**CACHE)
def r_assinatura(rotulo: str):
    return assinatura_de_rotulo(dados(), rotulo)


@st.cache_data(**CACHE)
def r_comparacao(rotulo: str):
    return comparar_com_global(dados(), rotulo)


@st.cache_data(**CACHE)
def r_serie(rotulo: str, coluna: str):
    """Valores de uma feature para um rotulo — insumo dos histogramas."""
    df = dados()
    return df.loc[df[config.COLUNA_ROTULO] == rotulo, coluna].to_numpy()


@st.cache_data(**CACHE)
def r_numericas():
    return colunas_numericas(dados())


# --- utilitarios de pagina --------------------------------------------------

def configurar_pagina(titulo: str, icone: str = "🔧") -> None:
    st.set_page_config(page_title=f"{titulo} — Manutencao Prescritiva",
                       page_icon=icone, layout="wide")


def aviso_csv_ausente(erro: Exception) -> None:
    """Mensagem util quando o CSV bruto nao esta no lugar."""
    st.error("Nao encontrei o `banner.csv`.")
    st.code(str(erro))
    st.info(
        "O CSV fica fora do git (dado da empresa). Coloque-o em `data/raw/banner.csv` "
        "ou defina a variavel de ambiente `MP_CSV` com o caminho completo."
    )
    st.stop()
