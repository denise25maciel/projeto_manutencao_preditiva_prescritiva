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
from mp.ingestion import (  # noqa: E402
    analise_corte_interno,
    comparar_abordagens,
    serie_com_eventos,
    series_por_evento,
    criterios_limiar,
    campos_pendentes,
    carregar_markdowns,
    cobertura_por_familia,
    construir_eventos,
    converter_todos,
    diagnostico_eventos,
    diagnostico_ordenacao,
    exemplo_desordem,
    matriz_campos,
    resumo_por_rotulo,
    validar_eventos,
)
from mp.analysis import (  # noqa: E402
    analise_intervalos,
    assinatura_de_rotulo,
    assinaturas_por_rotulo,
    carregar,
    e_estado,
    colunas_a_descartar,
    colunas_constantes,
    colunas_numericas,
    colunas_redundantes,
    comparar_com_global,
    duplicatas_consecutivas,
    janela_temporal,
    nulos_por_coluna,
    ordenar_por_tempo,
    outliers_iqr,
    perfil_rotulos,
    resumo_geral,
    serie_temporal,
    sugerir_familias,
    taxa_amostragem,
    timestamps_duplicados,
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
def r_tempos_duplicados():
    return timestamps_duplicados(dados())


@st.cache_data(**CACHE)
def r_intervalos():
    """Analise que justifica onde cortar um episodio."""
    return analise_intervalos(dados())


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


# --- series temporais -------------------------------------------------------
#
# `colunas` chega como tupla porque o cache do Streamlit exige argumento
# hasheavel — lista nao serve.


@st.cache_data(**CACHE)
def r_serie_temporal(rotulo: str, colunas: tuple[str, ...], max_pontos: int | None = None):
    """`max_pontos` e o orcamento POR ROTULO.

    Quando varios rotulos vao para o mesmo grafico, a pagina divide o teto
    global entre eles — senao o payload enviado ao navegador cresce com o
    numero de series e o Vega estoura o limite de linhas.
    """
    return serie_temporal(dados(), rotulo, list(colunas), max_pontos=max_pontos)


@st.cache_data(**CACHE)
def r_ordenado(rotulo: str):
    """Leituras do rotulo em ordem cronologica, com sessao e delta."""
    return ordenar_por_tempo(dados(), rotulo)


# --- eventos ----------------------------------------------------------------


@st.cache_data(**CACHE)
def r_eventos(limite_intervalo_s: float | None = None):
    """`(leituras com a coluna evento, tabela de eventos)`."""
    return construir_eventos(dados(), limite_intervalo_s=limite_intervalo_s)


@st.cache_data(**CACHE)
def r_analise_corte(limite_intervalo_s: float | None = None):
    """Como um corte por tempo agiria sobre os eventos ja formados."""
    leituras, _ = r_eventos(limite_intervalo_s)
    return analise_corte_interno(leituras)


@st.cache_data(**CACHE)
def r_criterios_limiar(limite_intervalo_s: float | None = None):
    """Limiar derivado por criterios automaticos, e os que falham aqui."""
    return criterios_limiar(r_analise_corte(limite_intervalo_s)["intervalos"])


@st.cache_data(**CACHE)
def r_comparar_abordagens():
    """Ordenar-depois-separar contra separar-depois-ordenar.

    Acrescenta a familia a cada evento das duas bases, para a tela poder filtrar.
    A familia vem do `fault_map.yaml` — o catalogo curado, nao o heuristico.
    """
    from mp.retrieval import familia_de

    resultado = comparar_abordagens(dados())
    for chave in ("eventos_a", "eventos_b"):
        eventos = resultado[chave].copy()
        eventos["familia"] = eventos[config.COLUNA_ROTULO].map(familia_de)
        resultado[chave] = eventos
    return resultado


@st.cache_data(**CACHE)
def r_serie_com_eventos(versao: str, coluna: str, familias: tuple[str, ...],
                        max_pontos: int = 3000):
    """Leituras de uma medida ao longo do tempo, sabendo o evento de cada ponto.

    `versao` e "A" ou "B" — muda so o agrupamento; as leituras sao as mesmas.
    """
    from mp.retrieval import familia_de

    comp = r_comparar_abordagens()
    leituras = comp["leituras_a"] if versao == "A" else comp["leituras_b"]

    if familias:
        familia_da_linha = leituras[config.COLUNA_ROTULO].map(familia_de)
        leituras = leituras[familia_da_linha.isin(familias)]

    return serie_com_eventos(leituras, coluna, "evento", max_pontos=max_pontos)


@st.cache_data(**CACHE)
def r_series_por_evento(versao: str, eventos: tuple[int, ...],
                        colunas: tuple[str, ...], max_pontos_por_evento: int = 200):
    """Serie padronizada de cada evento, alinhada no tempo decorrido."""
    comp = r_comparar_abordagens()
    leituras = comp["leituras_a"] if versao == "A" else comp["leituras_b"]
    return series_por_evento(
        leituras, list(eventos), list(colunas),
        max_pontos_por_evento=max_pontos_por_evento,
    )


@st.cache_data(**CACHE)
def r_ordenacao():
    return diagnostico_ordenacao(dados())


@st.cache_data(**CACHE)
def r_exemplo_desordem():
    return exemplo_desordem(dados())


@st.cache_data(**CACHE)
def r_validacao_eventos(limite_intervalo_s: float | None = None):
    leituras, eventos = r_eventos(limite_intervalo_s)
    return validar_eventos(leituras, eventos, dados())


@st.cache_data(**CACHE)
def r_resumo_eventos(limite_intervalo_s: float | None = None):
    _, eventos = r_eventos(limite_intervalo_s)
    return resumo_por_rotulo(eventos)


@st.cache_data(**CACHE)
def r_diagnostico_eventos(limite_intervalo_s: float | None = None,
                          limite_alerta_s: float = 60.0):
    leituras, eventos = r_eventos(limite_intervalo_s)
    return diagnostico_eventos(leituras, eventos, limite_alerta_s=limite_alerta_s)


# --- documentos -------------------------------------------------------------
#
# A conversao PDF -> Markdown escreve em disco, entao NAO fica em cache: e
# acionada por botao. So a leitura dos `.md` e cacheada, com o numero de
# arquivos e o mtime mais recente como chave — assim uma reconversao invalida
# o cache sozinha.


def _versao_md() -> tuple[int, float]:
    if not config.DOCS_MD_DIR.exists():
        return (0, 0.0)
    arqs = list(config.DOCS_MD_DIR.glob("*.md"))
    return (len(arqs), max((a.stat().st_mtime for a in arqs), default=0.0))


def converter_pdfs():
    """Roda a conversao. Efeito colateral em disco — sem cache, de proposito."""
    return converter_todos()


@st.cache_data(**CACHE)
def _r_docs(versao):
    return carregar_markdowns()


def r_docs():
    return _r_docs(_versao_md())


@st.cache_data(**CACHE)
def _r_matriz(versao):
    return matriz_campos(carregar_markdowns())


def r_matriz_campos():
    return _r_matriz(_versao_md())


@st.cache_data(**CACHE)
def _r_pendentes(versao):
    return campos_pendentes(carregar_markdowns())


def r_pendentes():
    return _r_pendentes(_versao_md())


@st.cache_data(**CACHE)
def r_familias_banner():
    """Familia sugerida x volume no banner.csv — o lado direito do diagrama."""
    df = dados()
    fam = sugerir_familias(df[config.COLUNA_ROTULO].dropna().unique())
    contagem = df[config.COLUNA_ROTULO].value_counts().rename("n_leituras")
    fam = fam.join(contagem, on="fault")
    agrupado = (
        fam.groupby("familia_sugerida", as_index=False)
        .agg(n_rotulos=("fault", "count"), n_leituras=("n_leituras", "sum"))
        .rename(columns={"familia_sugerida": "familia"})
    )
    # `e_problema` vem do NOME DA FAMILIA, nao do maximo sobre os rotulos.
    # Com o maximo, a familia `teste` era marcada como defeito por causa de
    # `new_tes` — rotulo truncado que nao contem o radical `teste` e escapa da
    # checagem por substring. A familia e a unidade de decisao dos guardrails.
    agrupado["e_problema"] = ~agrupado["familia"].map(e_estado)
    return agrupado.sort_values("n_leituras", ascending=False).reset_index(drop=True)


@st.cache_data(**CACHE)
def _r_cobertura(versao, familias):
    return cobertura_por_familia(carregar_markdowns(), familias)


def r_cobertura():
    return _r_cobertura(_versao_md(), tuple(r_familias_banner()["familia"]))


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
