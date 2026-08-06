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
# Import barato: os SDKs so sao carregados dentro de cada cliente, na hora de
# conectar. Aqui vem apenas o texto que descreve cada provedor.
from mp.llm.client import DESCRICAO as DESCRICAO_PROVEDOR  # noqa: E402
# O minimo do G4. A tela mostra ao lado de cada score para o numero poder ser
# comparado com alguma coisa; reexportado aqui porque a UI nao importa `mp`.
from mp.guardrails.rules import SCORE_MINIMO_CHUNK  # noqa: E402
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
    saltos_no_arquivo,
    serie_bruta,
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


# --- dado bruto -------------------------------------------------------------


@st.cache_data(**CACHE)
def r_serie_bruta(colunas: tuple[str, ...], inicio: int, quantidade: int,
                  rotulos: tuple[str, ...] = ()):
    """Trecho do arquivo na ordem de leitura, sem tratamento nenhum.

    `rotulos` vazio = o arquivo inteiro. Filtrar seleciona linhas e nada mais:
    a ordem e a numeracao continuam sendo as do arquivo.
    """
    return serie_bruta(dados(), list(colunas), inicio=inicio,
                       quantidade=quantidade, rotulos=list(rotulos))


@st.cache_data(**CACHE)
def r_rotulos_do_arquivo():
    """Os rotulos na ordem em que **aparecem pela primeira vez** no arquivo.

    Nao por frequencia nem alfabetica: a lista de filtro segue a mesma ordem
    natural do arquivo que o resto desta tela respeita.
    """
    coluna = dados()[config.COLUNA_ROTULO]
    return [str(r) for r in dict.fromkeys(coluna.tolist())]


@st.cache_data(**CACHE)
def r_saltos():
    """O quanto o tempo anda para tras quando se le o arquivo linha a linha."""
    return saltos_no_arquivo(dados())


@st.cache_data(**CACHE)
def r_linhas_cruas(inicio: int, quantidade: int, rotulos: tuple[str, ...] = ()):
    """As linhas do arquivo como estao, todas as colunas.

    Segue o mesmo filtro do grafico, para a tabela e a serie mostrarem o mesmo
    recorte. A coluna `linha do arquivo` preserva a posicao original.
    """
    base = dados().copy()
    base.insert(0, "linha do arquivo", range(len(base)))
    if rotulos:
        base = base[base[config.COLUNA_ROTULO].isin(list(rotulos))]
    return base.iloc[inicio:inicio + quantidade].copy()


@st.cache_data(**CACHE)
def r_colunas_todas():
    """Todas as colunas na ordem do arquivo — inclusive `id` e `created_at`."""
    return list(dados().columns)


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


# --- modelo de linguagem ----------------------------------------------------
#
# Chamada ao modelo NAO entra em cache: a mesma pergunta duas vezes tem de
# custar duas vezes, senao a tela mente sobre tempo e tokens. So a deteccao de
# quem esta disponivel e cacheada, e por pouco tempo — o Ollama pode subir ou
# cair enquanto a pagina esta aberta.


@st.cache_data(show_spinner=False, ttl=30)
def r_provedores():
    """Quem da para usar agora, e por que nao os outros."""
    from mp.llm import provedores_disponiveis

    return provedores_disponiveis()


@st.cache_data(show_spinner=False, ttl=30)
def r_modelos_do_provedor(provedor: str) -> list[str]:
    """Modelos que o provedor oferece. No Ollama, os realmente baixados."""
    from mp.llm import criar

    try:
        return criar(provedor).modelos()
    except Exception:  # noqa: BLE001 — sem lista, a tela ainda funciona
        return []


def testar_llm(provedor: str, modelo: str, temperatura: float, max_tokens: int):
    from mp.llm import criar

    cliente = criar(provedor, modelo=modelo, temperatura=temperatura,
                    max_tokens=max_tokens)
    return cliente.testar()


def conversar_llm(provedor: str, modelo: str, temperatura: float,
                  max_tokens: int, pergunta: str) -> dict:
    """Uma pergunta direta ao modelo, sem guardrails e sem contexto.

    E o teste do passo 5.1 e, de quebra, o contraste: mostra o modelo respondendo
    de cabeca, que e o comportamento que o resto do sistema vai barrar.
    """
    from mp.llm import Mensagem, criar

    cliente = criar(provedor, modelo=modelo, temperatura=temperatura,
                    max_tokens=max_tokens)
    r = cliente.gerar([
        Mensagem("system", "Voce e um assistente de manutencao industrial. "
                           "Responda em portugues, de forma objetiva."),
        Mensagem("user", pergunta),
    ])
    return {
        "texto": r.texto.strip(), "provedor": r.provedor, "modelo": r.modelo,
        "tokens_entrada": r.tokens_entrada, "tokens_saida": r.tokens_saida,
        "segundos": r.segundos,
    }


# --- resposta prescritiva ---------------------------------------------------


@st.cache_data(**CACHE)
def r_catalogo():
    """O fault_map inteiro como tabela."""
    from mp.retrieval.catalog import tabela_familias

    return tabela_familias()


@st.cache_data(**CACHE)
def r_rotulos_de(familia: str) -> list[str]:
    """Os rotulos crus que caem numa familia."""
    from mp.retrieval.catalog import carregar_fault_map

    dados = carregar_fault_map()["familias"].get(familia, {})
    return list(dados.get("aliases") or [])


@st.cache_resource(show_spinner="Carregando o modelo de embeddings...")
def _embedder_aquecido():
    """Deixa o embedder ajustado antes da primeira pergunta.

    **Chamada pelo efeito colateral, nao pelo retorno.** Quem chama descarta o
    valor de proposito: `rag._embedder_pronto()` guarda o embedder ajustado num
    cache do proprio `rag` (`_PRONTOS`, indexado pelo nome do modelo), e e de la
    que a busca o pega depois. Ninguem repassa o objeto — `abrir_sessao_por_texto`
    e companhia nao recebem `embedder`, deixam o `rag` resolver, e o `rag`
    encontra o ajuste ja feito. Passar o objeto de mao em mao pela UI so
    acrescentaria um parametro a cada assinatura do caminho.

    O retorno existe porque `_embedder_pronto` devolve o embedder e nao custa
    nada repassar — quem quiser inspecionar o modelo carregado tem por onde.

    `cache_resource` e nao `cache_data`: e um objeto com pesos, nao um valor
    serializavel. Sem isso, a primeira busca da sessao levaria um minuto
    carregando o modelo enquanto a tela parece travada.

    Sao dois caches empilhados, o do Streamlit e o `_PRONTOS`, e hoje isso e so
    redundancia inofensiva — os dois vivem no mesmo processo. Vira armadilha no
    dia em que a tela de documentos (Parte 6) permitir reindexar: `rag` tem
    `limpar_cache_embedder()`, mas ele nao alcanca o cache do Streamlit. Ali sera
    preciso chamar tambem `_embedder_aquecido.clear()`.
    """
    from mp.retrieval import rag

    return rag._embedder_pronto()


def responder_prescritivo(pergunta: str, rotulo: str, usar_llm: bool,
                          config_llm: dict | None = None, k: int = 5,
                          so_prescritivos: bool = True):
    """Roda o pipeline. Com `usar_llm=False`, para no estagio 1."""
    from mp import pipeline
    from mp.llm import criar

    _embedder_aquecido()

    cliente = None
    if usar_llm:
        cfg = dict(config_llm or {})
        provedor = cfg.pop("provedor", "ollama")
        # `modelo=None` sobrescreveria o padrao de cada cliente; so passa se veio.
        opcoes = {k_: v for k_, v in cfg.items()
                  if k_ in ("modelo", "temperatura", "max_tokens") and v is not None}
        cliente = criar(provedor, **opcoes)

    return pipeline.responder(pergunta, rotulo=rotulo, cliente=cliente, k=k,
                              so_prescritivos=so_prescritivos)


# --- conversa (Parte 5) -----------------------------------------------------
#
# Sem cache: uma conversa e estado, nao consulta. O objeto `Sessao` vive no
# `st.session_state` da pagina.


@st.cache_resource(show_spinner="Montando o indice de similaridade...")
def indice_knn():
    """O historico em memoria, pronto para responder por vizinhanca.

    `cache_resource` porque e um objeto com estado ajustado (escala + arvore),
    nao um valor serializavel. Construir leva ~3 s e vale para a sessao toda.
    """
    from mp.ingestion import construir_eventos
    from mp.similarity import Indice

    leituras, _ = construir_eventos(dados())
    return Indice().construir(leituras)


def diagnosticar(evento: dict, k: int = 25):
    """O que os numeros dizem sobre este evento. Sem catalogo, sem texto."""
    return indice_knn().consultar(evento, k=k)


@st.cache_data(**CACHE)
def r_evento_de_exemplo(rotulo: str | None = None, semente: int = 0) -> dict:
    """Um evento real do historico, para colar na tela sem digitar 16 numeros.

    Com `rotulo=None` sorteia qualquer leitura — e o caso honesto: ninguem sabe
    de antemao o que o sensor vai mandar.
    """
    import numpy as np

    from mp.similarity import colunas_de_similaridade

    df = dados()
    if rotulo:
        df = df[df[config.COLUNA_ROTULO] == rotulo]
    if df.empty:
        return {}

    pos = int(np.random.default_rng(semente).integers(0, len(df)))
    linha = df.iloc[pos]

    campos = colunas_de_similaridade(dados()) + ["rpm", "temperature_c"]
    evento = {c: round(float(linha[c]), 4) for c in campos if c in linha.index}
    evento[config.COLUNA_ROTULO] = str(linha[config.COLUNA_ROTULO])
    return evento


def abrir_conversa(evento: dict | None = None, rotulo: str | None = None,
                   diagnostico=None):
    from mp.agente import abrir_sessao

    _embedder_aquecido()
    return abrir_sessao(rotulo=rotulo, evento=evento, diagnostico=diagnostico)


def _cliente(usar_llm: bool, config_llm: dict | None):
    """O cliente de LLM, ou `None`. Um lugar so — tres telas montavam isto igual."""
    if not usar_llm:
        return None

    from mp.llm import criar

    cfg = dict(config_llm or {})
    provedor = cfg.pop("provedor", "ollama")
    opcoes = {k_: v for k_, v in cfg.items()
              if k_ in ("modelo", "temperatura", "max_tokens") and v is not None}
    return criar(provedor, **opcoes)


def abrir_conversa_por_texto(descricao: str, k: int = 8):
    """O caminho principal: o tecnico descreve o problema, o sistema acha o manual.

    Pode nao achar de primeira: quando os manuais empatam, a sessao volta com a
    **lista de candidatos** e a escolha e do tecnico. Nao ha `cliente` aqui —
    nenhum modelo participa de escolher o manual.
    """
    from mp.agente import abrir_sessao_por_texto

    _embedder_aquecido()
    return abrir_sessao_por_texto(descricao, k=k)


def detalhar_sintoma(sessao, sintoma: str, k: int = 8):
    """Mais um sintoma entra e a busca recomeca com todos — sem teto de vezes."""
    from mp.agente import acrescentar_sintoma

    _embedder_aquecido()
    return acrescentar_sintoma(sessao, sintoma, k=k)


def escolher_documento(sessao, documento: str):
    """O tecnico escolhe o manual quando a evidencia nao separou os candidatos."""
    from mp.agente import fixar_documento

    return fixar_documento(sessao, documento, por_escolha=True)


def resumo_de_abertura(sessao, usar_llm: bool = False,
                       config_llm: dict | None = None, k: int = 6):
    """Problema, sintomas e correcao, assim que o manual e fixado.

    E um turno normal — passa por G4, redacao e G5 —, so que a pergunta foi do
    sistema, nao do tecnico.
    """
    from mp.agente import resumo_inicial

    return resumo_inicial(sessao, cliente=_cliente(usar_llm, config_llm), k=k)


def responder_turno(sessao, pergunta: str, usar_llm: bool,
                    config_llm: dict | None = None, k: int = 5,
                    so_prescritivos: bool = True):
    from mp.agente import responder

    return responder(sessao, pergunta, cliente=_cliente(usar_llm, config_llm),
                     k=k, so_prescritivos=so_prescritivos)


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
