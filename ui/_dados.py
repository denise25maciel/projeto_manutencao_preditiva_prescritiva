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

import re
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
from mp.llm.prompts import SISTEMA as SISTEMA_PADRAO  # noqa: E402
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


def abrir_conversa_por_texto(descricao: str, k: int = 8, usar_llm: bool = False,
                             config_llm: dict | None = None):
    """O caminho principal: o tecnico descreve o problema, o sistema acha o manual.

    Pode nao achar de primeira: quando os manuais empatam, a sessao volta com a
    **lista de candidatos** e a escolha e do tecnico.

    O modelo entra aqui numa tarefa so — **separar a fala em sintomas** — e nao
    participa de escolher o manual. Sem ele, a descricao entra inteira e a busca
    funciona igual, so que pior: ver `agente.separar_sintomas`.
    """
    from mp.agente import abrir_sessao_por_texto

    _embedder_aquecido()
    return abrir_sessao_por_texto(descricao, k=k,
                                  cliente=_cliente(usar_llm, config_llm))


def detalhar_sintoma(sessao, sintoma: str, k: int = 8, usar_llm: bool = False,
                     config_llm: dict | None = None):
    """Mais um sintoma entra e a busca recomeca com todos — sem teto de vezes."""
    from mp.agente import acrescentar_sintoma

    _embedder_aquecido()
    return acrescentar_sintoma(sessao, sintoma, k=k,
                               cliente=_cliente(usar_llm, config_llm))


def escolher_documento(sessao, documento: str):
    """O tecnico escolhe o manual quando a evidencia nao separou os candidatos."""
    from mp.agente import fixar_documento

    return fixar_documento(sessao, documento, por_escolha=True)


def _sistema(config_llm: dict | None) -> str | None:
    """As regras do prompt escolhidas na tela, ou `None` para as versionadas.

    Texto vazio conta como `None`: apagar o campo inteiro restaura o padrao em
    vez de mandar uma `SystemMessage` em branco.
    """
    texto = (config_llm or {}).get("sistema") or ""
    return texto.strip() or None


def classificar_na_conversa(sessao, diagnostico, usar_llm: bool = False,
                            config_llm: dict | None = None):
    """O que o kNN apurou entra na conversa como fala, antes do manual.

    **Nao passa `sistema`.** As regras editaveis da tela sao as da resposta
    prescritiva; aplicá-las aqui trocaria as regras da classificacao pelas de
    outra tarefa, e o prompt de classificacao tem as suas — nao inventar numero,
    nao recomendar nada, destacar divergencia.
    """
    from mp.agente import turno_de_classificacao

    return turno_de_classificacao(sessao, diagnostico,
                                  cliente=_cliente(usar_llm, config_llm))


def resumo_de_abertura(sessao, usar_llm: bool = False,
                       config_llm: dict | None = None, k: int = 6):
    """Problema, sintomas e correcao, assim que o manual e fixado.

    E um turno normal — passa por G4, redacao e G5 —, so que a pergunta foi do
    sistema, nao do tecnico.
    """
    from mp.agente import resumo_inicial

    return resumo_inicial(sessao, cliente=_cliente(usar_llm, config_llm), k=k,
                          sistema=_sistema(config_llm))


def responder_turno(sessao, pergunta: str, usar_llm: bool,
                    config_llm: dict | None = None, k: int = 5,
                    so_prescritivos: bool = True):
    from mp.agente import responder

    return responder(sessao, pergunta, cliente=_cliente(usar_llm, config_llm),
                     k=k, so_prescritivos=so_prescritivos,
                     sistema=_sistema(config_llm))


# --- utilitarios de pagina --------------------------------------------------

# `## 3. Componentes do Rolamento` no inicio de uma linha. O `[ \t]*` aceita o
# titulo indentado, e o `#*$` come o fecho opcional do estilo `## Titulo ##`.
_TITULO_MD = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)

# `<!-- campo: objetivo -->`, o marcador que a conversao do PDF deixou em cada
# secao. Serve ao codigo, nao a quem le.
_COMENTARIO_HTML = re.compile(r"<!--.*?-->", re.DOTALL)


def para_exibir(texto: str, nivel: int = 5) -> str:
    """Rebaixa os titulos do texto e tira os marcadores de conversao.

    **Por que existe.** Os 168 trechos vieram dos PDFs como Markdown, e todos
    comecam com `## N. Titulo`. Esse texto vai inteiro para o prompt, entao o
    modelo aprende o estilo pelo exemplo e responde com `##` tambem — que dentro
    de um balao de conversa sai do tamanho de um titulo de pagina e grita mais
    alto que a propria resposta.

    Rebaixar para `#####` mantem a hierarquia (continua sendo titulo, continua
    tendo espaco antes) e devolve o texto ao tamanho de leitura.

    **E so aparencia, e so aqui.** O que foi gravado no turno e o que foi enviado
    ao modelo nao mudam — a aba de auditoria mostra o texto cru, com `st.code`,
    justamente para provar isso. Consertar no prompt seria pedir; consertar aqui
    e garantir.
    """
    if not texto:
        return ""
    limpo = _COMENTARIO_HTML.sub("", texto)
    return _TITULO_MD.sub(lambda m: f"{'#' * nivel} {m.group(1)}", limpo).strip()


def em_uma_linha(texto: str, maximo: int = 320) -> str:
    """O mesmo texto achatado numa linha so, sem marca de titulo, truncado.

    **Rebaixar nao serve aqui, remover sim.** `para_exibir` mantem o `#` porque
    no balao da conversa o titulo ainda e um titulo — tem espaco antes e depois,
    e a hierarquia ajuda a ler. Numa previa de uma linha nao ha hierarquia: o
    `##` que sobra vira o comeco da string, e o Markdown o trata como cabecalho
    de secao — no `st.caption` isso saia do tamanho de titulo de pagina dentro
    de uma legenda.

    Os 168 trechos comecam todos com `## N. Titulo`, entao **toda** previa caia
    nesse caso, nao um caso de borda.

    Corta na ultima palavra inteira: cortar no meio da palavra faz o leitor
    tentar completa-la.
    """
    if not texto:
        return ""
    limpo = " ".join(_COMENTARIO_HTML.sub("", texto).split())
    # `^#{1,6}\s*` nao basta: depois de achatar, o titulo e o corpo viraram uma
    # linha so, e o `_TITULO_MD` (ancorado em linha) nao casa mais.
    limpo = re.sub(r"#{1,6}\s+", "", limpo).strip()

    if len(limpo) <= maximo:
        return limpo
    return limpo[:maximo].rsplit(" ", 1)[0] + "..."


# --- classificacao supervisionada -------------------------------------------
#
# Import adiado dentro de cada funcao: `mp.classificacao` puxa o `sklearn`, e
# quem abre a tela de dados nao deve pagar esse carregamento.
#
# As funcoes caras aqui nao sao "caras" no sentido das outras. `r_clf_validacao`
# treina 10 florestas de 400 arvores; `r_clf_experimento_janela` treina 10 por
# tamanho testado. Por isso elas ficam atras de um botao na tela, e nunca sao
# chamadas na abertura da pagina.


@st.cache_data(**CACHE)
def r_clf_leituras():
    """Leituras com `evento` e `familia` — o insumo de tudo nesta secao."""
    from mp.classificacao import preparar

    return preparar(dados())


@st.cache_data(**CACHE)
def r_clf_amostras(modo: str = "janela", tamanho: int | None = None,
                   incluir_regime: bool = False):
    from mp.classificacao import criar_amostras

    return criar_amostras(r_clf_leituras(), modo=modo, tamanho=tamanho,
                          incluir_regime=incluir_regime)


@st.cache_data(**CACHE)
def r_clf_cobertura(tamanho: int | None = None):
    from mp.classificacao import cobertura_dos_eventos

    return cobertura_dos_eventos(r_clf_leituras(), tamanho=tamanho)


@st.cache_data(**CACHE)
def r_clf_colunas(incluir_regime: bool = False) -> list[str]:
    from mp.classificacao import colunas_de_entrada

    return colunas_de_entrada(r_clf_leituras(), incluir_regime=incluir_regime)


@st.cache_data(**CACHE)
def r_clf_nomes_features(incluir_regime: bool = False) -> list[str]:
    """Nome de cada coluna da matriz, na ordem.

    `criar_amostras` ja devolve isto em `DataFrame.attrs`, mas `attrs` nao
    atravessa o cache do Streamlit de forma garantida — ele serializa o
    DataFrame, e metadado de dicionario e o primeiro a se perder. Recalcular e
    barato e nao depende disso.
    """
    from mp.classificacao import nomes_das_features

    return nomes_das_features(r_clf_colunas(incluir_regime))


@st.cache_data(**CACHE)
def r_clf_csv(tamanho: int | None = None, incluir_regime: bool = False) -> str:
    """A matriz inteira em CSV, para o botao de download.

    Cacheado por parametro simples, e nao pelo DataFrame: o `download_button`
    precisa dos bytes prontos a cada rerun, e montar 6 mil linhas x 80 colunas
    toda vez travaria a tela.
    """
    import numpy as np
    import pandas as pd

    amostras = r_clf_amostras("janela", tamanho, incluir_regime)
    tabela = pd.DataFrame(
        np.vstack(amostras["features"].to_list()),
        columns=r_clf_nomes_features(incluir_regime),
    )
    tabela.insert(0, "familia", amostras["familia"].to_numpy())
    tabela.insert(1, "evento", amostras["evento"].to_numpy())
    return tabela.to_csv(index=False)


@st.cache_data(**CACHE)
def r_clf_divisao(tamanho: int | None = None, incluir_regime: bool = False,
                  por_evento: bool = True, fracao_teste: float = 0.2):
    """Um corte unico em treino e teste, com as duas bases de pe."""
    from mp.classificacao import dividir_treino_teste

    return dividir_treino_teste(
        r_clf_amostras("janela", tamanho, incluir_regime),
        fracao_teste=fracao_teste, por_evento=por_evento,
    )


@st.cache_data(**CACHE)
def r_clf_leituras_do_evento(evento: int, limite: int | None = None):
    """As leituras cruas de um evento — o lado esquerdo da transformacao."""
    leituras = r_clf_leituras()
    bloco = leituras[leituras["evento"] == evento]
    return bloco.head(limite) if limite else bloco


@st.cache_data(**CACHE)
def r_clf_estatisticas(evento: int, tamanho: int | None = None,
                       incluir_regime: bool = False):
    """As 5 estatisticas por coluna da primeira janela de um evento.

    O lado direito da transformacao. Sai de `resumir_bloco`, a mesma funcao que
    monta o conjunto de treino — nao e um calculo de vitrine.
    """
    from mp.classificacao import tabela_de_estatisticas

    tam = tamanho or config.CLF_JANELA_TAMANHO
    bloco = r_clf_leituras_do_evento(evento, limite=tam)
    return tabela_de_estatisticas(bloco, r_clf_colunas(incluir_regime))


@st.cache_data(**CACHE)
def r_clf_csv_base(qual: str, tamanho: int | None = None,
                   incluir_regime: bool = False, por_evento: bool = True) -> str:
    """Treino ou teste em CSV, para o botao de baixar."""
    return r_clf_divisao(tamanho, incluir_regime, por_evento)[qual].to_csv(index=False)


@st.cache_data(show_spinner="Treinando e validando 10 florestas...")
def r_clf_validacao(modo: str = "janela", tamanho: int | None = None,
                    incluir_regime: bool = False):
    from mp.classificacao import validar

    return validar(r_clf_amostras(modo, tamanho, incluir_regime))


@st.cache_data(**CACHE)
def r_clf_confusao(modo: str, tamanho: int | None, incluir_regime: bool,
                   estrategia: str, normalizar: bool = True):
    from mp.classificacao import matriz_de_confusao

    resultado = r_clf_validacao(modo, tamanho, incluir_regime)
    return matriz_de_confusao(resultado["estrategias"][estrategia], normalizar)


@st.cache_data(**CACHE)
def r_clf_por_familia(modo: str, tamanho: int | None, incluir_regime: bool,
                      estrategia: str):
    from mp.classificacao import acerto_por_familia

    resultado = r_clf_validacao(modo, tamanho, incluir_regime)
    return acerto_por_familia(resultado["estrategias"][estrategia])


@st.cache_data(show_spinner="Comparando os tamanhos de janela...")
def r_clf_experimento_janela(tamanhos: tuple[int, ...] | None = None):
    from mp.classificacao import experimento_janela

    return experimento_janela(r_clf_leituras(), tamanhos=tamanhos)


@st.cache_data(show_spinner="Medindo o efeito do regime de operacao...")
def r_clf_experimento_regime(tamanho: int | None = None):
    from mp.classificacao import experimento_regime

    return experimento_regime(r_clf_leituras(), tamanho=tamanho)


@st.cache_resource(show_spinner="Treinando a floresta...")
def clf_modelo(tamanho: int | None = None, incluir_regime: bool = False):
    """A floresta ajustada. `cache_resource`: e objeto, nao valor."""
    from mp.classificacao import Classificador

    return Classificador(tamanho=tamanho, incluir_regime=incluir_regime).treinar(dados())


@st.cache_data(show_spinner="Treinando sem este evento...")
def r_clf_previsao_honesta(evento: int, tamanho: int | None = None,
                           incluir_regime: bool = False):
    """Palpite para um evento que ficou de fora do treino."""
    from mp.classificacao import prever_evento_segurado

    return prever_evento_segurado(r_clf_leituras(), evento, tamanho=tamanho,
                                  incluir_regime=incluir_regime)


@st.cache_data(**CACHE)
def r_clf_eventos_consultaveis(tamanho: int | None = None):
    """Eventos que geram pelo menos uma janela — os que da para consultar."""
    leituras = r_clf_leituras()
    tam = tamanho or config.CLF_JANELA_TAMANHO
    por_evento = (
        leituras.groupby(["evento", "familia", config.COLUNA_ROTULO], observed=True)
        .size()
        .rename("n_leituras")
        .reset_index()
    )
    return por_evento[por_evento["n_leituras"] >= tam].reset_index(drop=True)


def configurar_pagina(titulo: str, icone: str = "🔧") -> None:
    st.set_page_config(page_title=f"{titulo} — Manutencao Prescritiva",
                       page_icon=icone, layout="wide")


# Dentro do balao de conversa, nenhum titulo passa do tamanho do texto corrido.
#
# **Por que existe, se `para_exibir` ja rebaixa.** Porque as duas travas pegam
# coisas diferentes, e a primeira depende de alguem lembrar de chama-la. O texto
# que chega ao balao vem de tres origens — o manual, a redacao do modelo e o
# historico — e qualquer uma pode trazer `#`. Sanear na origem e o conserto
# certo; isto e o teto que vale mesmo quando um caminho novo esquecer o saneamento.
#
# Escopo no `stChatMessage` de proposito: `st.header` e `st.title` sao NOSSOS e
# continuam do tamanho que devem ter. So o conteudo de terceiros e limitado.
#
# Se o `data-testid` mudar numa versao futura do Streamlit, a regra deixa de
# casar e a tela volta ao comportamento de hoje — degrada para o que ja
# funciona, nao para algo pior.
_CSS_CHAT = """
<style>
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] h5,
[data-testid="stChatMessage"] h6 {
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.4;
    margin: 0.8rem 0 0.3rem;
    padding: 0;
    letter-spacing: 0;
}
/* O primeiro titulo do balao nao precisa de respiro acima: o balao ja o da. */
[data-testid="stChatMessage"] h1:first-child,
[data-testid="stChatMessage"] h2:first-child,
[data-testid="stChatMessage"] h3:first-child,
[data-testid="stChatMessage"] h4:first-child,
[data-testid="stChatMessage"] h5:first-child,
[data-testid="stChatMessage"] h6:first-child {
    margin-top: 0;
}
</style>
"""


def estilo_do_chat() -> None:
    """Limita o tamanho dos titulos dentro dos baloes de conversa.

    Chamar uma vez por pagina que tenha chat, logo apos `configurar_pagina`.
    """
    st.markdown(_CSS_CHAT, unsafe_allow_html=True)


def aviso_csv_ausente(erro: Exception) -> None:
    """Mensagem util quando o CSV bruto nao esta no lugar."""
    st.error("Nao encontrei o `banner.csv`.")
    st.code(str(erro))
    st.info(
        "O CSV fica fora do git (dado da empresa). Coloque-o em `data/raw/banner.csv` "
        "ou defina a variavel de ambiente `MP_CSV` com o caminho completo."
    )
    st.stop()
