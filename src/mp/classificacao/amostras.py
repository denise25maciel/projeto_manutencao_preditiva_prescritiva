"""De leitura crua para a linha que o classificador enxerga.

Este modulo e a adaptacao do `prep.py` do projeto de classificacao. A ideia
central vem de la e nao muda: **uma amostra nao e uma leitura**. Uma leitura
isolada de vibracao nao descreve um defeito — o que descreve e como um trecho
de leituras se comporta. Entao um bloco de leituras consecutivas vira uma linha
so, resumida em poucos numeros por coluna.

Tres coisas do original foram trocadas por peca equivalente deste projeto, e o
motivo de cada troca esta abaixo. Sao trocas de **fonte da decisao**, nao de
algoritmo: o resumo estatistico e o mesmo, o modelo e o mesmo.

**1. O rotulo vem do `fault_map.yaml`, nao de regras no codigo.**
O original normalizava o rotulo numa funcao `classe_base()` com uma lista de
typos (`desabalanceado` -> `desbalanceado`), uma lista de prefixos e uma lista
de radicais, tudo escrito em Python. Aqui isso ja existe, curado a mao e
versionado, e e o principio 1 do GUIA.md: o rotulo resolve para familia por
`catalog.familia_de`, que e um lookup exato no YAML. Duas fontes de verdade
para a mesma pergunta seria a pior das opcoes — a que o resto do sistema usa
para achar o manual teria de concordar com a que o modelo usa para aprender, e
nada garantiria isso.

Consequencia pratica: o catalogo cobre os 151 rotulos observados, entao nenhuma
leitura e descartada por rotulo desconhecido. O original descartava `teste`,
`acelerando` e `new_tes`; aqui eles viram familias com `is_problem: false` e
quem decide se entram e o parametro `so_defeitos`.

**2. O grupo e o evento (`fault` + `rpm`), nao a troca de rotulo.**
O original abria um segmento novo quando o texto de `fault` mudava ou quando
havia mais de uma hora de pausa. E exatamente a regra que este projeto testou e
**rejeitou** na Parte 1: a bancada rodava 500, 1000 e 2000 rpm em sequencia sem
trocar o nome da falha, e 136 dos 205 grupos assim formados misturavam rotacoes
— num caso a velocidade RMS ia de 3,5 a 21,1 dentro do "mesmo" grupo. Aqui o
grupo e o evento de `ingestion.construir_eventos`, que quebra tambem no `rpm`.

Isso importa **duas** vezes. Na hora de resumir, porque a mediana de um bloco
que mistura tres regimes nao descreve nenhum dos tres. E na hora de validar,
porque o grupo e o que a validacao por grupo segura fora do treino: grupo mal
formado deixa metade de um ensaio no treino e a outra metade no teste, que e o
vazamento que a validacao por grupo existe para impedir.

**3. As colunas sao as da similaridade, e o regime fica de fora por padrao.**
O original usava 18 colunas, incluindo `rpm` e `temperature_c`, e depois listou
como limitacao numero 1 que "o modelo aprendeu o ensaio, nao o defeito — a
temperatura ambiente do dia, a rotacao exata". Aqui as colunas vem de
`similarity.features.colunas_de_similaridade`, que ja separa medida fisica de
regime de operacao pelo mesmo motivo, e `incluir_regime` liga e desliga isso
para a diferenca poder ser medida em vez de suposta.

O que **nao** mudou: as cinco estatisticas por coluna, o tamanho de janela com
50% de sobreposicao, e o formato da matriz final.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mp import config
from mp.ingestion import construir_eventos
from mp.retrieval.catalog import familia_de, is_problem
from mp.similarity.features import colunas_de_similaridade

__all__ = [
    "ESTATISTICAS",
    "preparar",
    "colunas_de_entrada",
    "nomes_das_features",
    "resumir_bloco",
    "criar_amostras",
    "cobertura_dos_eventos",
    "matriz_legivel",
    "tabela_de_estatisticas",
]


# As cinco estatisticas com que um bloco de leituras vira numero, por coluna.
#
# Elas nao sao intercambiaveis — cada uma responde uma pergunta diferente sobre
# o trecho, e o conjunto e escolhido para descrever forma, e nao so nivel:
#
#   mediana     em que patamar o trecho ficou (robusta ao pico isolado, que em
#               kurtosis e crest factor e frequente)
#   desvio      o quanto oscilou em torno do patamar
#   inclinacao  se estava subindo ou caindo ao longo do trecho — e a unica que
#               enxerga ORDEM; sem ela, embaralhar as leituras nao mudaria nada
#   amplitude   max - min, o extremo do trecho
#   p90_p10     a mesma ideia da amplitude, sem os 10% das pontas: separa
#               "oscilou o tempo todo" de "teve um pico e voltou"
ESTATISTICAS = ("mediana", "desvio", "inclinacao", "amplitude", "p90_p10")

COLUNA_CLASSE = "familia"
COLUNA_GRUPO = "evento"


def preparar(df: pd.DataFrame, so_defeitos: bool = False) -> pd.DataFrame:
    """Ordena no tempo, marca o evento de cada leitura e resolve a familia.

    Devolve as leituras com duas colunas a mais: `evento` (o grupo) e `familia`
    (a classe a prever). Nenhuma leitura e removida, exceto as de rotulo fora do
    catalogo — que hoje sao zero, porque o `fault_map.yaml` cobre os 151 rotulos
    observados.

    Com `so_defeitos=True` ficam de fora as familias de `is_problem: false`
    (normal, teste, acelerando, motor desligado). E o recorte que responde "que
    defeito e este?"; o padrao responde "em que estado a maquina esta?", que
    inclui reconhecer que ela esta bem.
    """
    leituras, _ = construir_eventos(df)

    rotulos = leituras[config.COLUNA_ROTULO].astype("string")
    # Um map por rotulo distinto, nao por linha: sao 151 chaves para 166 mil
    # linhas, e `familia_de` faz lookup em dicionario cacheado.
    familias = {r: familia_de(r) for r in rotulos.dropna().unique()}
    leituras[COLUNA_CLASSE] = rotulos.map(familias).astype("string")

    leituras = leituras[leituras[COLUNA_CLASSE].notna()]

    if so_defeitos:
        defeito = {f: bool(is_problem(f)) for f in leituras[COLUNA_CLASSE].unique()}
        leituras = leituras[leituras[COLUNA_CLASSE].map(defeito)]

    return leituras.reset_index(drop=True)


def colunas_de_entrada(df: pd.DataFrame, incluir_regime: bool = False) -> list[str]:
    """As colunas de medida que entram no resumo, na ordem.

    Mesma funcao que o motor de similaridade usa. Se um dia o descarte de uma
    coluna mudar la, muda aqui junto — e o principio 5 do GUIA.md aplicado entre
    dois consumidores da mesma decisao.
    """
    return colunas_de_similaridade(df, incluir_regime=incluir_regime)


def nomes_das_features(colunas: list[str]) -> list[str]:
    """`['z_kurtosis__mediana', 'z_kurtosis__desvio', ...]`, na ordem da matriz.

    Existe para a matriz poder ser lida por gente: sem isso a importancia de
    feature devolvida pelo modelo seria uma lista de indices.
    """
    return [f"{coluna}__{est}" for coluna in colunas for est in ESTATISTICAS]


def resumir_bloco(bloco: pd.DataFrame | np.ndarray, colunas: list[str] | None = None) -> np.ndarray:
    """Um trecho de leituras vira um vetor de `5 x len(colunas)` numeros.

    Aceita tambem a matriz numpy ja recortada. Nao e microotimizacao: recortar
    um DataFrame milhares de vezes reconstroi indice e blocos a cada fatia, e
    isso sozinho respondia por quase todo o tempo de montar o conjunto. Quem
    monta em lote converte o evento uma vez e fatia o array.
    """
    if isinstance(bloco, pd.DataFrame):
        valores = bloco[colunas].to_numpy(dtype="float64")
    else:
        valores = np.asarray(bloco, dtype="float64")
    n = len(valores)

    # A inclinacao e o coeficiente de uma reta ajustada por minimos quadrados
    # sobre o indice da leitura. Centrar o eixo em zero (t soma zero) faz o
    # coeficiente sair como um produto escalar, sem precisar montar sistema.
    # Como t soma zero, tirar a mediana de x antes nao muda o resultado — mas
    # deixa o nulo virar "leitura na mediana", que e o mesmo que dizer "sem
    # informacao", em vez de puxar a reta para baixo como um zero faria.
    if n <= 1:
        t = np.zeros(n)
        denominador = 1.0
    else:
        t = np.arange(n) - (n - 1) / 2
        denominador = float(t @ t)

    # As cinco estatisticas saem de uma vez para TODAS as colunas, em vez de um
    # laco por coluna. Sao 5 chamadas de numpy em vez de 5 x n_colunas, e a
    # diferenca nao e cosmetica: o conjunto de treino tem milhares de janelas, e
    # o laco respondia por quase todo o tempo de monta-lo.
    with np.errstate(invalid="ignore"):
        mediana = np.nanmedian(valores, axis=0)
        desvio = np.nanstd(valores, axis=0)
        maximo = np.nanmax(valores, axis=0)
        minimo = np.nanmin(valores, axis=0)
        p90, p10 = np.nanpercentile(valores, [90, 10], axis=0)
        # Nulo vira "leitura na mediana" — ver o comentario acima.
        centrado = np.nan_to_num(valores - mediana)
        inclinacao = (t @ centrado) / denominador

    estatisticas = np.vstack(
        [mediana, desvio, inclinacao, maximo - minimo, p90 - p10]
    )

    # Coluna toda nula vira zero em tudo: nao inventa sinal e nao entrega NaN a
    # floresta, que nao aceita NaN.
    estatisticas = np.nan_to_num(estatisticas, nan=0.0, posinf=0.0, neginf=0.0)

    # `.T` poe cada coluna de origem numa linha, e `.ravel()` achata na ordem
    # que `nomes_das_features` promete: as 5 estatisticas da coluna 0, depois as
    # 5 da coluna 1, e assim por diante.
    return estatisticas.T.ravel()


def tabela_de_estatisticas(
    bloco: pd.DataFrame, colunas: list[str] | None = None
) -> pd.DataFrame:
    """O vetor de features de um bloco, remontado como coluna x estatistica.

    `resumir_bloco` devolve os numeros achatados numa linha so, que e o formato
    que o modelo consome e o pior formato possivel para alguem ler. Aqui os
    mesmos numeros voltam a ter duas dimensoes: uma linha por coluna de medida,
    uma coluna por estatistica.

    E deliberadamente uma **remontagem**, e nao um segundo calculo: se esta
    funcao recalculasse as estatisticas por conta propria, a tela poderia
    mostrar um numero e o modelo receber outro, e ninguem notaria. O que aparece
    aqui e literalmente o que entra na floresta.
    """
    colunas = colunas or colunas_de_entrada(bloco)
    vetor = resumir_bloco(bloco, colunas)
    return pd.DataFrame(
        vetor.reshape(len(colunas), len(ESTATISTICAS)),
        index=pd.Index(colunas, name="coluna"),
        columns=list(ESTATISTICAS),
    )


def matriz_legivel(amostras: pd.DataFrame, nomes: list[str] | None = None) -> pd.DataFrame:
    """As amostras como uma tabela plana: `familia`, `evento` e uma coluna por feature.

    E o formato em que a base pode ser olhada, baixada e conferida — a coluna
    `features` que `criar_amostras` devolve guarda um array por linha, que serve
    ao `sklearn` e nao serve a ninguem mais.
    """
    if amostras.empty:
        return pd.DataFrame()

    nomes = nomes or amostras.attrs.get("nomes_features")
    X = np.vstack(amostras["features"].to_list())
    if not nomes or len(nomes) != X.shape[1]:
        nomes = [f"f{i}" for i in range(X.shape[1])]

    tabela = pd.DataFrame(X, columns=nomes)
    tabela.insert(0, COLUNA_CLASSE, amostras[COLUNA_CLASSE].to_numpy())
    tabela.insert(1, COLUNA_GRUPO, amostras[COLUNA_GRUPO].to_numpy())
    return tabela


def criar_amostras(
    leituras: pd.DataFrame,
    modo: str = "janela",
    tamanho: int | None = None,
    passo: int | None = None,
    incluir_regime: bool = False,
) -> pd.DataFrame:
    """Monta a tabela de amostras a partir das leituras ja preparadas.

    Duas formas de recortar, e a escolha muda o que o numero final significa:

    - `"janela"` — blocos de `tamanho` leituras consecutivas dentro do mesmo
      evento, andando de `passo` em `passo`. Da muitas amostras, e o evento
      curto demais para caber uma janela e **descartado inteiro**.
    - `"evento"` — o evento inteiro vira uma amostra. Nada e descartado, mas as
      amostras sao poucas e a validacao por grupo perde o sentido, porque cada
      grupo passa a ter uma amostra so.

    O padrao e `"janela"`: e o unico modo em que a diferenca entre as duas
    estrategias de validacao aparece, e essa diferenca e o achado do projeto.

    Devolve uma linha por amostra, com `features` (o vetor), `familia` (a
    classe), `evento` (o grupo) e `n_leituras`.
    """
    modo = str(modo).lower()
    if modo not in {"janela", "evento"}:
        raise ValueError(f"modo deve ser 'janela' ou 'evento'; veio {modo!r}")

    tamanho = tamanho or config.CLF_JANELA_TAMANHO
    passo = passo or max(1, tamanho // 2)
    colunas = colunas_de_entrada(leituras, incluir_regime=incluir_regime)

    amostras: list[dict] = []
    for evento, grupo in leituras.groupby(COLUNA_GRUPO, sort=True):
        if grupo.empty:
            continue

        classe = grupo[COLUNA_CLASSE].iloc[0]
        # Uma conversao por evento, nao uma por janela.
        valores = grupo[colunas].to_numpy(dtype="float64")

        if modo == "evento":
            amostras.append(
                {
                    "features": resumir_bloco(valores),
                    COLUNA_CLASSE: classe,
                    COLUNA_GRUPO: int(evento),
                    "n_leituras": len(grupo),
                }
            )
            continue

        if len(grupo) < tamanho:
            continue

        for inicio in range(0, len(grupo) - tamanho + 1, passo):
            amostras.append(
                {
                    "features": resumir_bloco(valores[inicio : inicio + tamanho]),
                    COLUNA_CLASSE: classe,
                    COLUNA_GRUPO: int(evento),
                    "n_leituras": tamanho,
                }
            )

    if not amostras:
        return pd.DataFrame(
            columns=["features", COLUNA_CLASSE, COLUNA_GRUPO, "n_leituras"]
        )

    tabela = pd.DataFrame(amostras)
    tabela.attrs["colunas"] = colunas
    tabela.attrs["nomes_features"] = nomes_das_features(colunas)
    return tabela


def matriz(amostras: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(X, y, grupos)` — o formato que o `sklearn` espera."""
    X = np.vstack(amostras["features"].to_list())
    y = amostras[COLUNA_CLASSE].to_numpy(dtype=object).astype(str)
    grupos = amostras[COLUNA_GRUPO].to_numpy()
    return X, y, grupos


def cobertura_dos_eventos(
    leituras: pd.DataFrame, tamanho: int | None = None
) -> pd.DataFrame:
    """Quais eventos a janela aproveita e quais ela joga fora, por familia.

    O descarte e o preco escondido da janela: ele nao aparece na acuracia, mas
    decide QUAIS defeitos o modelo tem chance de aprender. Uma familia cujos
    eventos sao todos curtos some do conjunto de treino sem aviso, e o modelo
    passa a ser incapaz de nomea-la — sem que nenhuma metrica caia por isso.
    """
    tamanho = tamanho or config.CLF_JANELA_TAMANHO

    por_evento = (
        leituras.groupby([COLUNA_CLASSE, COLUNA_GRUPO], observed=True)
        .size()
        .rename("n_leituras")
        .reset_index()
    )
    por_evento["cabe"] = por_evento["n_leituras"] >= tamanho

    resumo = (
        por_evento.groupby(COLUNA_CLASSE, observed=True)
        .agg(
            eventos=("n_leituras", "size"),
            eventos_aproveitados=("cabe", "sum"),
            leituras=("n_leituras", "sum"),
            mediana_leituras=("n_leituras", "median"),
        )
        .reset_index()
    )
    resumo["eventos_descartados"] = resumo["eventos"] - resumo["eventos_aproveitados"]
    resumo["pct_descartado"] = 100 * resumo["eventos_descartados"] / resumo["eventos"]
    return resumo.sort_values("pct_descartado", ascending=False).reset_index(drop=True)
