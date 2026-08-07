"""De leitura crua para a linha que o classificador enxerga.

**Uma leitura nao e um exemplo.** "A vibracao neste instante foi 3,2 mm/s" nao
descreve defeito nenhum; o que descreve e como um *trecho* se comporta. Entao um
bloco de leituras vira uma linha so, resumida em 5 numeros por coluna.

Adaptado do `prep.py` do projeto irmao. O algoritmo e o mesmo — o que trocou foi
a **fonte de cada decisao**, para nao existir uma segunda resposta a perguntas
que este projeto ja responde:

    o rotulo    regras no codigo        ->  data/fault_map.yaml
    o grupo     troca do texto `fault`  ->  o evento (`fault` + `rpm`)
    as colunas  lista propria           ->  `colunas.colunas_de_medida`

O grupo importa duas vezes: ao **resumir**, porque a mediana de um bloco que
mistura tres rotacoes nao descreve nenhuma; e ao **validar**, porque grupo mal
formado deixa metade de um ensaio no treino e a outra no teste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mp import config
from mp.ingestion import construir_eventos
from mp.retrieval.catalog import familia_de, is_problem
from mp.classificacao.colunas import colunas_de_medida

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

    Devolve as leituras com `evento` (o grupo) e `familia` (a classe). So sai a
    leitura de rotulo fora do catalogo — hoje, nenhuma.

    `so_defeitos=True` tira as familias de `is_problem: false` (normal, teste,
    motor desligado). O padrao as mantem: reconhecer que a maquina esta **bem**
    e uma resposta util.
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
    return colunas_de_medida(df, incluir_regime=incluir_regime)


def nomes_das_features(colunas: list[str]) -> list[str]:
    """`['z_kurtosis__mediana', 'z_kurtosis__desvio', ...]`, na ordem da matriz.

    Existe para a matriz poder ser lida por gente: sem isso a importancia de
    feature devolvida pelo modelo seria uma lista de indices.
    """
    return [f"{coluna}__{est}" for coluna in colunas for est in ESTATISTICAS]


def resumir_bloco(bloco: pd.DataFrame | np.ndarray, colunas: list[str] | None = None) -> np.ndarray:
    """Um trecho vira um vetor de `5 x len(colunas)` numeros.

    Aceita DataFrame ou a matriz numpy ja recortada. A segunda forma existe por
    desempenho: fatiar um DataFrame milhares de vezes reconstroi indice e blocos
    a cada corte, e isso sozinho dominava o tempo de montar o conjunto.
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

    - `"janela"` — blocos de `tamanho` leituras dentro do mesmo evento, andando
      de `passo` em `passo`. Evento curto demais e **descartado inteiro**.
    - `"evento"` — o evento inteiro vira uma amostra. Nada e descartado, mas a
      validacao por grupo perde o sentido: cada grupo passa a ter uma amostra.

    O padrao e `"janela"` — e o unico modo em que a diferenca entre as duas
    estrategias de validacao aparece, e essa diferenca e o achado do projeto.
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
    """Quais eventos a janela aproveita e quais joga fora, por familia.

    O descarte e o preco escondido da janela: nao aparece na acuracia, mas
    decide **quais defeitos o modelo tem chance de aprender**. Familia cujos
    eventos sao todos curtos some do conjunto sem nenhuma metrica cair.
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
