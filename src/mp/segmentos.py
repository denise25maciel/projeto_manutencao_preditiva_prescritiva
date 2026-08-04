"""Primitiva unica de segmentacao: quebrar linhas consecutivas em grupos.

Existe porque tres lugares do projeto precisam da mesma operacao — "percorra as
linhas em ordem e comece um grupo novo quando tal coisa acontecer":

- `analysis.profiling.ordenar_por_tempo` — separa campanhas de coleta
- `analysis.profiling.analise_intervalos` — conta episodios para varios cortes
- `ingestion.sensors.construir_episodios` — monta os eventos da Parte 1

Sem este modulo, a mesma logica ficaria escrita tres vezes, e uma correcao num
lugar nao chegaria nos outros. Aqui ela e escrita uma vez.

O modulo nao conhece o dominio: nao sabe o que e falha, sensor ou episodio.
Recebe colunas, devolve numeros de grupo. Quem da significado e quem chama.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "mudou_valor",
    "passou_intervalo",
    "numerar_grupos",
    "resumir_grupos",
    "maior_buraco_interno",
]


def mudou_valor(serie: pd.Series) -> np.ndarray:
    """`True` onde o valor difere da linha anterior. A primeira linha e sempre `True`.

    Sai para numpy antes de devolver: no pandas 3 as colunas de texto vem do
    Arrow, e comparar com `.shift()` produz nulo na primeira linha, que depois
    trava tanto o operador `|` quanto o `cumsum`.
    """
    return (serie != serie.shift()).fillna(True).to_numpy(dtype=bool)


def passou_intervalo(tempos: pd.Series, limite_s: float | None) -> np.ndarray:
    """`True` onde o intervalo desde a linha anterior passa de `limite_s`.

    `limite_s=None` desliga a regra — devolve tudo `False`. Serve para manter a
    opcao disponivel sem que ela atue.
    """
    if limite_s is None:
        return np.zeros(len(tempos), dtype=bool)
    delta = tempos.diff().dt.total_seconds()
    return (delta > limite_s).fillna(False).to_numpy(dtype=bool)


def numerar_grupos(*mascaras: np.ndarray) -> np.ndarray:
    """Combina as mascaras de corte e devolve o numero do grupo de cada linha.

    Qualquer mascara `True` inicia um grupo novo. A conta e uma soma cumulativa,
    feita em numpy porque o booleano do Arrow nao a suporta.
    """
    if not mascaras:
        raise ValueError("Informe ao menos uma mascara de corte.")

    combinada = np.zeros(len(mascaras[0]), dtype=bool)
    for m in mascaras:
        combinada |= np.asarray(m, dtype=bool)

    if len(combinada):
        combinada[0] = True  # a primeira linha sempre abre o primeiro grupo
    return np.cumsum(combinada)


def resumir_grupos(
    df: pd.DataFrame,
    grupos: np.ndarray,
    coluna_tempo: str,
    primeiro_de: list[str] | None = None,
) -> pd.DataFrame:
    """Uma linha por grupo, com tamanho, inicio, fim e duracao.

    `primeiro_de` lista colunas cujo primeiro valor do grupo entra no resumo —
    tipicamente o rotulo, que e constante dentro do grupo por construcao.
    """
    primeiro_de = primeiro_de or []

    trabalho = df.assign(_grupo=grupos)
    agregacoes = {
        "n_leituras": (coluna_tempo, "size"),
        "inicio": (coluna_tempo, "min"),
        "fim": (coluna_tempo, "max"),
    }
    for c in primeiro_de:
        agregacoes[c] = (c, "first")

    resumo = trabalho.groupby("_grupo", sort=True).agg(**agregacoes).reset_index()
    resumo["duracao_s"] = (resumo["fim"] - resumo["inicio"]).dt.total_seconds()
    resumo["duracao_min"] = (resumo["duracao_s"] / 60).round(2)

    ordem = ["_grupo", *primeiro_de, "n_leituras", "inicio", "fim",
             "duracao_s", "duracao_min"]
    return resumo[ordem]


def maior_buraco_interno(
    df: pd.DataFrame, grupos: np.ndarray, coluna_tempo: str
) -> pd.Series:
    """Maior intervalo entre duas linhas DENTRO de cada grupo, em segundos.

    Serve de diagnostico: um grupo com buraco grande por dentro provavelmente
    deveria ter sido dois. O intervalo que separa um grupo do anterior nao conta
    — so os de dentro.
    """
    delta = df[coluna_tempo].diff().dt.total_seconds()
    trabalho = pd.DataFrame({"_grupo": grupos, "_delta": delta.to_numpy()})
    # A primeira linha de cada grupo carrega o intervalo em relacao ao grupo
    # anterior; descartamos para medir so o que acontece por dentro.
    interno = trabalho[trabalho.groupby("_grupo").cumcount() > 0]
    return interno.groupby("_grupo")["_delta"].max().fillna(0.0)
