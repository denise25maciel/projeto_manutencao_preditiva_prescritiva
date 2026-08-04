"""Leitura e tipagem do CSV bruto.

Unico ponto do projeto que le `banner.csv`. Nao limpa nada: se o dado veio
sujo, ele chega sujo aqui de proposito — quem descreve a sujeira e o
`profiling` / `quality`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mp import config


def carregar(caminho: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Le o CSV bruto e devolve o DataFrame com os tipos corretos.

    Parameters
    ----------
    caminho : caminho do CSV. Se None, resolve por `config.caminho_csv()`.
    nrows   : le apenas as N primeiras linhas (util para teste rapido).

    Tratamentos aplicados — todos reversiveis e sem perda:
      1. `created_at` vira datetime com timezone (UTC). O texto traz offset
         `+00:00`; sem o parse ele ficaria como string e qualquer conta de
         intervalo seria impossivel.
      2. `fault` vira string com espacos aparados. Nao normalizamos caixa nem
         corrigimos typo aqui — a lista de rotulos crus e um resultado da
         Parte 0, e mascarar os erros de digitacao esconderia o achado.

    O que NAO fazemos: ordenar, deduplicar, descartar coluna ou tratar outlier.
    """
    caminho = Path(caminho) if caminho is not None else config.caminho_csv()

    df = pd.read_csv(caminho, nrows=nrows)

    # format="mixed" porque a fracao de segundo nao tem largura fixa no arquivo.
    if config.COLUNA_TEMPO in df.columns:
        df[config.COLUNA_TEMPO] = pd.to_datetime(
            df[config.COLUNA_TEMPO], format="mixed", utc=True
        )

    if config.COLUNA_ROTULO in df.columns:
        df[config.COLUNA_ROTULO] = df[config.COLUNA_ROTULO].astype("string").str.strip()

    return df


def colunas_numericas(df: pd.DataFrame, incluir_vazamento: bool = False) -> list[str]:
    """Colunas numericas do DataFrame.

    Por padrao exclui `id` e `created_at` (config.COLUNAS_VAZAMENTO): sao
    identificadores temporais, nao medidas fisicas. Entrariam no kNN como
    atalho e o modelo acertaria por proximidade de indice.
    """
    cols = df.select_dtypes(include="number").columns.tolist()
    if not incluir_vazamento:
        cols = [c for c in cols if c not in config.COLUNAS_VAZAMENTO]
    return cols
