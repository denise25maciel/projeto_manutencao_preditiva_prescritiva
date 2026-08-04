"""Transformacao das leituras de sensor em eventos.

Divisao de responsabilidade dentro do projeto:

- `analysis/`  — **descreve** o dado bruto. Nao altera nada.
- `ingestion/` — **transforma** o dado bruto no que vai para o banco.
- `segmentos`  — a primitiva de agrupar linhas consecutivas, usada pelos dois.

Este arquivo e o lado do sensor; `documents.py` e o lado dos PDFs.

O que e um evento
-----------------
Uma vez em que a maquina foi medida com o mesmo defeito, sem interrupcao.

Contar linhas engana: `rolamento_inner` tem 13 mil linhas, mas foram 12
medicoes de meia hora. A pergunta "quantas vezes isso aconteceu?" so tem
resposta depois de agrupar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mp import config, segmentos

# Regra atual: um evento novo comeca quando o rotulo muda. Nada mais.
#
# `limite_intervalo_s` existe e vem desligado. A analise da tela "Qualidade dos
# Dados" mostra que um corte de 10 s separaria ensaios que hoje ficam juntos —
# mas a decisao foi comecar so pelo rotulo. O parametro fica pronto para quando
# a decisao mudar, e `diagnostico_eventos` reporta o custo de mante-lo desligado.
LIMITE_INTERVALO_PADRAO: float | None = None


def construir_eventos(
    df: pd.DataFrame,
    limite_intervalo_s: float | None = LIMITE_INTERVALO_PADRAO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa as leituras em eventos.

    Devolve `(leituras, eventos)`:

    - `leituras` — o DataFrame de entrada ordenado por data, com a coluna
      `evento` dizendo a que evento cada linha pertence. Nenhuma linha e
      removida ou alterada.
    - `eventos`  — uma linha por evento: rotulo, quantas leituras, inicio, fim
      e duracao.

    A ordenacao por data e obrigatoria e feita aqui: o arquivo bruto vem fora de
    ordem, e agrupar linhas vizinhas num arquivo desordenado nao significa nada.
    """
    tempo, rotulo = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    leituras = df.sort_values(tempo, kind="stable").reset_index(drop=True)
    if leituras.empty:
        vazio = pd.DataFrame(
            columns=["evento", rotulo, "n_leituras", "inicio", "fim",
                     "duracao_s", "duracao_min"]
        )
        return leituras.assign(evento=pd.Series(dtype="int64")), vazio

    cortes = [segmentos.mudou_valor(leituras[rotulo])]
    if limite_intervalo_s is not None:
        cortes.append(segmentos.passou_intervalo(leituras[tempo], limite_intervalo_s))

    grupos = segmentos.numerar_grupos(*cortes)
    leituras["evento"] = grupos

    eventos = segmentos.resumir_grupos(
        leituras, grupos, coluna_tempo=tempo, primeiro_de=[rotulo]
    ).rename(columns={"_grupo": "evento"})

    return leituras, eventos


def diagnostico_eventos(
    leituras: pd.DataFrame, eventos: pd.DataFrame, limite_alerta_s: float = 60.0
) -> pd.DataFrame:
    """Aponta eventos que provavelmente deveriam ser mais de um.

    Como a regra atual so quebra na mudanca de rotulo, um evento pode conter uma
    interrupcao longa por dentro: pararam de gravar e retomaram o mesmo ensaio
    dias depois, sem trocar o nome.

    A coluna `maior_buraco_s` mede a maior interrupcao interna. Nada e corrigido
    — a funcao existe para o custo da regra ficar visivel, e nao escondido.
    """
    tempo = config.COLUNA_TEMPO

    buracos = segmentos.maior_buraco_interno(leituras, leituras["evento"], tempo)
    saida = eventos.copy()
    saida["maior_buraco_s"] = saida["evento"].map(buracos).fillna(0.0)
    saida["maior_buraco_h"] = (saida["maior_buraco_s"] / 3600).round(2)
    saida["suspeito"] = saida["maior_buraco_s"] > limite_alerta_s

    return (
        saida[saida["suspeito"]]
        .sort_values("maior_buraco_s", ascending=False)
        .reset_index(drop=True)
    )


def validar_eventos(
    leituras: pd.DataFrame, eventos: pd.DataFrame, original: pd.DataFrame
) -> pd.DataFrame:
    """Checagens que ou passam ou falham, sem espaco para interpretacao.

    Rodam sempre que os eventos sao construidos. Se alguma falhar, o
    agrupamento esta errado e nao adianta olhar o resultado.
    """
    rotulo = config.COLUNA_ROTULO

    rotulos_por_evento = leituras.groupby("evento")[rotulo].nunique()
    soma = int(eventos["n_leituras"].sum())
    ids = leituras["evento"].to_numpy()

    checagens = [
        (
            "Nenhum evento mistura dois rotulos",
            bool((rotulos_por_evento <= 1).all()),
            f"{int((rotulos_por_evento > 1).sum())} evento(s) com mais de um rotulo",
        ),
        (
            "Nenhuma leitura foi perdida ou duplicada",
            soma == len(original),
            f"{soma:,} leituras nos eventos vs {len(original):,} no arquivo".replace(",", "."),
        ),
        (
            "Toda leitura pertence a um evento",
            bool(leituras["evento"].notna().all()),
            f"{int(leituras['evento'].isna().sum())} leitura(s) sem evento",
        ),
        (
            "As leituras estao em ordem de data",
            bool(leituras[config.COLUNA_TEMPO].is_monotonic_increasing),
            "ordenacao crescente por created_at",
        ),
        (
            "Os numeros de evento sao consecutivos",
            bool(np.array_equal(np.unique(ids), np.arange(1, len(eventos) + 1))),
            f"{len(eventos)} eventos, de {ids.min()} a {ids.max()}",
        ),
    ]

    return pd.DataFrame(checagens, columns=["checagem", "passou", "detalhe"])


def resumo_por_rotulo(eventos: pd.DataFrame) -> pd.DataFrame:
    """Quantos eventos e quanto tempo cada rotulo acumula.

    E a tabela que responde a pergunta do operador: nao "quantas linhas", mas
    "quantas vezes".
    """
    rotulo = config.COLUNA_ROTULO

    resumo = (
        eventos.groupby(rotulo, observed=True)
        .agg(
            eventos=("evento", "size"),
            leituras=("n_leituras", "sum"),
            duracao_mediana_min=("duracao_min", "median"),
            duracao_total_h=("duracao_s", lambda s: s.sum() / 3600),
            primeira=("inicio", "min"),
            ultima=("fim", "max"),
        )
        .reset_index()
    )
    resumo["leituras_por_evento"] = (resumo["leituras"] / resumo["eventos"]).round(0)
    return resumo.sort_values("leituras", ascending=False).reset_index(drop=True)
