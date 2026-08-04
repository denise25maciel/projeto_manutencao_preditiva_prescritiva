"""Assinatura de vibracao por rotulo.

Entrega principal da Parte 0: uma tabela que resume, em medianas, como cada
rotulo se comporta. E ela que sera cruzada com o que os PDFs de procedimento
descrevem — divergencia entre a assinatura medida e a descrita e achado, nao
erro (GUIA.md, Parte 1).

Usamos MEDIANA, nao media. Kurtosis e crest factor sao definidos sobre picos:
um unico impacto isolado desloca a media do rotulo inteiro. A mediana descreve
o comportamento tipico; a dispersao fica nos quartis.
"""

from __future__ import annotations

import pandas as pd

from mp import config
from mp.analysis.quality import colunas_redundantes


def _com_razao(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a razao entre eixos como coluna derivada.

    A razao entre a velocidade RMS dos dois eixos e o discriminante classico
    entre desalinhamento (energia sobe no eixo axial) e desbalanceamento
    (energia concentra no radial). Como o dataset nao documenta a orientacao do
    sensor, o nome fica neutro e a interpretacao vem do cruzamento com os PDFs.
    """
    num, den = config.NUMERADOR_RAZAO, config.DENOMINADOR_RAZAO
    if num not in df.columns or den not in df.columns:
        return df

    out = df.copy()
    # Denominador zerado viraria inf e contaminaria a mediana; vira nulo.
    out[config.NOME_RAZAO] = (out[num] / out[den].where(out[den] != 0)).astype(float)
    return out


def _colunas_assinatura(df: pd.DataFrame) -> list[str]:
    cols = [c for c in config.COLUNAS_ASSINATURA if c in df.columns]
    if config.NOME_RAZAO in df.columns:
        cols = [config.NOME_RAZAO] + cols
    return cols


def assinaturas_por_rotulo(df: pd.DataFrame, min_leituras: int = 1) -> pd.DataFrame:
    """Mediana de cada feature de assinatura, um registro por rotulo.

    `min_leituras` corta rotulos com amostra pequena demais para uma mediana
    significar algo (varios rotulos do banner.csv tem 2 leituras).
    """
    d = _com_razao(df)
    cols = _colunas_assinatura(d)

    g = d.groupby(config.COLUNA_ROTULO, dropna=False, observed=True)
    tabela = g[cols].median().round(4)
    tabela.insert(0, "n_leituras", g.size())

    tabela = tabela[tabela["n_leituras"] >= min_leituras]
    return tabela.reset_index().sort_values("n_leituras", ascending=False).reset_index(drop=True)


def assinatura_de_rotulo(df: pd.DataFrame, rotulo: str) -> pd.DataFrame:
    """Estatistica detalhada de um unico rotulo.

    Devolve n, mediana, quartis, min/max e desvio por feature. Os quartis
    respondem a pergunta que a mediana sozinha nao responde: o rotulo e
    estavel ou a leitura varia muito dentro da propria classe? Classe dispersa
    e classe que o kNN vai confundir na Parte 3.
    """
    d = _com_razao(df)
    sub = d[d[config.COLUNA_ROTULO] == rotulo]
    if sub.empty:
        return pd.DataFrame()

    cols = _colunas_assinatura(d)
    desc = sub[cols].describe(percentiles=[0.25, 0.5, 0.75]).T

    desc = desc.rename(
        columns={
            "count": "n",
            "mean": "media",
            "std": "desvio",
            "min": "min",
            "25%": "q1",
            "50%": "mediana",
            "75%": "q3",
            "max": "max",
        }
    )
    desc["n"] = desc["n"].astype(int)

    # Coeficiente de variacao: dispersao relativa, comparavel entre features de
    # escalas diferentes (mm/s vs adimensional vs Hz).
    desc["cv"] = (desc["desvio"] / desc["media"].abs().where(desc["media"] != 0)).round(3)

    ordem = ["n", "mediana", "q1", "q3", "media", "desvio", "cv", "min", "max"]
    return desc[ordem].round(4).reset_index(names="feature")


def comparar_com_global(df: pd.DataFrame, rotulo: str) -> pd.DataFrame:
    """Mediana do rotulo contra a mediana do dataset inteiro.

    Responde "o que torna este rotulo diferente". A coluna `desvio_pct` e o
    quanto a feature se afasta do comportamento geral — e a leitura mais util
    para cruzar com a secao de sintomas dos PDFs.
    """
    d = _com_razao(df)
    cols = _colunas_assinatura(d)

    sub = d[d[config.COLUNA_ROTULO] == rotulo]
    if sub.empty:
        return pd.DataFrame()

    med_rotulo = sub[cols].median()
    med_global = d[cols].median()

    out = pd.DataFrame(
        {
            "feature": cols,
            "mediana_rotulo": med_rotulo.to_numpy().round(4),
            "mediana_global": med_global.to_numpy().round(4),
        }
    )
    out["desvio_pct"] = (
        (out["mediana_rotulo"] - out["mediana_global"])
        / out["mediana_global"].abs().where(out["mediana_global"] != 0)
        * 100
    ).round(1)

    return out.reindex(out["desvio_pct"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def colunas_a_descartar(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida a decisao de descarte da Parte 0, com o motivo de cada uma.

    Este e o artefato que a Parte 1 aplica. Tres motivos:
      - redundante: e conversao de unidade de outra coluna (confirmado numerico)
      - vazamento: identificador correlacionado com a ordem de coleta
      - constante: variancia nula, nao distingue nada
    """
    linhas = []

    red = colunas_redundantes(df)
    for _, r in red.iterrows():
        if r["redundante"]:
            linhas.append(
                {
                    "coluna": r["coluna_descartavel"],
                    "motivo": "redundante",
                    "detalhe": f"{r['relacao']} (erro max {r['erro_max']})",
                }
            )

    for c in config.COLUNAS_VAZAMENTO:
        if c in df.columns:
            linhas.append(
                {
                    "coluna": c,
                    "motivo": "vazamento",
                    "detalhe": "identificador temporal — nao e medida fisica, "
                    "vira atalho para o kNN",
                }
            )

    for c in df.columns:
        if df[c].nunique(dropna=True) <= config.MAX_DISTINTOS_QUASE_CONSTANTE:
            linhas.append(
                {"coluna": c, "motivo": "constante", "detalhe": "um unico valor distinto"}
            )

    if not linhas:
        return pd.DataFrame(columns=["coluna", "motivo", "detalhe"])

    return pd.DataFrame(linhas).drop_duplicates(subset="coluna").reset_index(drop=True)
