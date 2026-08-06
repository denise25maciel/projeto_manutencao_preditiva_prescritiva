"""Checagens de qualidade do dado bruto.

Estas funcoes voltam a rodar na ingestao (Parte 2): o que aqui vira relatorio,
la vira validacao. Por isso sao modulo e nao celula de notebook.

Nenhuma delas modifica o DataFrame. Todas devolvem descricao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mp import config
from mp.analysis.loader import colunas_numericas


def colunas_constantes(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas com um unico valor distinto — informacao zero.

    Variancia nula nao distingue nada e ainda quebra o StandardScaler
    (divisao por desvio padrao zero). Devolve tambem as quase-constantes para
    julgamento humano: uma coluna com 2 valores em 166 mil linhas raramente
    ajuda, mas pode ser justamente o sinal raro que interessa.
    """
    linhas = []
    for c in df.columns:
        distintos = int(df[c].nunique(dropna=True))
        moda = df[c].mode(dropna=True)
        valor_dominante = moda.iloc[0] if len(moda) else None
        freq = float((df[c] == valor_dominante).mean() * 100) if valor_dominante is not None else 0.0
        linhas.append(
            {
                "coluna": c,
                "distintos": distintos,
                # Texto: a coluna mistura o dominante de `fault` (str), de `rpm`
                # (float) e de `created_at` (timestamp). Deixar como object faria
                # a serializacao Arrow da UI falhar ao inferir um tipo unico.
                "valor_dominante": "" if valor_dominante is None else str(valor_dominante),
                "pct_dominante": round(freq, 2),
                "constante": distintos <= config.MAX_DISTINTOS_QUASE_CONSTANTE,
            }
        )
    return pd.DataFrame(linhas).sort_values("distintos").reset_index(drop=True)


def colunas_redundantes(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica os pares de unidade duplicada declarados no config.

    Testa a identidade numerica, nao a correlacao: `mm_s == in_s * 25.4` e
    `f == c * 9/5 + 32`. O erro maximo observado fica na casa do arredondamento
    do arquivo (4 casas decimais), o que confirma que uma coluna e conversao
    da outra e nao duas medidas independentes.

    Confirmada a redundancia, descartamos a versao imperial e ficamos com o SI.
    """
    linhas = []

    for imperial, si, fator in config.PARES_REDUNDANTES:
        if imperial not in df.columns or si not in df.columns:
            continue
        erro = (df[si] - df[imperial] * fator).abs()
        linhas.append(_linha_redundancia(imperial, si, f"{si} = {imperial} x {fator}", erro))

    f, c = config.PAR_TEMPERATURA
    if f in df.columns and c in df.columns:
        erro = (df[f] - (df[c] * 9 / 5 + 32)).abs()
        linhas.append(_linha_redundancia(f, c, f"{f} = {c} x 9/5 + 32", erro))

    return pd.DataFrame(linhas)


def _linha_redundancia(descartavel: str, manter: str, relacao: str, erro: pd.Series) -> dict:
    erro_max = float(erro.max())
    return {
        "coluna_descartavel": descartavel,
        "coluna_mantida": manter,
        "relacao": relacao,
        "erro_max": round(erro_max, 6),
        "erro_medio": round(float(erro.mean()), 6),
        "redundante": bool(erro_max <= config.TOLERANCIA_REDUNDANCIA),
    }


def duplicatas_consecutivas(df: pd.DataFrame) -> dict:
    """Linhas identicas a anterior em todas as colunas de medida.

    Ignora `id` e `created_at`, que sempre mudam e nunca acusariam duplicata.

    Duas leituras iguais em 4 casas decimais a 2 s de distancia sao a mesma
    amostra repetida pelo datalogger, nao dois instantes fisicos iguais. Elas
    inflam a contagem de ocorrencias e viram vizinhos de distancia zero no kNN.
    """
    cols = [c for c in df.columns if c not in (config.COLUNA_ID, config.COLUNA_TEMPO)]
    igual_a_anterior = (df[cols] == df[cols].shift()).all(axis=1)

    por_rotulo = (
        df.assign(_dup=igual_a_anterior)
        .groupby(config.COLUNA_ROTULO, dropna=False, observed=True)["_dup"]
        .agg(duplicadas="sum", total="size")
        .reset_index()
    )
    por_rotulo["pct"] = (por_rotulo["duplicadas"] / por_rotulo["total"] * 100).round(2)

    return {
        "total": int(igual_a_anterior.sum()),
        "pct": round(float(igual_a_anterior.mean() * 100), 2),
        "por_rotulo": por_rotulo.sort_values("duplicadas", ascending=False).reset_index(drop=True),
        "mascara": igual_a_anterior,
    }


def timestamps_duplicados(df: pd.DataFrame) -> dict:
    """Linhas que compartilham exatamente o mesmo `created_at`.

    Em `duplicatas_consecutivas` o que se repete e a MEDIDA; aqui e o INSTANTE.
    Duas leituras no mesmo instante ou sao o mesmo registro gravado duas vezes
    (as medidas coincidem) ou sao leituras reais com carimbo errado (as medidas
    variam) — `medidas_identicas` separa os casos.

    O segundo e o grave: o dado de vibracao serve para assinatura e
    similaridade, mas nao pode ancorar nada temporal.
    """
    tempo, ident, rot = config.COLUNA_TEMPO, config.COLUNA_ID, config.COLUNA_ROTULO

    dup = df[df[tempo].duplicated(keep=False)].sort_values([tempo, ident])
    if dup.empty:
        return {"total": 0, "instantes": 0,
                "resumo": pd.DataFrame(), "linhas": dup}

    medidas = [c for c in df.columns if c not in (ident, tempo)]
    g = dup.groupby(tempo, sort=True)

    resumo = pd.DataFrame(
        {
            "linhas": g.size(),
            "rotulos": g[rot].agg(lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
            "id_min": g[ident].min(),
            "id_max": g[ident].max(),
            # Todas as colunas de medida com um unico valor => copia do registro.
            "medidas_identicas": g[medidas].nunique().max(axis=1) == 1,
            # Quantas linhas do bloco repetem exatamente outra linha do bloco.
            "linhas_repetidas": g[medidas].apply(
                lambda bloco: int(bloco.duplicated().sum())
            ),
        }
    ).reset_index()

    # Se um bloco de N linhas ocupa exatamente N ids consecutivos, ele entrou no
    # arquivo de uma vez — carga em lote, nao coleta espalhada no tempo.
    resumo["ids_contiguos"] = (resumo["id_max"] - resumo["id_min"] + 1) == resumo["linhas"]

    return {
        "total": int(len(dup)),
        "instantes": int(resumo.shape[0]),
        "resumo": resumo,
        "linhas": dup,
    }


def outliers_iqr(
    df: pd.DataFrame,
    colunas: list[str] | None = None,
    por_rotulo: bool = False,
) -> pd.DataFrame:
    """Conta outliers por coluna pelo criterio de Tukey.

    Limites Q1 - k*IQR e Q3 + k*IQR, com k = 1,5 (moderado) e 3,0 (extremo).
    IQR e o *intervalo interquartil*, a distancia entre o 1o e o 3o quartil.

    IQR e nao z-score porque varias colunas sao muito assimetricas (kurtosis tem
    mediana 2,5 e maximo 65): a media e o desvio que o z-score usa ja vem
    contaminados pelos proprios extremos.

    `por_rotulo=True` calcula os limites dentro de cada `fault`. Globalmente,
    toda leitura de defeito severo parece outlier — o que e esperado, nao erro
    de medicao.

    NADA e removido ou corrigido aqui.
    """
    colunas = colunas or colunas_numericas(df)

    if por_rotulo:
        blocos = []
        for rotulo, g in df.groupby(config.COLUNA_ROTULO, dropna=False, observed=True):
            b = _outliers_bloco(g, colunas)
            b.insert(0, "fault", rotulo)
            blocos.append(b)
        return pd.concat(blocos, ignore_index=True)

    return _outliers_bloco(df, colunas)


def _outliers_bloco(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    linhas = []
    n = len(df)

    for c in colunas:
        s = df[c].dropna()
        if s.empty:
            continue

        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1

        lim_inf_mod = q1 - config.IQR_FATOR_MODERADO * iqr
        lim_sup_mod = q3 + config.IQR_FATOR_MODERADO * iqr
        lim_inf_ext = q1 - config.IQR_FATOR_EXTREMO * iqr
        lim_sup_ext = q3 + config.IQR_FATOR_EXTREMO * iqr

        moderados = ((s < lim_inf_mod) | (s > lim_sup_mod)).sum()
        extremos = ((s < lim_inf_ext) | (s > lim_sup_ext)).sum()

        linhas.append(
            {
                "coluna": c,
                "q1": round(q1, 4),
                "mediana": round(float(s.median()), 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lim_inferior": round(lim_inf_mod, 4),
                "lim_superior": round(lim_sup_mod, 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "outliers": int(moderados),
                "pct_outliers": round(float(moderados) / n * 100, 2) if n else 0.0,
                "extremos": int(extremos),
                "pct_extremos": round(float(extremos) / n * 100, 2) if n else 0.0,
                # Quantas vezes o maximo ultrapassa o limite superior. Valor alto
                # indica cauda longa de impacto, nao ruido disperso.
                "max_sobre_limite": (
                    round(float(s.max()) / lim_sup_mod, 2) if lim_sup_mod > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(linhas).sort_values("pct_outliers", ascending=False).reset_index(drop=True)


def relatorio_qualidade(df: pd.DataFrame) -> dict:
    """Roda todas as checagens de uma vez. Ponto de entrada da UI e do notebook."""
    return {
        "constantes": colunas_constantes(df),
        "redundantes": colunas_redundantes(df),
        "duplicatas": duplicatas_consecutivas(df),
        "tempos_duplicados": timestamps_duplicados(df),
        "outliers": outliers_iqr(df),
    }
