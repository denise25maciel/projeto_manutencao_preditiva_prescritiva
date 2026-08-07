"""As duas maneiras de dividir treino e teste — e por que discordam tanto.

    aleatoria    embaralha as amostras e reparte em 5 partes
    por_evento   reparte EVENTOS inteiros; nenhum fica dos dois lados

A primeira infla. Duas janelas do mesmo evento sao quase identicas — mesma
bancada, mesma montagem, e metade do conteudo repetido pela sobreposicao —,
entao ao embaralhar uma cai no treino e a outra no teste, e o modelo reencontra
na prova o que estudou.

**A distancia entre as duas notas mede o tamanho do problema**, em vez de so
afirmar que ele existe. So aparece no modo `janela`: no modo `evento` cada grupo
tem uma amostra so, e nao ha o que vazar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from mp import config
from mp.classificacao import amostras as A

__all__ = [
    "ESTRATEGIAS",
    "dividir_treino_teste",
    "validar",
    "matriz_de_confusao",
    "acerto_por_familia",
    "experimento_janela",
    "experimento_regime",
    "linha_de_base",
]

ESTRATEGIAS = {
    "aleatoria": "Sorteia amostras soltas — janelas do mesmo evento caem dos dois lados",
    "por_evento": "Sorteia eventos inteiros — nenhum evento aparece nos dois lados",
}


def _classes_com_folds_suficientes(
    y: np.ndarray, grupos: np.ndarray, n_folds: int
) -> tuple[np.ndarray, list[str]]:
    """Mascara das amostras que podem ser validadas, e as classes deixadas fora.

    Uma classe com menos de `n_folds` **eventos** nao pode ser repartida em
    `n_folds` partes sem que alguma fique vazia. Em vez de deixar o `sklearn`
    levantar excecao ou avisar em silencio, tiramos essas classes e devolvemos a
    lista — porque a classe que nao da para validar e um resultado do projeto,
    nao um detalhe de implementacao: significa que aquele defeito foi medido
    poucas vezes na bancada.
    """
    tabela = pd.DataFrame({"y": y, "grupo": grupos})
    eventos_por_classe = tabela.groupby("y")["grupo"].nunique()
    raras = sorted(eventos_por_classe[eventos_por_classe < n_folds].index.tolist())
    mascara = ~np.isin(y, raras)
    return mascara, raras


def validar(
    amostras: pd.DataFrame,
    n_arvores: int | None = None,
    n_folds: int | None = None,
    semente: int | None = None,
) -> dict:
    """Roda as duas estrategias sobre as mesmas amostras.

    Devolve, por estrategia: acuracia media, desvio entre folds, F1 macro e o
    detalhe de cada fold — com `y_true`/`y_pred` guardados, para a matriz de
    confusao sair depois sem retreinar.

    O F1 macro vem junto porque a acuracia e dominada pelas familias grandes; a
    distancia entre os dois diz se o acerto esta concentrado nelas.
    """
    n_arvores = n_arvores or config.CLF_N_ARVORES
    n_folds = n_folds or config.CLF_N_FOLDS
    semente = config.CLF_SEMENTE if semente is None else semente

    X, y, grupos = A.matriz(amostras)
    mascara, classes_raras = _classes_com_folds_suficientes(y, grupos, n_folds)
    X, y, grupos = X[mascara], y[mascara], grupos[mascara]

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Sobrou menos de duas familias validaveis. Reduza a janela ou o "
            "numero de folds."
        )

    divisores = {
        "aleatoria": StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=semente
        ),
        "por_evento": StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=semente
        ),
    }

    resultados = {}
    for nome, divisor in divisores.items():
        folds = []
        for i, (treino, teste) in enumerate(divisor.split(X, y, grupos), start=1):
            floresta = RandomForestClassifier(
                n_estimators=n_arvores, random_state=semente, n_jobs=-1
            )
            floresta.fit(X[treino], y[treino])
            previsto = floresta.predict(X[teste])

            eventos_treino = set(grupos[treino].tolist())
            eventos_teste = set(grupos[teste].tolist())

            folds.append(
                {
                    "fold": i,
                    "n_treino": int(len(treino)),
                    "n_teste": int(len(teste)),
                    "eventos_treino": len(eventos_treino),
                    "eventos_teste": len(eventos_teste),
                    # A prova do vazamento: quantos eventos aparecem dos dois
                    # lados. Na estrategia por evento tem de ser sempre zero.
                    "eventos_vazados": len(eventos_treino & eventos_teste),
                    "acuracia": float(accuracy_score(y[teste], previsto)),
                    "f1_macro": float(
                        f1_score(y[teste], previsto, average="macro", zero_division=0)
                    ),
                    "y_true": y[teste],
                    "y_pred": previsto,
                }
            )

        acuracias = [f["acuracia"] for f in folds]
        resultados[nome] = {
            "nome": nome,
            "descricao": ESTRATEGIAS[nome],
            "folds": folds,
            "acuracia": float(np.mean(acuracias)),
            "desvio": float(np.std(acuracias)),
            "f1_macro": float(np.mean([f["f1_macro"] for f in folds])),
            "eventos_vazados": int(sum(f["eventos_vazados"] for f in folds)),
        }

    inflacao = resultados["aleatoria"]["acuracia"] - resultados["por_evento"]["acuracia"]

    return {
        "estrategias": resultados,
        "n_amostras": int(len(y)),
        "n_eventos": int(pd.unique(grupos).size),
        "n_familias": int(len(np.unique(y))),
        "classes_raras": classes_raras,
        "n_folds": n_folds,
        "n_arvores": n_arvores,
        "linha_de_base": linha_de_base(y),
        # A distancia entre as duas notas. E o numero que resume o achado.
        "inflacao": float(inflacao),
    }


def dividir_treino_teste(
    amostras: pd.DataFrame,
    fracao_teste: float = 0.2,
    por_evento: bool = True,
    semente: int | None = None,
) -> dict:
    """Um corte unico em treino e teste, para a base poder ser vista e baixada.

    `por_evento=True` sorteia **eventos inteiros**. `eventos_vazados` audita o
    proprio corte: com `True` tem de ser zero.

    **Nao e daqui que sai a acuracia** — um corte unico depende do sorteio. Quem
    mede e `validar`, que repete e tira a media.
    """
    semente = config.CLF_SEMENTE if semente is None else semente
    X, y, grupos = A.matriz(amostras)

    if por_evento:
        divisor = GroupShuffleSplit(
            n_splits=1, test_size=fracao_teste, random_state=semente
        )
        idx_treino, idx_teste = next(divisor.split(X, y, groups=grupos))
    else:
        # `stratify=y` mantem a proporcao das familias nos dois lados; sem isso
        # a comparacao entre as duas estrategias misturaria dois efeitos.
        idx_treino, idx_teste = train_test_split(
            np.arange(len(y)), test_size=fracao_teste,
            random_state=semente, stratify=y,
        )

    tabela = A.matriz_legivel(amostras)
    treino = tabela.iloc[idx_treino].reset_index(drop=True)
    teste = tabela.iloc[idx_teste].reset_index(drop=True)

    eventos_treino = set(grupos[idx_treino].tolist())
    eventos_teste = set(grupos[idx_teste].tolist())
    vazados = sorted(eventos_treino & eventos_teste)

    return {
        "treino": treino,
        "teste": teste,
        "por_evento": por_evento,
        "fracao_pedida": fracao_teste,
        "fracao_real": len(teste) / len(tabela),
        "eventos_treino": len(eventos_treino),
        "eventos_teste": len(eventos_teste),
        "eventos_vazados": len(vazados),
        "lista_vazados": vazados[:20],
        "familias_treino": int(treino[A.COLUNA_CLASSE].nunique()),
        "familias_teste": int(teste[A.COLUNA_CLASSE].nunique()),
        # Familia que existe no treino e nao no teste nao pode ser avaliada; o
        # contrario e pior, porque o modelo sera cobrado por um nome que nunca
        # viu. Sortear eventos inteiros torna os dois casos possiveis.
        "familias_so_no_treino": sorted(
            set(treino[A.COLUNA_CLASSE]) - set(teste[A.COLUNA_CLASSE])
        ),
        "familias_so_no_teste": sorted(
            set(teste[A.COLUNA_CLASSE]) - set(treino[A.COLUNA_CLASSE])
        ),
    }


def linha_de_base(y: np.ndarray) -> dict:
    """Com o que a acuracia tem de ser comparada para significar algo.

    Duas referencias, porque uma so engana. Chutar acerta `1/n`; responder
    sempre a familia mais comum acerta a fatia dela — e essa e sempre maior,
    entao e a barra de verdade.
    """
    contagem = pd.Series(y).value_counts()
    return {
        "n_familias": int(len(contagem)),
        "aleatorio": float(1 / len(contagem)),
        "familia_mais_comum": str(contagem.index[0]),
        "maioria": float(contagem.iloc[0] / len(y)),
    }


def matriz_de_confusao(resultado_estrategia: dict, normalizar: bool = True) -> pd.DataFrame:
    """Junta os folds numa matriz so: o que era x o que o modelo disse.

    Somar os folds e legitimo porque cada amostra e testada exatamente uma vez
    na validacao cruzada — a soma cobre o conjunto inteiro, sem repeticao.
    """
    y_true = np.concatenate([f["y_true"] for f in resultado_estrategia["folds"]])
    y_pred = np.concatenate([f["y_pred"] for f in resultado_estrategia["folds"]])
    familias = sorted(set(y_true) | set(y_pred))

    m = confusion_matrix(y_true, y_pred, labels=familias)
    if normalizar:
        linhas = m.sum(axis=1, keepdims=True)
        m = np.divide(m, linhas, out=np.zeros_like(m, dtype="float64"), where=linhas > 0)

    return pd.DataFrame(m, index=familias, columns=familias)


def acerto_por_familia(resultado_estrategia: dict) -> pd.DataFrame:
    """Quanto o modelo acerta em cada familia, e com quem a confunde.

    A acuracia global esconde isto: pode ir bem na media e nunca acertar a
    familia rara. `n_amostras` fica ao lado de proposito — 100% em 6 amostras
    nao e 100% em 600.
    """
    y_true = np.concatenate([f["y_true"] for f in resultado_estrategia["folds"]])
    y_pred = np.concatenate([f["y_pred"] for f in resultado_estrategia["folds"]])

    tabela = pd.DataFrame({"familia": y_true, "acertou": y_true == y_pred})
    resumo = (
        tabela.groupby("familia", as_index=False)
        .agg(n_amostras=("acertou", "size"), acertos=("acertou", "sum"))
    )
    resumo["acuracia"] = resumo["acertos"] / resumo["n_amostras"]

    # Com que familia cada uma e confundida com mais frequencia. E o que
    # transforma "erra muito" em "erra COM QUEM" — e ai da para perguntar se as
    # duas familias compartilham manual, o que tornaria o erro barato.
    erros = pd.DataFrame({"familia": y_true, "previsto": y_pred})
    erros = erros[erros["familia"] != erros["previsto"]]
    if not erros.empty:
        confusao = (
            erros.groupby(["familia", "previsto"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .drop_duplicates("familia")
            .rename(columns={"previsto": "confundida_com", "size": "n_confusoes"})
        )
        resumo = resumo.merge(confusao, on="familia", how="left")

    return resumo.sort_values("acuracia").reset_index(drop=True)


def experimento_janela(
    leituras: pd.DataFrame,
    tamanhos: tuple[int, ...] | None = None,
    incluir_regime: bool = False,
    n_arvores: int | None = None,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Valida varios tamanhos de janela e devolve a comparacao.

    As duas colunas que importam sao a acuracia honesta e a fatia de eventos
    descartados: a primeira costuma variar pouco, a segunda dispara — e e a
    segunda que decide quais familias sobram no conjunto.
    """
    tamanhos = tamanhos or config.CLF_JANELAS_TESTADAS
    linhas = []

    total_eventos = leituras[A.COLUNA_GRUPO].nunique()

    for tamanho in tamanhos:
        tabela = A.criar_amostras(
            leituras, modo="janela", tamanho=tamanho, incluir_regime=incluir_regime
        )
        if tabela.empty:
            linhas.append(
                {
                    "janela": tamanho,
                    "amostras": 0,
                    "eventos_usados": 0,
                    "pct_eventos_descartados": 100.0,
                    "familias": 0,
                    "acuracia_honesta": np.nan,
                    "acuracia_inflada": np.nan,
                    "inflacao": np.nan,
                }
            )
            continue

        resultado = validar(tabela, n_arvores=n_arvores, n_folds=n_folds)
        usados = tabela[A.COLUNA_GRUPO].nunique()
        linhas.append(
            {
                "janela": tamanho,
                # Cadencia nominal de 2 s por leitura: a janela em minutos e o
                # que o tecnico consegue julgar, "180 leituras" nao e.
                "minutos": tamanho * config.INTERVALO_ESPERADO_S / 60,
                "amostras": int(len(tabela)),
                "eventos_usados": int(usados),
                "pct_eventos_descartados": 100 * (1 - usados / total_eventos),
                "familias": int(resultado["n_familias"]),
                "acuracia_honesta": resultado["estrategias"]["por_evento"]["acuracia"],
                "acuracia_inflada": resultado["estrategias"]["aleatoria"]["acuracia"],
                "inflacao": resultado["inflacao"],
            }
        )

    return pd.DataFrame(linhas)


def experimento_regime(
    leituras: pd.DataFrame,
    tamanho: int | None = None,
    n_arvores: int | None = None,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Mede o preco de deixar `rpm` e temperatura entrarem como feature.

    O sinal a procurar nao e a acuracia honesta subir: e a **inflacao** crescer.
    Feature que ajuda a reconhecer o ensaio, e nao a falha, aparece assim —
    melhora a nota falsa sem melhorar a verdadeira.
    """
    tamanho = tamanho or config.CLF_JANELA_TAMANHO
    linhas = []

    for incluir in (False, True):
        tabela = A.criar_amostras(
            leituras, modo="janela", tamanho=tamanho, incluir_regime=incluir
        )
        resultado = validar(tabela, n_arvores=n_arvores, n_folds=n_folds)
        linhas.append(
            {
                "regime_como_feature": incluir,
                "colunas": len(tabela.attrs["colunas"]),
                "features": len(tabela.attrs["nomes_features"]),
                "acuracia_honesta": resultado["estrategias"]["por_evento"]["acuracia"],
                "acuracia_inflada": resultado["estrategias"]["aleatoria"]["acuracia"],
                "inflacao": resultado["inflacao"],
            }
        )

    return pd.DataFrame(linhas)
