"""A floresta que nomeia a familia a partir de um trecho de leituras.

`RandomForestClassifier` com 400 arvores. Escolhida por tres motivos: nao exige
padronizar escala, aguenta classe com poucos exemplos, e sabe dizer em que se
baseou.

**O que ela decide, e o que nao decide.** Ela nomeia a familia que abre a
conversa — mas nao escolhe manual e nao entra no caminho do modelo de linguagem.
Quem autoriza a prescricao continua sendo o `fault_map.yaml`, por `SELECT`. Com
44% de acuracia honesta (ver `validacao`), o veredito dela passa antes pelo
**G1**, que barra confianca baixa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mp import config
from mp.classificacao import amostras as A

__all__ = ["Classificador", "treinar"]


class Classificador:
    """Treina numa passada e responde ranking de familias para um trecho novo.

    O preparo do trecho consultado usa exatamente o mesmo caminho do treino —
    as mesmas colunas, na mesma ordem, com o mesmo resumo. E o motivo de o
    objeto guardar `colunas` em vez de recalcular: se o trecho novo tivesse uma
    coluna a mais ou a menos, a matriz teria outra largura e o erro apareceria
    como numero errado, nao como excecao.
    """

    def __init__(
        self,
        modo: str = "janela",
        tamanho: int | None = None,
        incluir_regime: bool = False,
        so_defeitos: bool = False,
        n_arvores: int | None = None,
        semente: int | None = None,
    ):
        self.modo = modo
        self.tamanho = tamanho or config.CLF_JANELA_TAMANHO
        self.incluir_regime = incluir_regime
        self.so_defeitos = so_defeitos
        self.n_arvores = n_arvores or config.CLF_N_ARVORES
        self.semente = config.CLF_SEMENTE if semente is None else semente

        self.floresta = RandomForestClassifier(
            n_estimators=self.n_arvores,
            random_state=self.semente,
            n_jobs=-1,
        )
        self.colunas: list[str] = []
        self.nomes_features: list[str] = []
        self.classes_: np.ndarray | None = None
        self.n_amostras = 0
        self.n_eventos = 0

    def treinar(self, df_bruto: pd.DataFrame) -> "Classificador":
        """Do CSV cru ao modelo ajustado, sem etapa manual no meio."""
        leituras = A.preparar(df_bruto, so_defeitos=self.so_defeitos)
        tabela = A.criar_amostras(
            leituras,
            modo=self.modo,
            tamanho=self.tamanho,
            incluir_regime=self.incluir_regime,
        )
        return self.ajustar(tabela)

    def ajustar(self, tabela: pd.DataFrame) -> "Classificador":
        """Ajusta a floresta a uma tabela de amostras **ja montada**.

        `treinar` e a composicao completa; este e o ultimo passo dela. Existe
        separado por causa do painel de execucao, que cronometra cada etapa: se
        o treino remontasse as amostras por dentro, o tempo dele incluiria o das
        etapas anteriores.
        """
        if tabela.empty:
            raise ValueError(
                f"Nenhuma amostra com janela de {self.tamanho} leituras. "
                "Todo evento e mais curto que a janela."
            )

        X, y, grupos = A.matriz(tabela)
        self.floresta.fit(X, y)

        self.classes_ = self.floresta.classes_
        self.colunas = list(tabela.attrs.get("colunas") or [])
        self.nomes_features = list(tabela.attrs.get("nomes_features") or [])
        if not self.nomes_features:
            self.nomes_features = [f"f{i}" for i in range(X.shape[1])]
        self.n_amostras = len(tabela)
        self.n_eventos = int(pd.unique(grupos).size)
        return self

    # -- consulta ----------------------------------------------------------

    def _matriz_do_trecho(self, trecho: pd.DataFrame) -> np.ndarray:
        """Recorta o trecho consultado do mesmo jeito que o treino recortou.

        Trecho mais curto que a janela nao e recusado: vira uma amostra so, com
        o bloco inteiro. As cinco estatisticas nao dependem do comprimento, e
        recusar seria pior — na pratica quem consulta manda o que tem.
        """
        if trecho.empty:
            raise ValueError("O trecho esta vazio.")

        faltando = [c for c in self.colunas if c not in trecho.columns]
        if faltando:
            raise ValueError(f"Faltam colunas no trecho: {faltando}")

        if self.modo == "evento" or len(trecho) < self.tamanho:
            return np.vstack([A.resumir_bloco(trecho, self.colunas)])

        passo = max(1, self.tamanho // 2)
        blocos = [
            A.resumir_bloco(trecho.iloc[i : i + self.tamanho], self.colunas)
            for i in range(0, len(trecho) - self.tamanho + 1, passo)
        ]
        return np.vstack(blocos)

    def consultar(self, trecho: pd.DataFrame) -> pd.DataFrame:
        """Ranking de familias para um trecho de leituras.

        Varias janelas votam por **media de probabilidade**, nao por maioria: a
        media preserva a duvida, enquanto o voto devolveria a vencedora com cara
        de certeza.
        """
        if self.classes_ is None:
            raise RuntimeError("O modelo ainda nao foi treinado.")

        X = self._matriz_do_trecho(trecho)
        probabilidades = self.floresta.predict_proba(X).mean(axis=0)

        return (
            pd.DataFrame(
                {"familia": self.classes_, "probabilidade": probabilidades}
            )
            .sort_values("probabilidade", ascending=False)
            .reset_index(drop=True)
        )

    def importancia(self, top: int = 20) -> pd.DataFrame:
        """As features que mais pesaram, agregadas tambem por coluna de origem.

        A importancia bruta e por feature (`z_kurtosis__mediana`), e sozinha ela
        engana: uma coluna forte espalhada entre cinco estatisticas parece fraca
        ao lado de uma coluna que concentra tudo numa. Por isso a soma por
        coluna vem junto.
        """
        if self.classes_ is None:
            raise RuntimeError("O modelo ainda nao foi treinado.")

        tabela = pd.DataFrame(
            {
                "feature": self.nomes_features,
                "importancia": self.floresta.feature_importances_,
            }
        )
        tabela["coluna"] = tabela["feature"].str.split("__").str[0]
        tabela["estatistica"] = tabela["feature"].str.split("__").str[1]
        return tabela.sort_values("importancia", ascending=False).head(top).reset_index(
            drop=True
        )

    def importancia_por_coluna(self) -> pd.DataFrame:
        """Soma da importancia das cinco estatisticas de cada coluna."""
        if self.classes_ is None:
            raise RuntimeError("O modelo ainda nao foi treinado.")

        tabela = pd.DataFrame(
            {
                "coluna": [n.split("__")[0] for n in self.nomes_features],
                "importancia": self.floresta.feature_importances_,
            }
        )
        return (
            tabela.groupby("coluna", as_index=False)["importancia"]
            .sum()
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True)
        )


def treinar(df_bruto: pd.DataFrame, **kwargs) -> Classificador:
    """Atalho: `Classificador(**kwargs).treinar(df)`."""
    return Classificador(**kwargs).treinar(df_bruto)


def prever_evento_segurado(
    leituras: pd.DataFrame,
    evento: int,
    tamanho: int | None = None,
    incluir_regime: bool = False,
    n_arvores: int | None = None,
    semente: int | None = None,
) -> dict:
    """Treina **sem** um evento e depois pergunta a ele. Um caso, honesto.

    Existe para a demonstracao na tela. Perguntar a um modelo treinado no
    conjunto inteiro sobre um evento que estava nele nao demonstra nada — ele
    reconhece o que decorou. Aqui o evento fica de fora do treino, que e o mesmo
    criterio da estrategia `por_evento`.
    """
    tamanho = tamanho or config.CLF_JANELA_TAMANHO
    n_arvores = n_arvores or config.CLF_N_ARVORES
    semente = config.CLF_SEMENTE if semente is None else semente

    tabela = A.criar_amostras(
        leituras, modo="janela", tamanho=tamanho, incluir_regime=incluir_regime
    )
    if tabela.empty:
        raise ValueError("Nenhuma amostra foi gerada com esta janela.")

    X, y, grupos = A.matriz(tabela)
    fora = grupos == evento
    if not fora.any():
        raise ValueError(
            f"O evento {evento} nao gerou nenhuma janela — ele e mais curto que "
            f"{tamanho} leituras."
        )

    floresta = RandomForestClassifier(
        n_estimators=n_arvores, random_state=semente, n_jobs=-1
    )
    floresta.fit(X[~fora], y[~fora])

    probabilidades = floresta.predict_proba(X[fora]).mean(axis=0)
    ranking = (
        pd.DataFrame({"familia": floresta.classes_, "probabilidade": probabilidades})
        .sort_values("probabilidade", ascending=False)
        .reset_index(drop=True)
    )

    verdadeira = str(y[fora][0])
    return {
        "evento": int(evento),
        "familia_verdadeira": verdadeira,
        "ranking": ranking,
        "previsto": str(ranking.iloc[0]["familia"]),
        "acertou": str(ranking.iloc[0]["familia"]) == verdadeira,
        "n_janelas": int(fora.sum()),
        "n_amostras_treino": int((~fora).sum()),
    }
