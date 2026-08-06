"""kNN sobre o historico: dado um evento, quais leituras se parecem com ele.

Este e o unico lugar do projeto que responde **"que falha e essa?"** a partir de
numeros. Tudo o mais parte do rotulo ja resolvido.

O que a busca devolve nao e so o rotulo do vizinho mais proximo:

- **familia predominante** entre os vizinhos, com quantos votos
- **quantos episodios distintos** esses vizinhos ocupam — 25 vizinhos de um
  episodio so nao sao 25 confirmacoes, sao uma
- **distancia** ao mais proximo, que alimenta o **G1**
- **contexto operacional**: em que rotacao aqueles ensaios rodaram

A agregacao por episodio existe por causa da autocorrelacao: leituras
consecutivas do mesmo ensaio sao quase identicas, entao contar leituras
superestima a evidencia. Contar episodios nao.

**Sobre a acuracia.** Medida com validacao por grupo — segurando o rotulo
inteiro fora do treino —, ela e baixa. Isso e um achado sobre os dados, nao um
defeito da implementacao: o `banner.csv` traz medidas agregadas (RMS, pico,
kurtosis), e o manual de rolamentos diagnostica por frequencias de defeito
(BPFO, BPFI, BSF), que exigem espectro. O espectro nao esta no arquivo.
Ver `avaliar_por_grupo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from mp import config
from mp.similarity.features import Preparador

# Vizinhos consultados por padrao. 25 e o suficiente para uma maioria se formar
# sem varrer meio ensaio.
K_PADRAO = 25


@dataclass
class Diagnostico:
    """O que a similaridade viu, antes de qualquer texto."""

    familia: str | None = None
    rotulo: str | None = None
    confianca: float = 0.0
    votos: int = 0
    k: int = 0

    distancia_min: float = float("inf")
    distancia_media: float = 0.0
    # Viaja junto para o G1 nao precisar conhecer o indice.
    limiar_g1: float | None = None

    n_episodios: int = 0
    episodios: list = field(default_factory=list)
    rpm_dos_vizinhos: list = field(default_factory=list)

    vizinhos: pd.DataFrame | None = None
    distribuicao: pd.DataFrame | None = None

    # Preenchido quando o JSON traz `fault` do operador.
    rotulo_do_operador: str | None = None
    familia_do_operador: str | None = None

    @property
    def divergiu(self) -> bool:
        """O operador anotou uma familia e a similaridade indica outra."""
        return (
            self.familia_do_operador is not None
            and self.familia is not None
            and self.familia_do_operador != self.familia
        )

    @property
    def rpm_predominante(self) -> float | None:
        if not self.rpm_dos_vizinhos:
            return None
        return float(pd.Series(self.rpm_dos_vizinhos).mode().iloc[0])


class Indice:
    """O historico em memoria, pronto para responder por vizinhanca.

    `sklearn` em memoria e o suficiente no MVP: sao 166 mil linhas e 16 colunas,
    o que cabe folgado. Indice vetorial nativo esta marcado como `[R2]`.
    """

    def __init__(self, incluir_regime: bool = False, suavizar: int = 0):
        self.preparador = Preparador(incluir_regime=incluir_regime, suavizar=suavizar)
        self.modelo: NearestNeighbors | None = None
        self.rotulos: np.ndarray | None = None
        self.familias: np.ndarray | None = None
        self.eventos: np.ndarray | None = None
        self.rpm: np.ndarray | None = None
        self.limiar_g1: float | None = None

    # ----------------------------------------------------------------------
    def construir(self, df: pd.DataFrame, coluna_evento: str = "evento") -> "Indice":
        """Ajusta a escala e monta o indice. Roda uma vez."""
        from mp.retrieval.catalog import familia_de

        self.preparador.ajustar(df)
        X = self.preparador.transformar(df)

        self.modelo = NearestNeighbors(n_neighbors=K_PADRAO, algorithm="auto")
        self.modelo.fit(X)

        rot = df[config.COLUNA_ROTULO].astype(str).to_numpy()
        self.rotulos = rot

        mapa: dict[str, str | None] = {}
        for r in np.unique(rot):
            mapa[r] = familia_de(r)
        self.familias = np.array([mapa.get(r) or "?" for r in rot], dtype=object)

        self.eventos = (
            df[coluna_evento].to_numpy() if coluna_evento in df.columns
            else np.arange(len(df))
        )
        self.rpm = df["rpm"].to_numpy() if "rpm" in df.columns else np.zeros(len(df))

        self.limiar_g1 = self._calibrar_limiar(X)
        return self

    # ----------------------------------------------------------------------
    def _calibrar_limiar(self, X: np.ndarray, amostra: int = 3000,
                         percentil: float = 99.0) -> float:
        """De quao longe pode estar o vizinho mais proximo e ainda valer.

        Mede, numa amostra do historico, a distancia de cada ponto ao vizinho
        mais proximo que nao seja ele mesmo. Evento legitimo cai nessa faixa.

        O percentil 99 e frouxo de proposito: o **G1** barra o absurdo — outra
        maquina, sensor trocado —, nao decide entre falhas parecidas. Apertar
        aqui recusaria casos legitimos, que e o erro mais caro.
        """
        n = X.shape[0]
        idx = np.random.default_rng(42).choice(n, size=min(amostra, n), replace=False)
        distancias, _ = self.modelo.kneighbors(X[idx], n_neighbors=2)
        return float(np.percentile(distancias[:, 1], percentil))

    # ----------------------------------------------------------------------
    def consultar(self, evento: dict, k: int = K_PADRAO) -> Diagnostico:
        """Os `k` vizinhos do evento, agregados em um diagnostico."""
        from mp.retrieval.catalog import familia_de

        if self.modelo is None:
            raise RuntimeError("O indice ainda nao foi construido.")

        x = self.preparador.transformar_evento(evento)
        distancias, posicoes = self.modelo.kneighbors(x, n_neighbors=k)
        d, p = distancias[0], posicoes[0]

        vizinhos = pd.DataFrame(
            {
                "posicao": p,
                "distancia": d,
                "rotulo": self.rotulos[p],
                "familia": self.familias[p],
                "evento": self.eventos[p],
                "rpm": self.rpm[p],
            }
        )

        # O voto e por familia, nao por rotulo: `desalinhado_2` e
        # `new_desalinhado_0` sao o mesmo defeito com nome diferente.
        contagem = vizinhos["familia"].value_counts()
        familia = str(contagem.index[0]) if len(contagem) else None
        votos = int(contagem.iloc[0]) if len(contagem) else 0

        do_grupo = vizinhos[vizinhos["familia"] == familia]
        episodios = sorted(pd.unique(do_grupo["evento"]).tolist())

        distribuicao = pd.DataFrame(
            {
                "familia": contagem.index.astype(str),
                "vizinhos": contagem.to_numpy(),
                "pct": (contagem.to_numpy() / len(vizinhos) * 100).round(1),
            }
        )

        diag = Diagnostico(
            familia=familia,
            rotulo=str(do_grupo["rotulo"].mode().iloc[0]) if len(do_grupo) else None,
            confianca=round(votos / len(vizinhos), 3) if len(vizinhos) else 0.0,
            votos=votos,
            k=len(vizinhos),
            distancia_min=float(d.min()),
            distancia_media=float(d.mean()),
            limiar_g1=self.limiar_g1,
            n_episodios=len(episodios),
            episodios=episodios,
            rpm_dos_vizinhos=do_grupo["rpm"].tolist(),
            vizinhos=vizinhos,
            distribuicao=distribuicao,
        )

        # A anotacao do operador e opcional e nao manda: entra para ser
        # confrontada com o que os numeros dizem.
        if (anotado := evento.get(config.COLUNA_ROTULO)) is not None:
            diag.rotulo_do_operador = str(anotado)
            diag.familia_do_operador = familia_de(str(anotado))

        return diag


# --------------------------------------------------------------------------
# Validacao
# --------------------------------------------------------------------------


def avaliar_por_grupo(
    df: pd.DataFrame,
    indice: Indice | None = None,
    k: int = K_PADRAO,
    max_por_rotulo: int = 40,
) -> dict:
    """Acuracia segurando o **rotulo inteiro** fora do treino.

    Dividir linhas ao acaso infla o numero: leituras consecutivas do mesmo
    ensaio caem dos dois lados, e o vizinho mais proximo de uma leitura de teste
    e a de treino gravada dois segundos antes. Isso mede autocorrelacao, nao
    generalizacao.

    Aqui cada rotulo sai inteiro do indice, que e reconstruido sem ele: chega
    uma falha que o sistema nunca viu com esse nome — acerta a familia?
    """
    from mp.retrieval.catalog import familia_de

    rng = np.random.default_rng(7)
    rot = df[config.COLUNA_ROTULO].astype(str)
    familia_real = rot.map(lambda r: familia_de(r) or "?")

    # Familias com um unico rotulo nao tem como ser acertadas: segurar o rotulo
    # fora tira a familia inteira do indice. Ficam de fora da conta e sao
    # reportadas a parte — sao 6 das 16.
    rotulos_por_familia = familia_real.groupby(familia_real).size()
    n_rotulos_distintos = rot.groupby(familia_real).nunique()
    testaveis = set(n_rotulos_distintos[n_rotulos_distintos > 1].index)

    acertos = total = 0
    por_familia: dict[str, list[int]] = {}

    for rotulo in rot.unique():
        fam = familia_de(rotulo) or "?"
        if fam not in testaveis:
            continue

        fora = rot == rotulo
        treino = df[~fora]
        teste = df[fora]
        if teste.empty or treino.empty:
            continue

        # Amostra do rotulo segurado: rodar as 13 mil leituras de um rotulo nao
        # muda a conclusao e multiplica o tempo.
        if len(teste) > max_por_rotulo:
            teste = teste.iloc[rng.choice(len(teste), max_por_rotulo, replace=False)]

        idx = Indice(
            incluir_regime=indice.preparador.incluir_regime if indice else False
        ).construir(treino)

        for _, linha in teste.iterrows():
            evento = linha.drop(labels=[config.COLUNA_ROTULO], errors="ignore").to_dict()
            d = idx.consultar(evento, k=k)
            certo = int(d.familia == fam)
            acertos += certo
            total += 1
            por_familia.setdefault(fam, []).append(certo)

    # Baseline: chutar sempre a familia mais comum.
    maior = rotulos_por_familia.max() / len(df) if len(df) else 0.0

    detalhe = pd.DataFrame(
        [
            {"familia": f, "testes": len(v), "acertos": sum(v),
             "acuracia": round(sum(v) / len(v), 3)}
            for f, v in sorted(por_familia.items())
        ]
    )

    return {
        "acuracia": round(acertos / total, 3) if total else 0.0,
        "acertos": acertos,
        "testes": total,
        "baseline_maior_classe": round(float(maior), 3),
        "familias_testaveis": sorted(testaveis),
        "familias_nao_testaveis": sorted(set(n_rotulos_distintos.index) - testaveis),
        "por_familia": detalhe,
    }
