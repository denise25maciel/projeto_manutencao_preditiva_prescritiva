"""Transformar texto em numeros, para a busca por significado.

Um **embedding** e uma lista de numeros que representa o sentido de um trecho.
Trechos que falam da mesma coisa ficam proximos nessa representacao, mesmo sem
repetir palavra — e o que permite "vibracao alta no eixo" encontrar uma secao que
diz "amplitude elevada radial".

Dois embedders, com a mesma interface
-------------------------------------
`TfidfLsa`      — conta palavras e comprime com SVD. Sem dependencia pesada, roda
                  em segundos. Acha sinonimo so quando as palavras aparecem juntas
                  nos mesmos trechos; nao entende sentido de verdade.
`Multilingue`   — modelo neural treinado em varias linguas, incluindo portugues.
                  E o que o projeto pede. Precisa de `sentence-transformers`.

Os dois devolvem vetores L2-normalizados, entao a similaridade por cosseno vira
um produto escalar simples. Trocar um pelo outro nao muda nenhuma linha da busca.

Por que guardar como BLOB
-------------------------
Sao 168 trechos. Um indice vetorial dedicado seria complexidade sem retorno: o
vetor vira bytes na coluna `chunks.embedding`, e a comparacao roda em memoria com
numpy. Migrar para pgvector depois e trocar a consulta, nao o modelo.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Modelo padrao quando `sentence-transformers` esta disponivel. Multilingue e
# pequeno o bastante para rodar em CPU — os documentos sao 168 trechos curtos.
MODELO_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    """Contrato minimo. Quem consome a busca nao precisa saber qual e."""

    nome: str
    dimensao: int

    def ajustar(self, textos: list[str]) -> "Embedder":
        """Aprende, a partir do corpus, a regra que converte texto em vetor.

        Nao devolve vetor e nao grava nada — deixa o objeto em condicao de
        responder a `codificar`, e retorna a si mesmo para encadear. Cada
        implementacao aproveita `textos` de um jeito: o `TfidfLsa` tira dali o
        vocabulario e os eixos do SVD; o `Multilingue` ignora, porque ja vem
        treinado, e so carrega os pesos.
        """
        ...

    def codificar(self, textos: list[str]) -> np.ndarray:
        """Textos -> matriz (um vetor L2-normalizado por linha). Exige `ajustar` antes."""
        ...


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    """L2 por linha. Com vetores unitarios, cosseno = produto escalar."""
    norma = np.linalg.norm(matriz, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    return (matriz / norma).astype(np.float32)


class TfidfLsa:
    """Contagem de palavras comprimida por SVD — o classico LSA.

    TF-IDF (*term frequency-inverse document frequency*) da um vetor enorme e
    esparso, com uma posicao por palavra. O SVD (*singular value decomposition*)
    comprime para poucas centenas de dimensoes densas, e nessa compressao
    palavras que sempre aparecem juntas ficam proximas — o mais perto de
    "significado" sem modelo neural. Os dois juntos sao o LSA, *latent semantic
    analysis*.

    **Precisa ver o corpus antes** (`ajustar`): dele saem o vocabulario e as
    direcoes do SVD. Um embedder neural nao precisa, ja vem treinado.
    """

    def __init__(self, dimensao: int = 256, min_df: int = 1):
        self.nome = f"tfidf-lsa-{dimensao}"
        self.dimensao = dimensao
        self._min_df = min_df
        self._vetorizador = None
        self._svd = None

    def ajustar(self, textos: list[str]) -> "TfidfLsa":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vetorizador = TfidfVectorizer(
            lowercase=True,
            # Pares de palavras alem das isoladas: "pista externa" vale mais que
            # "pista" e "externa" soltas.
            ngram_range=(1, 2),
            min_df=self._min_df,
            max_df=0.9,
            sublinear_tf=True,
        )
        X = self._vetorizador.fit_transform(textos)

        # O SVD nao pode pedir mais dimensoes do que o corpus tem.
        k = min(self.dimensao, min(X.shape) - 1)
        self._svd = TruncatedSVD(n_components=max(k, 2), random_state=0)
        self._svd.fit(X)
        self.dimensao = self._svd.n_components
        return self

    def codificar(self, textos: list[str]) -> np.ndarray:
        if self._vetorizador is None or self._svd is None:
            raise RuntimeError("Chame `ajustar` com o corpus antes de codificar.")
        X = self._vetorizador.transform(textos)
        return _normalizar(self._svd.transform(X))


class Multilingue:
    """Modelo neural multilingue. E o que o projeto pede.

    Ja vem treinado, entao `ajustar` nao faz nada — existe so para a interface
    ser a mesma do `TfidfLsa`.
    """

    def __init__(self, modelo: str = MODELO_PADRAO):
        self.nome = modelo.split("/")[-1]
        self._nome_modelo = modelo
        self._modelo = None
        self.dimensao = 0

    def disponivel(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def ajustar(self, textos: list[str]) -> "Multilingue":
        self._carregar()
        return self

    def _carregar(self):
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer

            self._modelo = SentenceTransformer(self._nome_modelo)
            self.dimensao = int(self._modelo.get_sentence_embedding_dimension())
        return self._modelo

    def codificar(self, textos: list[str]) -> np.ndarray:
        modelo = self._carregar()
        vetores = modelo.encode(
            textos, convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vetores.astype(np.float32)


def criar(nome: str = "auto", **kwargs) -> Embedder:
    """Devolve o embedder pedido.

    `auto` usa o multilingue se `sentence-transformers` estiver instalado, e cai
    no TF-IDF+LSA caso contrario. A busca funciona nos dois casos — muda a
    qualidade, nao o funcionamento.
    """
    if nome == "tfidf":
        return TfidfLsa(**kwargs)
    if nome in ("multilingue", "st"):
        return Multilingue(**kwargs)
    if nome == "auto":
        m = Multilingue()
        return m if m.disponivel() else TfidfLsa()
    raise ValueError(f"embedder desconhecido: {nome}")


# --------------------------------------------------------------------------
# BLOB
# --------------------------------------------------------------------------


def para_bytes(vetor: np.ndarray) -> bytes:
    """Vetor -> bytes, para a coluna `chunks.embedding`."""
    return np.asarray(vetor, dtype=np.float32).tobytes()


def de_bytes(dados: bytes, dimensao: int | None = None) -> np.ndarray:
    """Bytes -> vetor. `dimensao` so serve de conferencia."""
    vetor = np.frombuffer(dados, dtype=np.float32)
    if dimensao is not None and vetor.size != dimensao:
        raise ValueError(
            f"embedding com {vetor.size} numeros, esperado {dimensao}. "
            "O modelo mudou? Refaca os embeddings."
        )
    return vetor
