"""Busca do trecho certo do procedimento.

A ordem das duas etapas e a decisao central do projeto:

    1. filtro por familia -> documento    consulta exata
    2. busca por significado, so ali      semelhanca

**Por que nao o contrario.** Uma busca por semelhanca sempre devolve alguma
coisa. Se ela viesse primeiro, varreria os 6 manuais e entregaria o trecho "menos
diferente" — ainda que de outro defeito, ainda que a falha nao tenha manual
nenhum. O sistema responderia com fonte e confianca sobre algo sem base.

Filtrando antes, o sistema so procura dentro do manual certo. Se a familia nao
tem manual, nao ha subconjunto e nao ha o que buscar: o **G3** encerra antes.

E o principio 4 do projeto, em codigo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select

from mp import config
from mp.db.models import Chunk, Documento
from mp.db.session import sessao
from mp.retrieval import embeddings as emb
from mp.retrieval.catalog import documentos_de

# Tipos de secao que respondem "o que fazer". A pergunta prescritiva prioriza
# estes; os demais entram so se sobrar espaco.
CAMPOS_PRESCRITIVOS = ("correcao", "validacao", "criterios_aceitacao")


@dataclass
class Trecho:
    """Um pedaco de manual recuperado, com o endereco para citar."""

    documento_id: str
    numero: str
    titulo: str
    campo: str | None
    texto: str
    score: float

    @property
    def citacao(self) -> str:
        """O endereco que a resposta tera de citar — e que o G5 vai conferir."""
        return f"{self.documento_id}, secao {self.numero}"


@dataclass
class Resultado:
    familia: str | None
    documentos: list[str] = field(default_factory=list)
    trechos: list[Trecho] = field(default_factory=list)
    motivo: str = ""

    @property
    def vazio(self) -> bool:
        return not self.trechos


# --------------------------------------------------------------------------
# Indexacao
# --------------------------------------------------------------------------


def indexar(embedder=None, motor=None, verboso: bool = True) -> dict:
    """Calcula e grava o embedding de cada trecho.

    Roda uma vez. Sao 168 trechos — segundos com TF-IDF, um pouco mais com o
    modelo neural.

    O embedder e gravado junto (`chunks.embedding_modelo`): comparar vetores de
    modelos diferentes nao significa nada, entao a busca confere isso depois.
    """
    embedder = embedder or emb.criar("auto")

    with sessao(motor) as s:
        chunks = list(s.scalars(select(Chunk).order_by(Chunk.id)))
        if not chunks:
            return {"chunks": 0, "modelo": embedder.nome, "dimensao": 0}

        textos = [c.texto for c in chunks]
        embedder.ajustar(textos)
        vetores = embedder.codificar(textos)

        for chunk, vetor in zip(chunks, vetores):
            chunk.embedding = emb.para_bytes(vetor)
            chunk.embedding_modelo = embedder.nome

    # O embedder recem-ajustado e exatamente o que a busca precisa; guardamos
    # para nao refazer o ajuste na primeira consulta.
    _PRONTOS[embedder.nome] = embedder

    resultado = {
        "chunks": len(chunks),
        "modelo": embedder.nome,
        "dimensao": int(vetores.shape[1]),
    }
    if verboso:
        print(f"{resultado['chunks']} trechos indexados com {resultado['modelo']} "
              f"({resultado['dimensao']} dimensoes)")
    return resultado


def modelo_indexado(motor=None) -> str | None:
    """Qual embedder gerou os vetores que estao no banco."""
    with sessao(motor) as s:
        return s.scalar(
            select(Chunk.embedding_modelo).where(Chunk.embedding.is_not(None)).limit(1)
        )


# Embedders ja ajustados, por nome. Evita refazer o ajuste a cada busca.
_PRONTOS: dict[str, object] = {}


def _embedder_pronto(embedder=None, motor=None):
    """Devolve o embedder ajustado no **corpus completo**.

    Este detalhe e critico e ja custou um bug. O `TfidfLsa` aprende o vocabulario
    e as direcoes do SVD a partir do corpus — ajusta-lo no subconjunto filtrado
    de uma busca cria um espaco vetorial **diferente** do usado na indexacao, e
    comparar vetores dos dois nao significa nada. (No caso, nem rodava: 3 trechos
    davam 2 dimensoes contra as 167 gravadas.)

    A solucao e ajustar sempre nos mesmos 168 trechos, na mesma ordem
    (`order_by(Chunk.id)`, igual a `indexar`). O resultado e deterministico.

    O embedder neural nao precisa de nada disso — ja vem treinado, e `ajustar` e
    um no-op. A funcao serve aos dois pela mesma interface.
    """
    embedder = embedder or emb.criar("auto")
    if embedder.nome in _PRONTOS:
        return _PRONTOS[embedder.nome]

    with sessao(motor) as s:
        textos = list(s.scalars(select(Chunk.texto).order_by(Chunk.id)))

    embedder.ajustar(textos)
    _PRONTOS[embedder.nome] = embedder
    return embedder


def limpar_cache_embedder() -> None:
    """Esquece os embedders ajustados. Use depois de reindexar."""
    _PRONTOS.clear()


# --------------------------------------------------------------------------
# Busca
# --------------------------------------------------------------------------


def buscar(
    pergunta: str,
    familia: str | None,
    k: int = 5,
    embedder=None,
    campos: tuple[str, ...] | None = None,
    motor=None,
) -> Resultado:
    """Os `k` trechos mais parecidos com a pergunta, **dentro** do manual da familia.

    `campos` restringe o tipo de secao — para a pergunta prescritiva, use
    `CAMPOS_PRESCRITIVOS`.

    Devolve `Resultado` vazio, com motivo, quando a familia nao tem manual. Nao
    levanta excecao: recusar e um caminho previsto, nao um erro.
    """
    if familia is None:
        return Resultado(None, motivo="Rotulo fora do catalogo.")

    docs = [d["id"] for d in documentos_de(familia)]
    if not docs:
        return Resultado(
            familia, motivo=f"Sem documentacao para '{familia}' — registre um documento."
        )

    # --- estagio 1: filtro exato -------------------------------------------
    with sessao(motor) as s:
        consulta = select(Chunk).where(
            Chunk.documento_id.in_(docs), Chunk.embedding.is_not(None)
        )
        if campos:
            consulta = consulta.where(Chunk.campo.in_(campos))
        candidatos = list(s.scalars(consulta))

        if not candidatos and campos:
            # Sem secao do tipo pedido, tenta o documento inteiro antes de desistir.
            candidatos = list(
                s.scalars(
                    select(Chunk).where(
                        Chunk.documento_id.in_(docs), Chunk.embedding.is_not(None)
                    )
                )
            )

        if not candidatos:
            return Resultado(
                familia, docs,
                motivo="Os trechos ainda nao foram indexados. Rode `indexar()`.",
            )

        dados = [
            {
                "documento_id": c.documento_id, "numero": c.numero, "titulo": c.titulo,
                "campo": c.campo, "texto": c.texto,
                "vetor": emb.de_bytes(c.embedding), "modelo": c.embedding_modelo,
            }
            for c in candidatos
        ]

    # --- estagio 2: semelhanca dentro do subconjunto -----------------------
    embedder = embedder or emb.criar("auto")
    modelo_no_banco = dados[0]["modelo"]
    if modelo_no_banco != embedder.nome:
        return Resultado(
            familia, docs,
            motivo=(
                f"Os trechos foram indexados com '{modelo_no_banco}' e a busca esta "
                f"usando '{embedder.nome}'. Vetores de modelos diferentes nao sao "
                "comparaveis — reindexe."
            ),
        )

    # Ajustado no corpus COMPLETO, nunca no subconjunto — ver `_embedder_pronto`.
    embedder = _embedder_pronto(embedder, motor)
    consulta_vetor = embedder.codificar([pergunta])[0]

    matriz = np.vstack([d["vetor"] for d in dados])
    # Vetores normalizados: produto escalar = cosseno.
    scores = matriz @ consulta_vetor
    ordem = np.argsort(scores)[::-1][:k]

    trechos = [
        Trecho(
            documento_id=dados[i]["documento_id"], numero=dados[i]["numero"],
            titulo=dados[i]["titulo"], campo=dados[i]["campo"],
            texto=dados[i]["texto"], score=float(scores[i]),
        )
        for i in ordem
    ]
    return Resultado(familia, docs, trechos,
                     motivo=f"{len(trechos)} trecho(s) em {', '.join(docs)}.")


def buscar_prescritivo(pergunta: str, familia: str | None, k: int = 5, **kwargs):
    """Busca priorizando as secoes que respondem 'o que fazer'."""
    return buscar(pergunta, familia, k=k, campos=CAMPOS_PRESCRITIVOS, **kwargs)


def como_tabela(resultado: Resultado) -> pd.DataFrame:
    """Os trechos em DataFrame, para a UI e os notebooks."""
    if resultado.vazio:
        return pd.DataFrame(
            columns=["citacao", "documento", "secao", "titulo", "campo", "score", "texto"]
        )
    return pd.DataFrame(
        [
            {
                "citacao": t.citacao, "documento": t.documento_id, "secao": t.numero,
                "titulo": t.titulo, "campo": t.campo, "score": round(t.score, 4),
                "texto": t.texto,
            }
            for t in resultado.trechos
        ]
    )
