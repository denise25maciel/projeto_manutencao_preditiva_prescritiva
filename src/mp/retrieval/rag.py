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

# As secoes que descrevem "como o defeito se manifesta". Sao a materia-prima da
# pergunta de investigacao: para separar dois candidatos, o sistema pergunta
# sobre o que os manuais deles descrevem, nunca sobre o que o modelo imagina.
CAMPOS_SINTOMA = ("sintomas", "causas", "indicadores", "diagnostico")


@dataclass
class Trecho:
    """Um pedaco de manual recuperado, com o endereco para citar."""

    documento_id: str
    numero: str
    titulo: str
    campo: str | None
    texto: str
    score: float
    pagina_inicio: int | None = None
    pagina_fim: int | None = None

    @property
    def citacao(self) -> str:
        """O endereco que a resposta tera de citar — e que o G5 vai conferir.

        **Sem a pagina, de proposito.** O G5 casa exatamente este formato no
        texto gerado; se a pagina entrasse aqui, o modelo teria de reproduzi-la
        e poderia errar o numero. A pagina e metadado do banco e e a interface
        que a acrescenta — ver `referencia`.
        """
        return f"{self.documento_id}, secao {self.numero}"

    @property
    def pagina(self) -> str:
        """A pagina como texto: "4", "4-5" ou vazio quando nao ha paginacao."""
        if self.pagina_inicio is None:
            return ""
        if self.pagina_fim and self.pagina_fim != self.pagina_inicio:
            return f"{self.pagina_inicio}-{self.pagina_fim}"
        return str(self.pagina_inicio)

    @property
    def referencia(self) -> str:
        """O endereco completo para mostrar ao tecnico, com a pagina do PDF."""
        pag = self.pagina
        return f"{self.citacao}" + (f" (pag. {pag})" if pag else "")


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
                "pagina_inicio": c.pagina_inicio, "pagina_fim": c.pagina_fim,
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
            pagina_inicio=dados[i]["pagina_inicio"],
            pagina_fim=dados[i]["pagina_fim"],
        )
        for i in ordem
    ]
    return Resultado(familia, docs, trechos,
                     motivo=f"{len(trechos)} trecho(s) em {', '.join(docs)}.")


def buscar_prescritivo(pergunta: str, familia: str | None, k: int = 5, **kwargs):
    """Busca priorizando as secoes que respondem 'o que fazer'."""
    return buscar(pergunta, familia, k=k, campos=CAMPOS_PRESCRITIVOS, **kwargs)


def _trecho_de(dado: dict, score: float) -> Trecho:
    """Monta o `Trecho` a partir da linha carregada do banco."""
    return Trecho(
        documento_id=dado["documento_id"], numero=dado["numero"],
        titulo=dado["titulo"], campo=dado["campo"], texto=dado["texto"],
        score=score,
        pagina_inicio=dado["pagina_inicio"], pagina_fim=dado["pagina_fim"],
    )


def _todos_os_chunks(motor=None) -> tuple[list[dict], str]:
    """Todos os trechos indexados. Devolve `(dados, erro)` — erro vazio se deu certo.

    Um lugar so para carregar, porque a busca livre e a busca por sintomas usam
    exatamente o mesmo conjunto: os seis manuais inteiros, sem filtro de familia.
    """
    with sessao(motor) as s:
        candidatos = list(
            s.scalars(select(Chunk).where(Chunk.embedding.is_not(None)))
        )
        if not candidatos:
            return [], "Os trechos ainda nao foram indexados. Rode `indexar()`."

        return [
            {
                "documento_id": c.documento_id, "numero": c.numero, "titulo": c.titulo,
                "campo": c.campo, "texto": c.texto,
                "pagina_inicio": c.pagina_inicio, "pagina_fim": c.pagina_fim,
                "vetor": emb.de_bytes(c.embedding), "modelo": c.embedding_modelo,
            }
            for c in candidatos
        ], ""


def buscar_livre(pergunta: str, k: int = 8, embedder=None, motor=None) -> Resultado:
    """Busca em **todos** os manuais, sem filtro de familia.

    Serve ao caso em que o tecnico chega descrevendo o problema por escrito, sem
    evento de sensor: ninguem sabe ainda qual e a falha, entao nao ha familia
    para filtrar. Sao os proprios trechos que apontam o documento.

    **Isto contorna o estagio 1 e enfraquece a garantia.** A busca por semelhanca
    nunca volta vazia: mesmo uma pergunta de outro assunto recebe o trecho "menos
    diferente" de algum manual. Com o filtro por familia essa possibilidade nem
    existia — sem manual, nao havia subconjunto onde procurar.

    Aqui a unica trava e o **G4**, o score minimo. Por isso quem chama tem de
    conferir o score antes de tratar o resultado como resposta, e a interface
    mostra o numero em vez de escondê-lo.

    Depois de escolhido o documento, a conversa volta ao caminho normal: o manual
    e fixado e as perguntas seguintes sao filtradas dentro dele.
    """
    dados, erro = _todos_os_chunks(motor)
    if erro:
        return Resultado(None, motivo=erro)

    embedder = _embedder_pronto(embedder, motor)
    if dados[0]["modelo"] != embedder.nome:
        return Resultado(
            None,
            motivo=(
                f"Os trechos foram indexados com '{dados[0]['modelo']}' e a busca "
                f"esta usando '{embedder.nome}'. Reindexe."
            ),
        )

    consulta_vetor = embedder.codificar([pergunta])[0]
    matriz = np.vstack([d["vetor"] for d in dados])
    scores = matriz @ consulta_vetor
    ordem = np.argsort(scores)[::-1][:k]

    trechos = [_trecho_de(dados[i], float(scores[i])) for i in ordem]
    docs = list(dict.fromkeys(t.documento_id for t in trechos))
    return Resultado(None, docs, trechos,
                     motivo=f"{len(trechos)} trecho(s) em {', '.join(docs)}.")


def documento_predominante(resultado: Resultado) -> tuple[str | None, float]:
    """Qual documento os trechos apontam, e com que peso.

    O voto e ponderado pelo score: tres trechos fracos do Doc5 nao devem vencer
    um trecho forte do Doc2. Devolve `(documento, soma_dos_scores)`.
    """
    if resultado.vazio:
        return None, 0.0

    peso: dict[str, float] = {}
    for t in resultado.trechos:
        peso[t.documento_id] = peso.get(t.documento_id, 0.0) + max(t.score, 0.0)

    melhor = max(peso.items(), key=lambda kv: kv[1])
    return melhor[0], round(melhor[1], 4)


def ranking_documentos(resultado: Resultado) -> list[tuple[str, float]]:
    """Todos os documentos com seu peso, do maior para o menor.

    `documento_predominante` devolve so o vencedor — e um `max`, que sempre
    encontra um. Este devolve a **lista**, que e o que permite perguntar a coisa
    seguinte: *o primeiro ganhou de longe, ou por um fio?*
    """
    if resultado.vazio:
        return []

    peso: dict[str, float] = {}
    for t in resultado.trechos:
        peso[t.documento_id] = peso.get(t.documento_id, 0.0) + max(t.score, 0.0)

    return sorted(
        ((d, round(p, 4)) for d, p in peso.items()),
        key=lambda kv: kv[1], reverse=True,
    )


def buscar_por_sintomas(
    sintomas: list[str], k: int = 8, embedder=None, motor=None
) -> Resultado:
    """Busca livre com **varios sintomas**, cada um como uma consulta propria.

    A alternativa obvia — juntar tudo numa frase so e embedar de uma vez — tem
    um defeito conhecido: o vetor resultante e mais ou menos a **media** dos
    sintomas. O segundo sintoma, que costuma ser o que discrimina, e diluido
    pelo primeiro. Descrever mais acabaria piorando a busca, que e o oposto do
    que o loop de investigacao quer.

    Aqui cada sintoma e codificado sozinho e o score de um trecho e o **maior**
    entre eles. Um trecho que responde fortemente a *um* sintoma continua
    valendo, mesmo que ignore os outros — que e como um manual funciona: a
    secao de temperatura fala de temperatura, nao de vibracao.
    """
    consultas = [s.strip() for s in sintomas if s and s.strip()]
    if not consultas:
        return Resultado(None, motivo="Nenhum sintoma informado.")

    dados, erro = _todos_os_chunks(motor)
    if erro:
        return Resultado(None, motivo=erro)

    embedder = _embedder_pronto(embedder, motor)
    if dados[0]["modelo"] != embedder.nome:
        return Resultado(
            None,
            motivo=(
                f"Os trechos foram indexados com '{dados[0]['modelo']}' e a busca "
                f"esta usando '{embedder.nome}'. Reindexe."
            ),
        )

    matriz = np.vstack([d["vetor"] for d in dados])
    # (n_sintomas, n_chunks) -> o maior por chunk, ao longo dos sintomas.
    todos = np.vstack([matriz @ v for v in embedder.codificar(consultas)])
    scores = todos.max(axis=0)

    ordem = np.argsort(scores)[::-1][:k]
    trechos = [_trecho_de(dados[i], float(scores[i])) for i in ordem]
    docs = list(dict.fromkeys(t.documento_id for t in trechos))
    return Resultado(
        None, docs, trechos,
        motivo=f"{len(trechos)} trecho(s) em {', '.join(docs)}, "
               f"a partir de {len(consultas)} sintoma(s).",
    )


def secoes_de_sintomas(
    documentos: list[str], motor=None, campos: tuple[str, ...] = CAMPOS_SINTOMA
) -> list[Trecho]:
    """As secoes que descrevem como cada defeito se manifesta.

    E um `SELECT`, nao uma busca por semelhanca. E o que a pergunta de
    investigacao usa como materia-prima: o modelo redige em cima **disto**, e
    nao de conhecimento proprio sobre manutencao.
    """
    if not documentos:
        return []

    with sessao(motor) as s:
        achados = list(
            s.scalars(
                select(Chunk)
                .where(Chunk.documento_id.in_(documentos), Chunk.campo.in_(campos))
                .order_by(Chunk.documento_id, Chunk.numero)
            )
        )
        return [
            Trecho(
                documento_id=c.documento_id, numero=c.numero, titulo=c.titulo,
                campo=c.campo, texto=c.texto, score=0.0,
                pagina_inicio=c.pagina_inicio, pagina_fim=c.pagina_fim,
            )
            for c in achados
        ]


def como_tabela(resultado: Resultado) -> pd.DataFrame:
    """Os trechos em DataFrame, para a UI e os notebooks."""
    if resultado.vazio:
        return pd.DataFrame(
            columns=["citacao", "documento", "secao", "pagina", "titulo", "campo",
                     "score", "texto"]
        )
    return pd.DataFrame(
        [
            {
                "citacao": t.citacao, "documento": t.documento_id, "secao": t.numero,
                "pagina": t.pagina or "—", "titulo": t.titulo, "campo": t.campo,
                "score": round(t.score, 4), "texto": t.texto,
            }
            for t in resultado.trechos
        ]
    )
