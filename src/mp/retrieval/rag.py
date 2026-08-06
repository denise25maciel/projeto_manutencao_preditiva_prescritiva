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
    """Um pedaco de manual recuperado: o texto, de onde veio e quanto casou."""

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
        """O endereco que a resposta tem de citar — e que o G5 procura no texto gerado.

        **Sem a pagina, de proposito:** o G5 casa este formato letra por letra, e
        cada numero a mais e um numero que o modelo pode errar. A pagina vem do
        banco e quem a mostra e a interface — ver `referencia`.
        """
        return f"{self.documento_id}, secao {self.numero}"

    @property
    def pagina(self) -> str:
        """A pagina como texto: `"4"`, `"4-5"`, ou vazio quando nao ha paginacao."""
        if self.pagina_inicio is None:
            return ""
        if self.pagina_fim and self.pagina_fim != self.pagina_inicio:
            return f"{self.pagina_inicio}-{self.pagina_fim}"
        return str(self.pagina_inicio)

    @property
    def referencia(self) -> str:
        """A citacao mais a pagina do PDF — a versao para o tecnico ler, nao para o G5."""
        pag = self.pagina
        return f"{self.citacao}" + (f" (pag. {pag})" if pag else "")


@dataclass
class Resultado:
    """O que uma busca devolve — inclusive quando nao acha nada.

    `motivo` e sempre preenchido: com a contagem quando deu certo, com a
    explicacao quando veio vazio. Recusar e caminho previsto, entao nao ha
    excecao a capturar; quem chama le `vazio` e mostra `motivo`.
    """

    familia: str | None
    documentos: list[str] = field(default_factory=list)
    trechos: list[Trecho] = field(default_factory=list)
    motivo: str = ""

    @property
    def vazio(self) -> bool:
        """Nenhum trecho recuperado. O `motivo` diz por que."""
        return not self.trechos


# --------------------------------------------------------------------------
# Indexacao
# --------------------------------------------------------------------------


def indexar(embedder=None, motor=None, verboso: bool = True) -> dict:
    """Transforma cada trecho num vetor e o grava no banco. Roda uma vez.

    Sao 168 trechos: segundos com TF-IDF, um pouco mais com o modelo neural.

    Grava tambem **qual** embedder gerou o vetor (`chunks.embedding_modelo`).
    Vetores de modelos diferentes nao sao comparaveis, e a busca confere isso
    antes de calcular qualquer distancia.
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
    """O nome do embedder que gerou os vetores gravados, ou `None` se nao ha vetor."""
    with sessao(motor) as s:
        return s.scalar(
            select(Chunk.embedding_modelo).where(Chunk.embedding.is_not(None)).limit(1)
        )


# Embedders ja ajustados, por nome. Evita refazer o ajuste a cada busca.
_PRONTOS: dict[str, object] = {}


def _embedder_pronto(embedder=None, motor=None):
    """Entrega o embedder pronto para codificar a pergunta — e guarda para a proxima.

    `ajustar(textos)` nao produz vetor: le o corpus e monta a **regra** que
    converte texto em vetor. No `TfidfLsa` essa regra e o vocabulario mais as
    direcoes do SVD (*Singular Value Decomposition*, a decomposicao que comprime
    o vetor esparso), e sem ela `codificar` levanta `RuntimeError`. No
    `Multilingue` o modelo ja vem treinado e `ajustar` so carrega os pesos.

    "Pronto" e **ajustado nos mesmos trechos da indexacao**, na mesma ordem. Os
    vetores no banco nasceram desse ajuste; a pergunta so pode ser comparada com
    eles se passar pelo mesmo. Vocabulario ou eixos diferentes = outro espaco, e
    cosseno entre espacos distintos e numero sem sentido. Por isso ajusta no
    corpus completo, nunca no subconjunto filtrado — isso ja custou um bug de 3
    trechos gerando 2 dimensoes contra as 167 gravadas.

    O ajuste nao vai para o disco (o banco guarda os vetores, nao o embedder),
    entao cada processo refaz. Como depende so do corpus, da para adiantar: a UI
    aquece na abertura da tela e `indexar()` ja deixa o seu em `_PRONTOS`.
    """
    embedder = embedder or emb.criar("auto")
    if embedder.nome in _PRONTOS:
        return _PRONTOS[embedder.nome]

    with sessao(motor) as s:
        textos = list(s.scalars(select(Chunk.texto).order_by(Chunk.id)))

    # Aqui o embedder aprende o espaco: vocabulario + eixos do SVD, no caso do
    # TF-IDF; carga dos pesos, no caso do modelo neural.
    embedder.ajustar(textos)
    _PRONTOS[embedder.nome] = embedder
    return embedder


def limpar_cache_embedder() -> None:
    """Esquece os embedders ja ajustados.

    Necessario quando os trechos mudam sem passar por `indexar` — que ja
    atualiza o cache com o ajuste novo. Sem isto, a busca seguiria usando um
    ajuste que nao corresponde mais aos textos do banco.
    """
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
    documentos: list[str] | None = None,
) -> Resultado:
    """Os `k` trechos mais parecidos com a pergunta, **dentro** do manual da familia.

    Os dois estagios do topo do modulo: o `SELECT` reduz os 168 trechos aos do
    manual certo, e o cosseno roda so nesse punhado.

    `campos` restringe o tipo de secao (`CAMPOS_PRESCRITIVOS` para "o que
    fazer"); nao havendo nenhuma daquele tipo, procura no documento inteiro
    antes de desistir.

    Familia sem manual volta vazia com o motivo — e o **G3**, o guardrail que
    exige documento no catalogo, e ele acontece aqui, antes de qualquer conta.

    **`documentos` pula o primeiro estagio**, e existe para quem ja travou o
    manual. Numa sessao aberta por texto o documento e o que foi fixado; ir dele
    para a familia e da familia de volta para o documento e um desvio que so
    fecha porque hoje toda familia tem um documento so. No dia em que uma delas
    tiver dois, o desvio devolve um manual a mais e a busca sai de dentro do que
    foi travado. Quem sabe qual e o manual passa o manual.
    """
    docs = list(documentos) if documentos is not None else None

    if docs is None:
        if familia is None:
            return Resultado(None, motivo="Rotulo fora do catalogo.")
        docs = [d["id"] for d in documentos_de(familia)]

    if not docs:
        return Resultado(
            familia,
            motivo=(
                f"Sem documentacao para '{familia}' — registre um documento."
                if familia else "Nenhum manual fixado para esta conversa."
            ),
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
    """`buscar` limitada as secoes que respondem "o que fazer".

    E a busca do turno prescritivo: correcao, validacao e criterios de aceitacao
    na frente de qualquer secao descritiva.

    Aceita `documentos=` pelo `**kwargs`, com o mesmo sentido que tem la.
    """
    return buscar(pergunta, familia, k=k, campos=CAMPOS_PRESCRITIVOS, **kwargs)


def _trecho_de(dado: dict, score: float) -> Trecho:
    """Converte a linha crua do banco num `Trecho`, com o score ja calculado."""
    return Trecho(
        documento_id=dado["documento_id"], numero=dado["numero"],
        titulo=dado["titulo"], campo=dado["campo"], texto=dado["texto"],
        score=score,
        pagina_inicio=dado["pagina_inicio"], pagina_fim=dado["pagina_fim"],
    )


def _todos_os_chunks(motor=None) -> tuple[list[dict], str]:
    """Carrega os seis manuais inteiros, sem filtro. Devolve `(dados, erro)`.

    `erro` vem vazio quando deu certo. Um lugar so, porque a busca livre e a
    busca por sintomas partem exatamente do mesmo conjunto.
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
    """Busca nos seis manuais de uma vez, sem filtro de familia.

    Para o tecnico que descreve o problema por escrito: sem falha identificada
    nao ha familia para filtrar, entao sao os trechos que apontam o documento.

    **Pular o estagio 1 enfraquece a garantia.** Semelhanca nunca volta vazia:
    ate pergunta de outro assunto recebe o trecho menos diferente de algum
    manual. Resta so o **G4**, o score minimo — por isso quem chama confere o
    score, e a tela mostra o numero em vez de esconder.
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
    """O documento mais votado pelos trechos. Devolve `(documento, peso)`.

    Cada trecho vota com o proprio score, nao com uma unidade: tres trechos
    fracos do Doc5 nao devem vencer um trecho forte do Doc2.
    """
    if resultado.vazio:
        return None, 0.0

    peso: dict[str, float] = {}
    for t in resultado.trechos:
        peso[t.documento_id] = peso.get(t.documento_id, 0.0) + max(t.score, 0.0)

    melhor = max(peso.items(), key=lambda kv: kv[1])
    return melhor[0], round(melhor[1], 4)


def ranking_documentos(resultado: Resultado) -> list[tuple[str, float]]:
    """A votacao inteira, do maior peso para o menor.

    `documento_predominante` e um `max` — sempre acha um vencedor, mesmo num
    empate tecnico. A lista completa e o que permite ver a diferenca entre o
    primeiro e o segundo e decidir se ha duvida a resolver com o tecnico.
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
    """Busca livre com varios sintomas: cada um vira uma consulta, e vale o **maior** score.

    Numa frase so, o vetor cairia perto da media dos sintomas e o segundo — em
    geral o que discrimina — seria diluido pelo primeiro. Detalhar mais pioraria
    a busca, o oposto do que se quer.

    Pelo maior score, o trecho que responde forte a *um* sintoma continua
    valendo. E como um manual e escrito: a secao de temperatura fala de
    temperatura, nao de vibracao.
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

    matriz = np.vstack([d["vetor"] for d in dados])          # (n_chunks, 384)

    # Um sintoma de cada vez contra TODOS os trechos. `matriz @ v` multiplica as
    # 384 componentes uma a uma e soma: sobra um numero por trecho. Como os
    # vetores sao unitarios, esse numero e o cosseno — de −1 (oposto) a +1
    # (texto identico), com 0 significando "nada em comum". Na pratica fica
    # entre 0 e 0,7: pergunta curta nunca e parafrase de uma secao inteira.
    todos = np.vstack([matriz @ v for v in embedder.codificar(consultas)])

    # (n_sintomas, n_chunks): uma linha por sintoma, uma coluna por trecho.
    # `axis=0` percorre a coluna, entao guarda o melhor sintoma de cada trecho.
    # Media diluiria: a secao "Mancal Aquecido" faz 0,71 no sintoma de
    # temperatura e 0,12 no de vibracao, e a media (0,41) a derrubaria no
    # ranking. Cada secao de manual trata de um assunto so.
    scores = todos.max(axis=0)

    # `argsort` devolve indices, nao valores, e em ordem crescente: `[::-1]`
    # inverte e `[:k]` corta nos melhores. Os indices apontam para `dados`.
    ordem = np.argsort(scores)[::-1][:k]
    trechos = [_trecho_de(dados[i], float(scores[i])) for i in ordem]

    # `dict.fromkeys` = conjunto que preserva a ordem; `set` embaralharia. Isto
    # e so a lista de documentos que apareceram — quem RANQUEIA os candidatos e
    # `ranking_documentos`, que soma os scores por documento.
    docs = list(dict.fromkeys(t.documento_id for t in trechos))
    return Resultado(
        None, docs, trechos,
        motivo=f"{len(trechos)} trecho(s) em {', '.join(docs)}, "
               f"a partir de {len(consultas)} sintoma(s).",
    )


def secoes_por_campo(
    documentos: list[str], motor=None, campos: tuple[str, ...] = CAMPOS_SINTOMA
) -> list[Trecho]:
    """Todas as secoes de um tipo, nos documentos dados. Por padrao, as de sintoma.

    E um `SELECT`, nao uma busca por semelhanca: nao ha pergunta, score nem `k` —
    vem tudo o que existe daquele tipo, em ordem estavel.

    E a materia-prima da pergunta de investigacao. O modelo redige em cima
    **disto**, nunca do que ele sabe sobre manutencao.
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
    """Os trechos em DataFrame, para a UI e os notebooks.

    Resultado vazio devolve as mesmas colunas, sem linha: quem mostra a tabela
    nao precisa tratar o caso a parte.
    """
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
