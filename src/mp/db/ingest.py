"""Popula o banco a partir do CSV bruto e dos manuais convertidos.

Um comando:

    python -m mp.db.ingest

O que ele faz, em ordem:

1. cria o esquema do zero
2. le o `banner.csv` e monta os eventos nas duas versoes
3. grava as leituras, com as duas colunas de evento
4. grava os eventos das duas versoes
5. grava os documentos e suas secoes

**Repetivel de proposito.** Cada execucao apaga e recria: rodar duas vezes produz
exatamente o mesmo banco. E o que permite ajustar uma regra de agrupamento e
regerar sem medo de duplicar.

Nao aplica os descartes de coluna nem a deduplicacao (pendencias P3 e P2). O banco
guarda o dado como veio; o que e decisao de modelagem fica para a Parte 3.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from mp import config
from mp.analysis import carregar
from mp.db.models import Chunk, Documento, Episodio, Leitura
from mp.db.session import criar_engine, criar_esquema, sessao
from mp.ingestion import (
    carregar_markdowns,
    coesao_eventos,
    construir_eventos,
    construir_eventos_por_rotulo,
)
from mp.segmentos import maior_buraco_interno

# Colunas de medida que vao para `readings`, na ordem do modelo.
COLUNAS_MEDIDA = [
    "z_rms_velocity_in_s", "z_rms_velocity_mm_s",
    "z_peak_velocity_in_s", "z_peak_velocity_mm_s",
    "z_rms_acceleration_g", "z_peak_acceleration_g",
    "z_high_freq_rms_accel_g", "z_peak_vel_comp_freq_hz",
    "z_kurtosis", "z_crest_factor",
    "x_rms_velocity_in_s", "x_rms_velocity_mm_s",
    "x_peak_velocity_in_s", "x_peak_velocity_mm_s",
    "x_rms_acceleration_g", "x_peak_acceleration_g",
    "x_high_freq_rms_accel_g", "x_peak_vel_comp_freq_hz",
    "x_kurtosis", "x_crest_factor",
    "temperature_c", "temperature_f",
]


def _familia_por_rotulo() -> dict:
    """Mapa rotulo -> familia, lido do catalogo curado."""
    from mp.retrieval import familia_de

    mapa = {}

    def resolver(rotulo):
        if rotulo not in mapa:
            mapa[rotulo] = familia_de(rotulo)
        return mapa[rotulo]

    return resolver


def preparar_eventos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta as duas versoes e devolve `(leituras, eventos)` prontos para gravar.

    `leituras` sai com as colunas `evento_a` e `evento_b`. Como as duas versoes
    ordenam o DataFrame de formas diferentes, o casamento e feito pelo `id`, que
    e a unica coluna estavel entre elas.
    """
    tempo, rotulo, ident = config.COLUNA_TEMPO, config.COLUNA_ROTULO, config.COLUNA_ID
    resolver_familia = _familia_por_rotulo()

    leituras_a, eventos_a = construir_eventos(df)
    leituras_b, eventos_b = construir_eventos_por_rotulo(df)

    coesao_a = coesao_eventos(leituras_a).set_index("evento")["dispersao"]
    coesao_b = coesao_eventos(leituras_b).set_index("evento")["dispersao"]
    buraco_a = maior_buraco_interno(leituras_a, leituras_a["evento"], tempo)
    buraco_b = maior_buraco_interno(leituras_b, leituras_b["evento"], tempo)

    # As duas versoes ordenam diferente; o `id` e a ponte entre elas.
    leituras = leituras_a.rename(columns={"evento": "evento_a"})
    leituras["evento_b"] = leituras[ident].map(
        leituras_b.set_index(ident)["evento"]
    )

    def tabela(eventos, versao, coesao, buracos):
        saida = eventos.rename(columns={"evento": "numero"}).copy()
        saida["versao"] = versao
        saida["familia"] = saida[rotulo].map(resolver_familia)
        saida["dispersao"] = saida["numero"].map(coesao)
        saida["maior_buraco_s"] = saida["numero"].map(buracos).fillna(0.0)
        if "rpm" not in saida.columns:
            saida["rpm"] = None
        return saida[
            ["versao", "numero", rotulo, "familia", "rpm", "n_leituras",
             "inicio", "fim", "duracao_s", "dispersao", "maior_buraco_s"]
        ]

    eventos = pd.concat(
        [tabela(eventos_a, "A", coesao_a, buraco_a),
         tabela(eventos_b, "B", coesao_b, buraco_b)],
        ignore_index=True,
    )
    return leituras, eventos


def gravar_leituras(motor, leituras: pd.DataFrame, lote: int = 20_000) -> int:
    """Grava `readings`. Em lotes porque sao 166 mil linhas."""
    colunas = [config.COLUNA_ID, config.COLUNA_TEMPO, config.COLUNA_ROTULO,
               "rpm", "evento_a", "evento_b", *COLUNAS_MEDIDA]
    recorte = leituras[colunas].rename(
        columns={config.COLUNA_TEMPO: "created_at", config.COLUNA_ROTULO: "fault"}
    )

    total = 0
    with sessao(motor) as s:
        for inicio in range(0, len(recorte), lote):
            bloco = recorte.iloc[inicio:inicio + lote]
            s.bulk_insert_mappings(Leitura, bloco.to_dict(orient="records"))
            total += len(bloco)
    return total


def gravar_eventos(motor, eventos: pd.DataFrame) -> int:
    registros = eventos.rename(columns={config.COLUNA_ROTULO: "fault"}).to_dict(
        orient="records"
    )
    with sessao(motor) as s:
        s.bulk_insert_mappings(Episodio, registros)
    return len(registros)


def gravar_documentos(motor, docs: list[dict]) -> tuple[int, int]:
    """Grava `documents` e `chunks`. O texto de cada secao sai do `.md`."""
    n_docs = n_chunks = 0

    with sessao(motor) as s:
        for d in docs:
            s.add(
                Documento(
                    id=d["documento"],
                    titulo=d["titulo"],
                    arquivo_md=d["arquivo"].name,
                    pdf_origem=f"{d['documento']}.pdf",
                    origem_texto=d["origem_texto"],
                    n_secoes=d["n_secoes"],
                )
            )
            n_docs += 1

            texto_md = d["arquivo"].read_text(encoding="utf-8")
            for secao, corpo in _fatiar(texto_md, d["secoes"]):
                s.add(
                    Chunk(
                        documento_id=d["documento"],
                        numero=secao["numero"],
                        titulo=secao["titulo"],
                        nivel=secao["nivel"],
                        campo=secao["campo"],
                        texto=corpo,
                        n_caracteres=len(corpo),
                    )
                )
                n_chunks += 1

    return n_docs, n_chunks


# Casa "## 1. Objetivo" e "### 4.1. Defeito na Pista Externa". O numero inteiro
# precisa ser capturado de uma vez: pegar so ate o primeiro ponto transformaria
# a subsecao 4.1 em "4", ela nao casaria com nenhuma secao e seria descartada.
_CABECALHO_MD = re.compile(r"^#{2,6}\s+(\d+(?:\.\d+)*)\.\s")


def _fatiar(texto_md: str, secoes: list[dict]):
    """Corta o Markdown nos titulos de secao e devolve `(secao, corpo)`.

    O corpo vai do titulo ate o proximo titulo — de qualquer nivel. Inclui o
    proprio titulo, para o trecho continuar legivel sozinho quando for citado.
    """
    linhas = texto_md.splitlines()

    marcas = []
    for i, linha in enumerate(linhas):
        if m := _CABECALHO_MD.match(linha):
            marcas.append((i, m.group(1)))

    por_numero = {s["numero"]: s for s in secoes}

    for pos, (inicio, numero) in enumerate(marcas):
        fim = marcas[pos + 1][0] if pos + 1 < len(marcas) else len(linhas)
        secao = por_numero.get(numero)
        if secao is None:
            continue
        corpo = "\n".join(linhas[inicio:fim]).strip()
        yield secao, corpo


def executar(caminho_db: str | Path | None = None, verboso: bool = True) -> dict:
    """Roda a ingestao inteira. Devolve o que foi gravado."""
    t0 = time.time()
    motor = criar_engine(caminho_db)

    def diga(msg):
        if verboso:
            print(msg, flush=True)

    diga("1/5  criando o esquema do zero...")
    criar_esquema(motor, apagar_antes=True)

    diga("2/5  lendo o banner.csv...")
    df = carregar()
    diga(f"     {len(df):,} leituras".replace(",", "."))

    diga("3/5  montando os eventos nas duas versoes...")
    leituras, eventos = preparar_eventos(df)
    n_a = int((eventos["versao"] == "A").sum())
    n_b = int((eventos["versao"] == "B").sum())
    diga(f"     versao A: {n_a} eventos | versao B: {n_b} eventos")

    diga("4/5  gravando leituras e eventos...")
    n_leituras = gravar_leituras(motor, leituras)
    n_eventos = gravar_eventos(motor, eventos)

    diga("5/5  gravando documentos e secoes...")
    docs = carregar_markdowns()
    if not docs:
        diga("     nenhum .md encontrado — rode a conversao dos PDFs antes")
        n_docs = n_chunks = 0
    else:
        n_docs, n_chunks = gravar_documentos(motor, docs)

    caminho = Path(caminho_db) if caminho_db else config.DB_PATH
    resultado = {
        "banco": str(caminho),
        "leituras": n_leituras,
        "eventos": n_eventos,
        "eventos_a": n_a,
        "eventos_b": n_b,
        "documentos": n_docs,
        "chunks": n_chunks,
        "segundos": round(time.time() - t0, 1),
        "tamanho_mb": round(caminho.stat().st_size / 1024**2, 1)
        if caminho.exists()
        else 0.0,
    }

    if verboso:
        print()
        print(f"pronto em {resultado['segundos']}s — {resultado['tamanho_mb']} MB")
        for chave in ("leituras", "eventos", "documentos", "chunks"):
            print(f"  {chave:12s} {resultado[chave]:>8,}".replace(",", "."))
        print(f"  banco        {resultado['banco']}")

    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Popula o banco a partir do CSV bruto e dos manuais."
    )
    parser.add_argument("--db", default=None, help="caminho do arquivo .db")
    parser.add_argument("--silencioso", action="store_true")
    args = parser.parse_args()
    executar(args.db, verboso=not args.silencioso)


if __name__ == "__main__":
    main()
