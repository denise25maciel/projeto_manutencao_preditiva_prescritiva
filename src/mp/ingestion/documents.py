"""Conversao dos PDFs de procedimento em Markdown.

Por que Markdown e nao texto puro: a Parte 4 faz chunking **por secao
numerada**, e o Markdown preserva essa hierarquia de forma legivel por humano
e trivial de reparsear. O `.md` intermediario tambem permite revisar a
extracao antes de indexar — se o chunk sair errado, da para ver onde.

Nada aqui interpreta o conteudo tecnico. O modulo separa secoes e classifica
titulos; o que cada procedimento diz e assunto do RAG.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mp import config

# --------------------------------------------------------------------------
# Extracao
# --------------------------------------------------------------------------


def _sem_acento(texto: str) -> str:
    """Remove acentos. Os padroes de `CAMPOS_CANONICOS` rodam sobre isso."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def extrair_texto(caminho_pdf: Path) -> tuple[str, str, list[int | None]]:
    """Extrai o texto de um PDF, sabendo de que pagina veio cada linha.

    Devolve `(texto, origem, paginas)`, onde `paginas[i]` e o numero da pagina
    (1-based) da linha `i` de `texto.splitlines()`, ou `None` quando nao da para
    saber. Origem e:

      - `pdf`      — camada de texto do proprio PDF
      - `sidecar`  — arquivo `.txt` ao lado, usado quando o PDF e digitalizado
      - `vazio`    — sem texto e sem sidecar; precisa de OCR

    **Por que rastrear a pagina.** A resposta cita "Doc2, secao 9.2", que e o
    endereco logico. Mas quem esta com o manual impresso na mao procura por
    pagina. Guardar as duas coisas custa uma lista de inteiros e evita que o
    numero da pagina precise ser adivinhado depois — ou, pior, gerado pelo
    modelo de linguagem.

    **Por que o sidecar existe:** um dos procedimentos veio escaneado, so com
    imagens e sem camada de texto. Rodar OCR exigiria o binario do Tesseract,
    que nao e resolvivel por `pip` e quebraria o "clone limpo". Em vez disso o
    conversor aceita uma transcricao manual em `data/raw/<nome>.txt`, e marca a
    origem no front matter para ninguem confundir com extracao automatica.
    Transcricao manual **nao tem pagina**: ali as paginas saem todas `None`, e o
    sistema diz "pagina nao disponivel" em vez de inventar uma.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(caminho_pdf))

    blocos, paginas = [], []
    for numero, pagina in enumerate(reader.pages, start=1):
        conteudo = pagina.extract_text() or ""
        blocos.append(conteudo)
        paginas.extend([numero] * len(conteudo.splitlines()))
        # O "\n" que emenda uma pagina na seguinte tambem vira uma linha.
        if numero < len(reader.pages):
            paginas.append(numero)

    texto = "\n".join(blocos)

    # Um PDF de procedimento tem milhares de caracteres. Menos que isso e
    # residuo de cabecalho: o conteudo esta em imagem.
    if len(texto.strip()) >= 500:
        # A contagem acima pode divergir por uma linha em PDFs que terminam sem
        # quebra; alinhamos pelo texto final, que e o que sera fatiado.
        n = len(texto.splitlines())
        paginas = (paginas + [paginas[-1] if paginas else None] * n)[:n]
        return texto, "pdf", paginas

    sidecar = caminho_pdf.with_suffix(".txt")
    if sidecar.exists():
        conteudo = sidecar.read_text(encoding="utf-8")
        return conteudo, "sidecar", [None] * len(conteudo.splitlines())

    return "", "vazio", []


# --------------------------------------------------------------------------
# Estrutura
# --------------------------------------------------------------------------

# Casa "1. Objetivo", "2.1 Desalinhamento Paralelo", "4.3 Defeito nos Rolantes".
_CABECALHO = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")


@dataclass
class Secao:
    numero: str
    titulo: str
    nivel: int
    campo: str | None = None
    linhas: list[str] = field(default_factory=list)
    # Pagina do PDF onde a secao comeca e onde termina. `None` quando o texto
    # veio de transcricao manual (sidecar), que nao tem paginacao.
    pagina_inicio: int | None = None
    pagina_fim: int | None = None

    @property
    def conteudo(self) -> str:
        return "\n".join(self.linhas).strip()

    @property
    def paginas(self) -> str:
        """Como a pagina aparece na citacao: "4", "4-5" ou vazio."""
        if self.pagina_inicio is None:
            return ""
        if self.pagina_fim and self.pagina_fim != self.pagina_inicio:
            return f"{self.pagina_inicio}-{self.pagina_fim}"
        return str(self.pagina_inicio)


def _e_cabecalho(numero: str, titulo: str, proximo_topo: int, topo_atual: int) -> bool:
    """Distingue titulo de secao de item de lista numerada.

    Necessario porque os procedimentos usam listas ordenadas dentro das secoes
    ("1. Desligar o equipamento."), que casam com o mesmo regex do titulo.

    Dois testes, ambos obrigatorios:

    **Posicional** — uma secao de topo so e valida se o numero for exatamente o
    proximo da sequencia; uma subsecao, se pertencer a secao de topo aberta.

    **Pontuacao** — o titulo nao pode terminar em `.`, `;` ou `:`. Sozinho, o
    teste posicional falha num caso real: a secao 5 do Doc2 tem uma lista de 7
    passos, e o item "6. Utilizar os EPIs adequados." casa com o proximo topo
    esperado. Ele era aceito como secao 6, consumia o numero e fazia a secao 6
    verdadeira ("Diagnostico Inicial") e suas subsecoes virarem corpo de texto —
    o documento perdia dois campos que de fato possui. Nenhum titulo de secao
    dos seis procedimentos termina em pontuacao; todo item de lista termina.
    """
    if titulo.rstrip().endswith((".", ";", ":")):
        return False

    partes = numero.split(".")
    if len(partes) == 1:
        return int(partes[0]) == proximo_topo
    return int(partes[0]) == topo_atual


def classificar_campo(titulo: str) -> str | None:
    """Mapeia o titulo da secao para um dos campos canonicos do `config`."""
    alvo = _sem_acento(titulo).lower()
    for chave, _rotulo, padrao in config.CAMPOS_CANONICOS:
        if re.search(padrao, alvo):
            return chave
    return None


def separar_secoes(
    texto: str, paginas: list[int | None] | None = None
) -> tuple[str, list[Secao]]:
    """Quebra o texto em `(titulo_do_documento, secoes)`.

    O titulo sao as linhas antes da secao 1 — nos PDFs ele vem quebrado em
    duas ou tres linhas pela largura da pagina, entao juntamos.

    `paginas` e a lista devolvida por `extrair_texto`, paralela as linhas. Quando
    presente, cada secao registra em que pagina comeca e termina.
    """
    linhas = [ln.rstrip() for ln in texto.splitlines()]
    paginas = paginas or [None] * len(linhas)

    def pagina_de(i: int) -> int | None:
        return paginas[i] if i < len(paginas) else None

    secoes: list[Secao] = []
    cabecalho_doc: list[str] = []
    proximo_topo, topo_atual = 1, 0

    for i, ln in enumerate(linhas):
        limpa = ln.strip()
        if not limpa:
            if secoes:
                secoes[-1].linhas.append("")
            continue

        m = _CABECALHO.match(limpa)
        if m and _e_cabecalho(m.group(1), m.group(2), proximo_topo, topo_atual):
            numero, titulo = m.group(1), m.group(2).strip()
            nivel = numero.count(".") + 1
            if nivel == 1:
                topo_atual = int(numero)
                proximo_topo = topo_atual + 1
            pag = pagina_de(i)
            secoes.append(
                Secao(numero=numero, titulo=titulo, nivel=nivel,
                      campo=classificar_campo(titulo),
                      pagina_inicio=pag, pagina_fim=pag)
            )
            continue

        if secoes:
            secoes[-1].linhas.append(limpa)
            # A secao se estende ate a ultima linha de conteudo que ela recebeu.
            if (pag := pagina_de(i)) is not None:
                secoes[-1].pagina_fim = pag
        else:
            cabecalho_doc.append(limpa)

    return " ".join(cabecalho_doc).strip(), secoes


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

# Os PDFs usam "•" para lista e quebram paragrafos na largura da pagina.
_MARCADOR = re.compile(r"^[•·▪]\s*")


def _corpo_markdown(secao: Secao) -> list[str]:
    """Converte o corpo de uma secao em Markdown.

    Duas normalizacoes, ambas de forma e nao de conteudo:
      1. "• item" vira "- item"
      2. linhas soltas viram paragrafo — o PDF quebra na largura da pagina, e
         manter essas quebras produziria chunks cortados no meio da frase
    """
    saida: list[str] = []
    buffer: list[str] = []

    def descarrega():
        if buffer:
            saida.append(" ".join(buffer))
            saida.append("")
            buffer.clear()

    for ln in secao.linhas:
        if not ln:
            descarrega()
            continue
        if _MARCADOR.match(ln):
            descarrega()
            saida.append("- " + _MARCADOR.sub("", ln))
            continue
        if re.match(r"^\d+\.\s+", ln):
            descarrega()
            saida.append(ln)
            continue
        buffer.append(ln)

    descarrega()

    # Colapsa linhas em branco repetidas
    resultado: list[str] = []
    for ln in saida:
        if ln == "" and resultado and resultado[-1] == "":
            continue
        resultado.append(ln)
    return resultado


def pdf_para_markdown(caminho_pdf: Path, destino: Path | None = None) -> dict:
    """Converte um PDF em `.md` e devolve o resumo do que foi feito."""
    caminho_pdf = Path(caminho_pdf)
    destino = Path(destino) if destino else config.DOCS_MD_DIR
    destino.mkdir(parents=True, exist_ok=True)

    texto, origem, paginas = extrair_texto(caminho_pdf)
    slug = caminho_pdf.stem
    arquivo_md = destino / f"{slug}.md"

    if origem == "vazio":
        return {
            "documento": slug, "arquivo": None, "origem": origem,
            "titulo": None, "secoes": 0, "campos": [], "ok": False,
            "aviso": "PDF sem camada de texto e sem sidecar — precisa de OCR ou "
                     f"de uma transcricao em {caminho_pdf.with_suffix('.txt').name}",
        }

    titulo, secoes = separar_secoes(texto, paginas)
    campos = sorted({s.campo for s in secoes if s.campo})

    linhas = [
        "---",
        f'documento: "{slug}"',
        f'titulo: "{titulo}"',
        f'origem_pdf: "{caminho_pdf.name}"',
        f"origem_texto: {origem}",
        f"secoes: {len(secoes)}",
        "---",
        "",
        f"# {titulo}",
        "",
    ]

    for s in secoes:
        marcador = "#" * min(s.nivel + 1, 6)
        # Metadados no comentario HTML: invisiveis na leitura, mas reparseaveis.
        marcas = []
        if s.campo:
            marcas.append(f"campo: {s.campo}")
        if s.paginas:
            marcas.append(f"pagina: {s.paginas}")
        rotulo = f" <!-- {' | '.join(marcas)} -->" if marcas else ""
        linhas.append(f"{marcador} {s.numero}. {s.titulo}{rotulo}")
        linhas.append("")
        linhas.extend(_corpo_markdown(s))
        if linhas and linhas[-1] != "":
            linhas.append("")

    arquivo_md.write_text("\n".join(linhas).rstrip() + "\n", encoding="utf-8")

    return {
        "documento": slug, "arquivo": arquivo_md, "origem": origem,
        "titulo": titulo, "secoes": len(secoes), "campos": campos,
        "ok": True, "aviso": None,
    }


def converter_todos(origem: Path | None = None, destino: Path | None = None) -> pd.DataFrame:
    """Converte todos os PDFs de `data/raw/`. Ponto de entrada do CLI e da UI."""
    origem = Path(origem) if origem else config.RAW_DIR
    pdfs = sorted(origem.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF em {origem}")
    return pd.DataFrame([pdf_para_markdown(p, destino) for p in pdfs])


# --------------------------------------------------------------------------
# Leitura de volta
# --------------------------------------------------------------------------

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
# O comentario final carrega campo e/ou pagina, em qualquer combinacao:
#   <!-- campo: correcao | pagina: 4 -->   <!-- pagina: 2 -->   (ou nenhum)
_TITULO_SECAO = re.compile(
    r"^#{2,6}\s+(\d+(?:\.\d+)*)\.\s+(.*?)(?:\s*<!--(.*?)-->)?\s*$",
    re.M,
)
_MARCA_CAMPO = re.compile(r"campo:\s*(\w+)")
_MARCA_PAGINA = re.compile(r"pagina:\s*(\d+)(?:\s*-\s*(\d+))?")


def _ler_marcas(comentario: str) -> tuple[str | None, int | None, int | None]:
    """Extrai `(campo, pagina_inicio, pagina_fim)` do comentario do titulo."""
    if not comentario:
        return None, None, None
    campo = m.group(1) if (m := _MARCA_CAMPO.search(comentario)) else None
    if p := _MARCA_PAGINA.search(comentario):
        inicio = int(p.group(1))
        fim = int(p.group(2)) if p.group(2) else inicio
        return campo, inicio, fim
    return campo, None, None


def carregar_markdowns(diretorio: Path | None = None) -> list[dict]:
    """Le os `.md` gerados e devolve titulo, campos e secoes de cada um."""
    diretorio = Path(diretorio) if diretorio else config.DOCS_MD_DIR
    if not diretorio.exists():
        return []

    docs = []
    for arq in sorted(diretorio.glob("*.md")):
        texto = arq.read_text(encoding="utf-8")

        meta = {}
        if m := _FRONT.match(texto):
            for ln in m.group(1).splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')

        secoes = []
        for num, tit, comentario in _TITULO_SECAO.findall(texto):
            campo, pag_ini, pag_fim = _ler_marcas(comentario)
            secoes.append(
                {"numero": num, "titulo": tit.strip(), "campo": campo,
                 "nivel": num.count(".") + 1,
                 "pagina_inicio": pag_ini, "pagina_fim": pag_fim}
            )

        docs.append(
            {
                "documento": meta.get("documento", arq.stem),
                "titulo": meta.get("titulo", ""),
                "origem_texto": meta.get("origem_texto", "?"),
                "arquivo": arq,
                "secoes": secoes,
                "n_secoes": len(secoes),
                "campos": sorted({s["campo"] for s in secoes if s["campo"]}),
                "caracteres": len(texto),
            }
        )
    return docs


# --------------------------------------------------------------------------
# Cobertura de campos
# --------------------------------------------------------------------------


def matriz_campos(docs: list[dict]) -> pd.DataFrame:
    """Matriz campo canonico x documento.

    Celula = numero da secao que cobre o campo, ou vazio se pendente. Mostramos
    o numero e nao um "X" porque o numero e o endereco da citacao que o LLM vai
    ter de produzir na Parte 5 ("Doc2, secao 9").
    """
    linhas = []
    for chave, rotulo, _padrao in config.CAMPOS_CANONICOS:
        linha = {"campo": rotulo, "chave": chave}
        for d in docs:
            nums = [s["numero"] for s in d["secoes"] if s["campo"] == chave]
            linha[d["documento"]] = ", ".join(nums) if nums else ""
        linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    colunas_doc = [d["documento"] for d in docs]
    tabela["documentos_com"] = (tabela[colunas_doc] != "").sum(axis=1)
    tabela["pendente_em"] = len(docs) - tabela["documentos_com"]
    return tabela


def campos_pendentes(docs: list[dict]) -> pd.DataFrame:
    """Um registro por (documento, campo ausente). Lista de acao, nao de status."""
    rotulos = {c: r for c, r, _ in config.CAMPOS_CANONICOS}
    linhas = [
        {"documento": d["documento"], "titulo": d["titulo"],
         "campo_ausente": rotulos[chave], "chave": chave}
        for d in docs
        for chave in rotulos
        if chave not in d["campos"]
    ]
    return pd.DataFrame(linhas, columns=["documento", "titulo", "campo_ausente", "chave"])


# --------------------------------------------------------------------------
# Ligacao com a coluna `fault`
# --------------------------------------------------------------------------


def cobertura_por_familia(docs: list[dict], familias_no_banner) -> pd.DataFrame:
    """Cruza cada familia de `fault` do banner.csv com o documento que a cobre.

    Esta e a tabela que o guardrail **G3** consulta: familia sem documento
    encerra o fluxo prescritivo com a mensagem padronizada, sem chamar o LLM.

    `cobertura` assume tres valores:
      - `documentado`  — ha procedimento dedicado
      - `parcial`      — o fenomeno aparece num documento de outro componente
                         (ver `config.COBERTURA_PARCIAL`); nao vale para G3
      - `sem_documento`— G3 recusa
    """
    titulos = {d["documento"]: d["titulo"] for d in docs}
    existentes = set(titulos)

    familia_para_doc: dict[str, str] = {}
    for doc, familias in config.MAPA_DOC_FAMILIA.items():
        for f in familias:
            familia_para_doc[f] = doc

    familia_parcial: dict[str, str] = {}
    for doc, familias in config.COBERTURA_PARCIAL.items():
        for f in familias:
            familia_parcial[f] = doc

    linhas = []
    for fam in sorted(set(familias_no_banner)):
        doc = familia_para_doc.get(fam)
        parcial = familia_parcial.get(fam)

        if doc and doc in existentes:
            cobertura, documento = "documentado", doc
        elif parcial and parcial in existentes:
            cobertura, documento = "parcial", parcial
        else:
            cobertura, documento = "sem_documento", None

        linhas.append(
            {
                "familia": fam,
                "cobertura": cobertura,
                "documento": documento,
                "titulo_documento": titulos.get(documento, ""),
                "g3_libera": cobertura == "documentado",
            }
        )

    ordem = {"sem_documento": 0, "parcial": 1, "documentado": 2}
    return (
        pd.DataFrame(linhas)
        .sort_values(["cobertura", "familia"], key=lambda s: s.map(ordem).fillna(s))
        .reset_index(drop=True)
    )
