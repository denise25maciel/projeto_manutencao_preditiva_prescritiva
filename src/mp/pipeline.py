"""Ponto unico de entrada: do rotulo ate a resposta prescritiva.

O fluxo inteiro em uma funcao, na ordem em que o GUIA.md o descreve. Ler esta
funcao de cima para baixo e ler a arquitetura do projeto.

**O que este modulo torna visivel.** A resposta existe em dois estagios, e os
dois sao devolvidos:

    resposta_bruta   o que o sistema achou    — sem modelo de linguagem nenhum
    resposta_llm     o mesmo, redigido        — o modelo entra so aqui

O primeiro e o conteudo. O segundo e a forma. Se o modelo cair, sumir a chave da
API ou a GPU faltar, o primeiro continua funcionando — e continua sendo uma
resposta util, so que menos agradavel de ler.

Essa separacao nao e didatica: e o caminho de degradacao real. Quando o G5
reprova duas vezes, e para o estagio 1 que o sistema volta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from mp.guardrails import rules as g
from mp.llm import prompts
from mp.retrieval import rag
from mp.retrieval.catalog import verificar_existencia_conserto

# Quantas vezes tentar de novo quando o G5 reprova, antes de degradar.
MAX_TENTATIVAS = 2


@dataclass
class Prescricao:
    """Tudo que o fluxo produziu, incluindo o que foi recusado e por que."""

    pergunta: str
    rotulo: str | None = None
    familia: str | None = None
    documentos: list[str] = field(default_factory=list)

    vereditos: list = field(default_factory=list)
    trechos: list = field(default_factory=list)

    # --- estagio 1: sem modelo de linguagem --------------------------------
    resposta_bruta: str = ""

    # --- estagio 2: o modelo redige ----------------------------------------
    prompt: str = ""
    resposta_llm: str | None = None
    tentativas: int = 0
    degradou: bool = False

    provedor: str | None = None
    modelo: str | None = None
    tokens_entrada: int = 0
    tokens_saida: int = 0
    segundos_llm: float = 0.0
    segundos_total: float = 0.0

    recusa: str | None = None

    @property
    def usou_llm(self) -> bool:
        return self.resposta_llm is not None

    @property
    def parou_em(self) -> str | None:
        """O primeiro guardrail que reprovou, se houve."""
        for v in self.vereditos:
            if not v.passou:
                return v.id
        return None

    @property
    def resposta_final(self) -> str:
        """O que o tecnico ve. Degrada para o texto do manual quando preciso."""
        if self.recusa:
            return self.recusa
        if self.resposta_llm and not self.degradou:
            return self.resposta_llm
        return self.resposta_bruta


def montar_resposta_bruta(trechos, familia: str) -> str:
    """O estagio 1: os trechos do manual, sem uma palavra escrita por modelo.

    E o texto original do procedimento, com o endereco de cada pedaco. Nao
    responde a pergunta especifica — despeja o que a busca encontrou —, mas
    **nada aqui pode estar errado**, porque nada aqui foi reescrito.
    """
    if not trechos:
        return "Nenhum trecho de procedimento recuperado."

    partes = [f"Trechos do procedimento para **{familia}**, na ordem de relevancia:\n"]
    for i, t in enumerate(trechos, 1):
        campo = f" — {t.campo}" if t.campo else ""
        pagina = f", pag. {t.pagina}" if t.pagina else ""
        partes.append(
            f"**{i}. {t.documento_id}, secao {t.numero}{pagina} — {t.titulo}{campo}** "
            f"(similaridade {t.score:.3f})\n\n{t.texto.strip()}\n"
        )
    return "\n".join(partes)


def responder(
    pergunta: str,
    rotulo: str | None = None,
    evento: dict | None = None,
    cliente=None,
    k: int = 5,
    so_prescritivos: bool = True,
    fatos: dict | None = None,
    motor=None,
) -> Prescricao:
    """Executa o fluxo completo e devolve os dois estagios.

    `cliente=None` roda tudo **menos** a redacao — util para provar que o
    conteudo nao depende do modelo, e e o que a tela usa no modo "antes".

    A ordem e a do GUIA.md e nao pode mudar: as verificacoes baratas vem antes
    das caras, e o modelo e a mais cara de todas. Uma familia sem manual e
    recusada sem que uma linha chegue ao LLM.
    """
    t0 = time.time()
    p = Prescricao(pergunta=pergunta, rotulo=rotulo)

    # --- G0: a entrada faz sentido? ----------------------------------------
    if evento is not None:
        v = g.g0_entrada(evento)
        p.vereditos.append(v)
        if not v:
            p.recusa = v.mensagem
            p.segundos_total = round(time.time() - t0, 2)
            return p
        p.rotulo = rotulo or evento.get("fault")

    if not p.rotulo:
        p.recusa = "Sem rotulo para consultar o catalogo."
        return p

    # --- G1: ha ocorrencias parecidas? -------------------------------------
    # Ainda um stub: a Parte 3 e quem produz a distancia. Fica no fluxo para
    # que a ordem esteja certa quando ela chegar.
    v = g.g1_similaridade(None)
    p.vereditos.append(v)

    # --- G2 e G3: catalogo, nao similaridade -------------------------------
    p.familia = verificar_existencia_conserto(p.rotulo).familia

    v = g.g2_e_problema(p.rotulo)
    p.vereditos.append(v)
    if not v:
        p.recusa = v.mensagem
        p.segundos_total = round(time.time() - t0, 2)
        return p

    v = g.g3_tem_documento(p.familia)
    p.vereditos.append(v)
    if not v:
        p.recusa = v.mensagem
        p.segundos_total = round(time.time() - t0, 2)
        return p

    p.documentos = v.detalhe.get("documentos", [])

    # --- busca: filtro exato, depois semelhanca ----------------------------
    busca = (rag.buscar_prescritivo if so_prescritivos else rag.buscar)(
        pergunta, p.familia, k=k, motor=motor
    )
    p.trechos = busca.trechos

    # --- G4: os trechos servem? --------------------------------------------
    v = g.g4_trechos_relevantes(p.trechos)
    p.vereditos.append(v)
    if not v:
        p.recusa = v.mensagem
        p.segundos_total = round(time.time() - t0, 2)
        return p

    # --- estagio 1: a resposta sem modelo ----------------------------------
    p.resposta_bruta = montar_resposta_bruta(p.trechos, p.familia)

    mensagens = prompts.montar(pergunta, p.familia, p.trechos, fatos=fatos)
    p.prompt = prompts.texto_enviado(mensagens)

    if cliente is None:
        p.segundos_total = round(time.time() - t0, 2)
        return p

    # --- estagio 2: o modelo redige, e o G5 confere -------------------------
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        p.tentativas = tentativa
        r = cliente.gerar(mensagens)

        p.resposta_llm = r.texto.strip()
        p.provedor, p.modelo = r.provedor, r.modelo
        p.tokens_entrada += r.tokens_entrada
        p.tokens_saida += r.tokens_saida
        p.segundos_llm = round(p.segundos_llm + r.segundos, 2)

        v = g.g5_citacoes_existem(p.resposta_llm, p.trechos)
        if v.passou or tentativa == MAX_TENTATIVAS:
            p.vereditos.append(v)
            p.degradou = not v.passou
            break

        # Nao passou e ainda ha tentativa: reforca a regra violada e repete.
        # O historico da tentativa ruim NAO entra — senao o proprio erro vira
        # contexto e o modelo o trata como algo ja aceito.
        disponiveis = ", ".join(
            f"{t.documento_id}, secao {t.numero}" for t in p.trechos
        )
        mensagens = prompts.montar(
            f"{pergunta}\n\n[Aviso do sistema: a resposta anterior foi rejeitada — "
            f"{v.mensagem} As unicas fontes citaveis sao: {disponiveis}. "
            "Cite apenas essas, no formato (DocN, secao X).]",
            p.familia, p.trechos, fatos=fatos,
        )

    p.segundos_total = round(time.time() - t0, 2)
    return p
