"""As seis verificacoes, na ordem.

| ID | Pergunta | Se falhar |
|----|----------|-----------|
| G0 | O JSON recebido faz sentido? | rejeita a entrada |
| G1 | Achou vizinhos parecidos o bastante? | "sem ocorrencias similares" |
| G2 | E defeito, ou so um estado normal? | encerra o fluxo prescritivo |
| G3 | Essa familia tem manual? | "sem documentacao" |
| G4 | Os trechos recuperados servem? | trata como G3 |
| G5 | A citacao existe mesmo no texto? | regenera ou degrada |

**Sao codigo, nao prompt.** Pedir num prompt que o modelo "nao invente citacao"
e uma sugestao. Conferir depois, comparando com o texto que foi enviado, e uma
garantia.

**A ordem importa e e barata.** G0 a G3 rodam antes de qualquer coisa cara: se a
falha nao tem manual, o sistema recusa sem chamar o modelo de linguagem uma unica
vez. O caminho de recusa e o primeiro que deve funcionar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mp import config
from mp.retrieval.catalog import documentos_de, familia_de, is_problem

# --------------------------------------------------------------------------
# Faixas fisicas aceitaveis para o G0
# --------------------------------------------------------------------------
#
# Derivadas do observado no banner.csv, com folga generosa. Nao servem para
# detectar defeito — servem para barrar entrada corrompida: texto onde deveria
# haver numero, sinal trocado, unidade errada por um fator de mil.
#
# `None` em qualquer extremo significa "sem limite daquele lado".
FAIXAS = {
    "rpm": (0.0, 10_000.0),
    "temperature_c": (-40.0, 200.0),
    "z_rms_velocity_mm_s": (0.0, 1_000.0),
    "x_rms_velocity_mm_s": (0.0, 1_000.0),
    "z_peak_velocity_mm_s": (0.0, 1_000.0),
    "x_peak_velocity_mm_s": (0.0, 1_000.0),
    "z_rms_acceleration_g": (0.0, 100.0),
    "x_rms_acceleration_g": (0.0, 100.0),
    "z_peak_acceleration_g": (0.0, 500.0),
    "x_peak_acceleration_g": (0.0, 500.0),
    "z_high_freq_rms_accel_g": (0.0, 100.0),
    "x_high_freq_rms_accel_g": (0.0, 100.0),
    "z_kurtosis": (0.0, 500.0),
    "x_kurtosis": (0.0, 500.0),
    "z_crest_factor": (0.0, 100.0),
    "x_crest_factor": (0.0, 100.0),
    "z_peak_vel_comp_freq_hz": (0.0, 10_000.0),
    "x_peak_vel_comp_freq_hz": (0.0, 10_000.0),
}

# Sem estas, nao ha o que comparar.
CAMPOS_OBRIGATORIOS = ("z_rms_velocity_mm_s", "x_rms_velocity_mm_s", "rpm")

# Score de cosseno abaixo disso significa que o trecho nao fala do assunto.
# Provisorio: sera calibrado na Parte 4 com perguntas reais.
SCORE_MINIMO_CHUNK = 0.15


@dataclass
class Veredito:
    """O resultado de uma verificacao. `passou=False` interrompe o fluxo."""

    id: str
    passou: bool
    mensagem: str
    detalhe: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passou


# --------------------------------------------------------------------------
# G0 — a entrada faz sentido?
# --------------------------------------------------------------------------


def g0_entrada(evento: dict) -> Veredito:
    """Schema e faixas fisicas do JSON recebido.

    Nao julga se a maquina esta boa ou ruim: julga se o **dado** e utilizavel.
    Uma temperatura de 5000 graus nao e uma falha grave, e um sensor quebrado ou
    uma unidade trocada.

    O campo `fault` e opcional de proposito. O enunciado traz um exemplo com ele
    preenchido, mas em producao o evento chega sem rotulo — quem infere e a
    similaridade. Quando vier, e tratado como anotacao do operador, a ser
    confrontada com o que os vizinhos indicam.
    """
    if not isinstance(evento, dict):
        return Veredito("G0", False, "A entrada precisa ser um objeto JSON.",
                        {"tipo_recebido": type(evento).__name__})

    faltando = [c for c in CAMPOS_OBRIGATORIOS if c not in evento]
    if faltando:
        return Veredito("G0", False,
                        f"Faltam campos obrigatorios: {', '.join(faltando)}.",
                        {"faltando": faltando})

    nao_numericos, fora_da_faixa = [], []
    for campo, valor in evento.items():
        if campo not in FAIXAS:
            continue
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            nao_numericos.append(campo)
            continue
        minimo, maximo = FAIXAS[campo]
        if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
            fora_da_faixa.append(f"{campo}={valor} (esperado {minimo} a {maximo})")

    if nao_numericos:
        return Veredito("G0", False,
                        f"Campos com valor nao numerico: {', '.join(nao_numericos)}.",
                        {"campos": nao_numericos})
    if fora_da_faixa:
        return Veredito("G0", False,
                        "Valores fora da faixa fisica: " + "; ".join(fora_da_faixa),
                        {"campos": fora_da_faixa})

    return Veredito("G0", True, "Entrada valida.",
                    {"campos_recebidos": len(evento),
                     "tem_rotulo": config.COLUNA_ROTULO in evento})


# --------------------------------------------------------------------------
# G1 — os vizinhos sao parecidos o bastante?
# --------------------------------------------------------------------------


def g1_similaridade(distancia: float | None, limiar: float | None = None) -> Veredito:
    """Distancia do vizinho mais proximo contra o limiar.

    **Incompleto — depende da Parte 3.** O motor de similaridade ainda nao existe,
    entao nao ha distancia para comparar nem distribuicao para calibrar o limiar.

    Com `distancia=None`, devolve `passou=True` e diz na mensagem que nao foi
    verificado. Deixar passar e o comportamento certo para um guardrail nao
    implementado: bloquear tudo esconderia o resto do fluxo e daria a impressao
    falsa de que a trava funciona.

    Quando a Parte 3 existir, o limiar sai da distribuicao das distancias dentro
    de cada classe e vem para `config`.
    """
    if distancia is None:
        return Veredito("G1", True,
                        "Nao verificado — o motor de similaridade e a Parte 3.",
                        {"implementado": False})

    limiar = limiar if limiar is not None else getattr(config, "LIMIAR_G1", None)
    if limiar is None:
        return Veredito("G1", True,
                        "Nao verificado — limiar ainda nao calibrado.",
                        {"implementado": False, "distancia": distancia})

    ok = distancia <= limiar
    return Veredito(
        "G1", ok,
        "Ha ocorrencias similares no historico." if ok
        else "Sem ocorrencias similares o bastante no historico.",
        {"distancia": distancia, "limiar": limiar, "implementado": True},
    )


# --------------------------------------------------------------------------
# G2 — e defeito ou estado?
# --------------------------------------------------------------------------


def g2_e_problema(rotulo_ou_familia: str | None) -> Veredito:
    """Encerra o fluxo quando a maquina esta apenas operando.

    Aceita rotulo cru ou familia: resolve pelo catalogo quando preciso.
    """
    if rotulo_ou_familia is None:
        return Veredito("G2", False, "Sem rotulo para avaliar.", {})

    familia = rotulo_ou_familia
    problema = is_problem(familia)
    if problema is None:
        familia = familia_de(rotulo_ou_familia)
        problema = is_problem(familia) if familia else None

    if problema is None:
        return Veredito("G2", False,
                        f"'{rotulo_ou_familia}' nao esta no catalogo — "
                        "registre-o no fault_map.yaml.",
                        {"familia": None})

    return Veredito(
        "G2", bool(problema),
        f"'{familia}' e um defeito." if problema
        else f"'{familia}' e um estado da maquina, nao um defeito. "
             "Nao ha acao corretiva a prescrever.",
        {"familia": familia, "is_problem": bool(problema)},
    )


# --------------------------------------------------------------------------
# G3 — existe manual?
# --------------------------------------------------------------------------


def g3_tem_documento(familia: str | None) -> Veredito:
    """`SELECT` no catalogo. Nunca similaridade.

    E o principio 4 do projeto: busca vetorial nunca volta vazia, entao ela nao
    pode ser quem responde "existe documento?". Aqui e uma consulta exata que
    devolve lista vazia quando nao ha.

    Cobertura **parcial** conta como ausencia. O `eccentric_rotor` aparece numa
    secao do manual de polias, mas e excentricidade de polia, nao de rotor —
    prescrever ajuste de polia para um problema de rotor seria pior que recusar.
    """
    if familia is None:
        return Veredito("G3", False, "Sem familia para consultar.", {})

    docs = documentos_de(familia)
    if not docs:
        return Veredito(
            "G3", False,
            "Sem documentacao — registre um documento.",
            {"familia": familia, "documentos": []},
        )

    ids = [d["id"] for d in docs]
    return Veredito("G3", True, f"{len(ids)} documento(s): {', '.join(ids)}.",
                    {"familia": familia, "documentos": ids})


# --------------------------------------------------------------------------
# G4 — os trechos servem?
# --------------------------------------------------------------------------


def g4_trechos_relevantes(trechos, score_minimo: float = SCORE_MINIMO_CHUNK) -> Veredito:
    """Descarta trechos fracos demais para sustentar uma resposta.

    O manual existe, mas pode nao falar do que foi perguntado. Sem esta trava, a
    busca entregaria o "menos ruim" e o modelo escreveria em cima dele.

    Falhar aqui e tratado como G3: sem base, sem prescricao.
    """
    if not trechos:
        return Veredito("G4", False, "Nenhum trecho recuperado.", {"acima_do_minimo": 0})

    scores = [t.score for t in trechos]
    bons = [s for s in scores if s >= score_minimo]

    return Veredito(
        "G4", bool(bons),
        f"{len(bons)} de {len(scores)} trechos acima do minimo." if bons
        else f"Nenhum trecho atingiu o minimo de {score_minimo} — o manual existe, "
             "mas nao responde a esta pergunta.",
        {"melhor_score": round(max(scores), 4), "score_minimo": score_minimo,
         "acima_do_minimo": len(bons), "total": len(scores)},
    )


# --------------------------------------------------------------------------
# G5 — a citacao existe?
# --------------------------------------------------------------------------

# Casa "Doc1, secao 19", "Doc2 secao 9.1", "(Doc6, secao 16)".
_CITACAO = re.compile(r"(Doc\d+)\s*,?\s*se[çc][ãa]o\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


def g5_citacoes_existem(resposta: str, trechos) -> Veredito:
    """Confere que toda citacao da resposta aponta para um trecho enviado.

    E a ultima trava, e a que pega o erro mais perigoso: uma resposta correta na
    forma, com fonte plausivel, apontando para uma secao que nunca foi lida.

    A conferencia e contra os **trechos efetivamente enviados ao modelo**, nao
    contra o catalogo inteiro. Citar uma secao que existe no manual mas nao foi
    mostrada tambem e invencao — o modelo nao tinha como saber o que ela diz.
    """
    enviadas = {(t.documento_id.lower(), t.numero) for t in trechos}
    encontradas = {(d.lower(), s) for d, s in _CITACAO.findall(resposta or "")}

    if not encontradas:
        return Veredito(
            "G5", False,
            "A resposta nao cita nenhuma fonte. A citacao e obrigatoria.",
            {"citadas": [], "disponiveis": sorted(f"{d}/{s}" for d, s in enviadas)},
        )

    inventadas = sorted(f"{d}, secao {s}" for d, s in encontradas - enviadas)
    if inventadas:
        return Veredito(
            "G5", False,
            "A resposta cita trechos que nao foram enviados: " + "; ".join(inventadas),
            {"inventadas": inventadas,
             "disponiveis": sorted(f"{d}/{s}" for d, s in enviadas)},
        )

    return Veredito("G5", True, f"{len(encontradas)} citacao(oes) conferida(s).",
                    {"citadas": sorted(f"{d}, secao {s}" for d, s in encontradas)})


# --------------------------------------------------------------------------
# Execucao em ordem
# --------------------------------------------------------------------------


def avaliar(
    evento: dict | None = None,
    rotulo: str | None = None,
    distancia: float | None = None,
    trechos=None,
    resposta: str | None = None,
) -> list[Veredito]:
    """Roda as verificacoes na ordem e **para na primeira que falhar**.

    Parar cedo nao e otimizacao, e o comportamento correto: se a falha nao tem
    manual, nao faz sentido perguntar se os trechos sao bons.

    Cada argumento ausente pula a verificacao correspondente, o que permite usar
    a funcao em pedacos enquanto o pipeline nao esta completo.
    """
    vereditos: list[Veredito] = []

    if evento is not None:
        v = g0_entrada(evento)
        vereditos.append(v)
        if not v:
            return vereditos
        if rotulo is None:
            rotulo = evento.get(config.COLUNA_ROTULO)

    v = g1_similaridade(distancia)
    vereditos.append(v)
    if not v:
        return vereditos

    if rotulo is None:
        return vereditos

    v = g2_e_problema(rotulo)
    vereditos.append(v)
    if not v:
        return vereditos

    familia = v.detalhe.get("familia")

    v = g3_tem_documento(familia)
    vereditos.append(v)
    if not v:
        return vereditos

    if trechos is None:
        return vereditos

    v = g4_trechos_relevantes(trechos)
    vereditos.append(v)
    if not v:
        return vereditos

    if resposta is None:
        return vereditos

    vereditos.append(g5_citacoes_existem(resposta, trechos))
    return vereditos
