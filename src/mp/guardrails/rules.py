"""As oito verificacoes, na ordem.

| ID | Pergunta | Se falhar |
|-----|----------|-----------|
| G0  | O JSON recebido faz sentido? | rejeita a entrada |
| G1  | Achou vizinhos parecidos o bastante? | "sem ocorrencias similares" |
| G1T | A evidencia aponta UM manual? | mostra a lista, o tecnico escolhe |
| G2  | E defeito, ou so um estado normal? | encerra o fluxo prescritivo |
| G3  | Essa familia tem manual? | "sem documentacao" |
| G4  | Os trechos recuperados servem? | trata como G3 |
| G5  | A citacao existe mesmo no texto? | regenera ou degrada |
| G5N | Os numeros da prosa foram apurados? | regenera ou escreve por codigo |

As duas ultimas sao a mesma pergunta sobre materias diferentes: o **G5** cuida
do turno que responde a partir do manual, e confere citacao; o **G5N** cuida do
turno que anuncia o que a similaridade apurou, onde nao ha manual a citar e o
que se confere e numero. Sem ele, o unico turno sem verificacao seria justamente
o que da o diagnostico.

O **G1T** so existe no caminho por texto, quando o documento e o resultado da
busca em vez da entrada dela.

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
from mp.retrieval.catalog import verificar_existencia_conserto

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

    Julga se o **dado** e utilizavel, nao se a maquina esta boa: 5000 graus nao
    e falha grave, e sensor quebrado ou unidade trocada.

    `fault` e opcional de proposito — em producao o evento chega sem rotulo e
    quem infere e a similaridade. Quando vier, e anotacao do operador a
    confrontar com os vizinhos.
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

    Com `distancia=None` devolve `passou=True` e avisa na mensagem que nao
    verificou. Guardrail nao implementado deve deixar passar: bloquear tudo
    esconderia o resto do fluxo e fingiria que a trava funciona.

    O limiar definitivo sai da distribuicao das distancias dentro de cada classe
    e vai para `config`.
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

    Aceita rotulo cru ou familia. Quem decide e `verificar_existencia_conserto`;
    aqui so se le o veredito — nao ha uma segunda leitura do YAML que pudesse
    discordar dele.

    Reprova tambem o rotulo desconhecido: sem catalogo nao da para afirmar que
    e defeito. Sao recusas diferentes, e a verificacao traz a frase de cada uma.
    """
    c = verificar_existencia_conserto(rotulo_ou_familia)
    return Veredito(
        "G2", c.e_defeito,
        f"'{c.familia}' e um defeito." if c.e_defeito else c.mensagem,
        {"familia": c.familia, "is_problem": c.e_defeito, "situacao": c.situacao},
    )


# --------------------------------------------------------------------------
# G3 — existe manual?
# --------------------------------------------------------------------------


def g3_tem_documento(familia: str | None) -> Veredito:
    """`SELECT` no catalogo. Nunca similaridade.

    Busca vetorial nunca volta vazia, entao ela nao pode responder "existe
    documento?". Uma consulta exata pode.

    Cobertura **parcial** conta como ausencia: `eccentric_rotor` aparece no
    manual de polias, mas e excentricidade de polia, nao de rotor — prescrever
    ajuste de polia para problema de rotor seria pior que recusar. Essa regra
    vive em `verificar_existencia_conserto`, junto com a frase que a explica.
    """
    c = verificar_existencia_conserto(familia)
    return Veredito(
        "G3", c.prescrever, c.mensagem,
        {"familia": c.familia, "documentos": c.documento_ids,
         "situacao": c.situacao},
    )


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
# G1T — a evidencia aponta UM manual? (caminho por texto)
# --------------------------------------------------------------------------


def g1t_evidencia_decide(
    ranking: list[tuple[str, float]],
    margem_minima: float = config.MARGEM_MINIMA_DOCUMENTO,
    share_minima: float = config.SHARE_MINIMO_DOCUMENTO,
) -> Veredito:
    """A evidencia aponta UM manual, ou ainda ha varios plausiveis?

    Existe porque o vencedor sai de um `max`, que sempre devolve alguem — mesmo
    com 1,43 contra 1,39. E manual travado nao muda mais.

    Duas condicoes, ambas obrigatorias:

    `margem`  o quanto o 1o ganha do 2o. Pega o empate na cabeca.
    `share`   quanto do peso TOTAL o 1o concentra. Pesos [1,0; 0,5; 0,5; 0,5]
              tem margem de 50% e ainda assim deixam tres manuais plausiveis:
              ganhar do segundo nao e ganhar de todos.

    Reprovar aqui nao e recusar, e dizer "ainda nao da para decidir".
    """
    if not ranking:
        return Veredito("G1T", False, "Nenhum documento candidato.",
                        {"candidatos": 0})

    if len(ranking) == 1:
        return Veredito(
            "G1T", True, f"Um unico candidato: {ranking[0][0]}.",
            {"candidatos": 1, "margem": 1.0, "share": 1.0, "ranking": ranking},
        )

    (d1, p1), (d2, p2) = ranking[0], ranking[1]
    total = sum(p for _, p in ranking)
    margem = (p1 - p2) / p1 if p1 > 0 else 0.0
    share = p1 / total if total > 0 else 0.0

    detalhe = {"candidatos": len(ranking), "margem": round(margem, 4),
               "share": round(share, 4), "margem_minima": margem_minima,
               "share_minima": share_minima, "ranking": ranking,
               "primeiro": d1, "segundo": d2}

    if margem < margem_minima:
        return Veredito(
            "G1T", False,
            f"{d1} ({p1:.2f}) e {d2} ({p2:.2f}) estao empatados — margem de "
            f"{margem:.0%}, minimo {margem_minima:.0%}.",
            detalhe,
        )

    if share < share_minima:
        return Veredito(
            "G1T", False,
            f"{d1} ganha do segundo, mas concentra so {share:.0%} da evidencia "
            f"(minimo {share_minima:.0%}): {len(ranking)} manuais ainda "
            "plausiveis.",
            detalhe,
        )

    return Veredito(
        "G1T", True,
        f"{d1} ganha de {d2} por {margem:.0%} e concentra {share:.0%} da "
        "evidencia.",
        detalhe,
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
# G5N — a prosa preservou os numeros apurados?
# --------------------------------------------------------------------------

# Numero com virgula ou ponto decimal, com ou sem sinal de porcentagem depois.
_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _formas(valor) -> set[str]:
    """As grafias aceitaveis de um numero apurado.

    O mesmo fato pode ser escrito de varios jeitos legitimos, e reprovar por
    formatacao seria reprovar o certo: 0,72 e 72% sao o mesmo numero, e o
    portugues escreve virgula onde o Python escreve ponto.
    """
    formas: set[str] = set()
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return formas

    candidatos = [valor, round(float(valor), 1), round(float(valor), 2),
                  round(float(valor), 3)]
    # Fracao tambem vale escrita como porcentagem, e vice-versa.
    if 0 <= float(valor) <= 1:
        candidatos += [round(float(valor) * 100, 1), round(float(valor) * 100)]
    if float(valor) == int(valor):
        candidatos.append(int(valor))

    for c in candidatos:
        texto = f"{c}".rstrip("0").rstrip(".") if isinstance(c, float) else f"{c}"
        formas.add(texto)
        formas.add(texto.replace(".", ","))
    return formas


def g5n_numeros_apurados(resposta: str, fatos: dict) -> Veredito:
    """Todo numero da prosa tem de ser um numero que o codigo apurou.

    **Existe porque o G5 nao alcanca este turno.** O G5 confere citacao de
    manual; a classificacao de um evento nao vem de manual nenhum — vem do kNN.
    Sem uma trava propria, o unico turno da conversa sem verificacao seria
    justamente o que anuncia o diagnostico.

    A regra e a mesma dos outros guardrails: o modelo **redige**, nunca produz.
    Aqui isso vira uma pergunta conferivel — cada numero escrito aparece no
    bloco de fatos que foi enviado?

    Aceita a mesma quantidade escrita de formas diferentes (`0,72`, `0.72`,
    `72%`), porque reprovar por formatacao reprovaria o certo. Nao aceita numero
    novo, que e o que interessa barrar.
    """
    permitidos: set[str] = set()
    for valor in fatos.values():
        permitidos |= _formas(valor)

    escritos = _NUMERO.findall(resposta or "")
    # Normaliza para comparar: `0,720` e `0,72` sao o mesmo numero escrito.
    inventados = sorted(
        {n for n in escritos
         if n not in permitidos
         and n.replace(",", ".").rstrip("0").rstrip(".") not in
            {p.replace(",", ".") for p in permitidos}}
    )

    if inventados:
        return Veredito(
            "G5N", False,
            "A resposta traz numeros que nao foram apurados: "
            + ", ".join(inventados),
            {"inventados": inventados, "permitidos": sorted(permitidos)},
        )

    return Veredito("G5N", True, f"{len(escritos)} numero(s) conferido(s).",
                    {"escritos": escritos})


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
