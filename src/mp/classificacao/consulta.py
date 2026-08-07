"""O que a floresta diz sobre um trecho de leituras que acabou de chegar.

Ponte entre o modelo e a conversa. O `Classificador` devolve um ranking de
probabilidades; o que a sessao precisa e um veredito com os numeros ao lado, e
um lugar para confrontar a anotacao do operador.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mp import config
from mp.retrieval.catalog import familia_de

__all__ = ["Classificacao", "classificar_bloco"]

# Abaixo disso a floresta nao viu nada de que tenha certeza, e o fluxo para.
#
# 1/13 = 7,7% e o chute; 0,30 exige quatro vezes isso. Nao e calibrado: e um
# piso, e o que sustenta o fluxo depois dele sao o G2 e o G3, que sao `SELECT`.
CONFIANCA_MINIMA = 0.30

# Menos leituras que isto e a floresta responde mal: quatro das cinco
# estatisticas (desvio, inclinacao, amplitude, p90-p10) ficam perto de zero, e
# zero variacao e justamente a assinatura de `motor_desligado`. Medido: uma
# leitura so de um evento de desalinhamento saiu como `motor_desligado`.
MINIMO_DE_LEITURAS = 10


@dataclass
class Classificacao:
    """O veredito da floresta sobre um trecho, com os numeros que o sustentam."""

    familia: str | None = None
    confianca: float = 0.0
    n_leituras: int = 0
    n_janelas: int = 0
    ranking: pd.DataFrame | None = None

    # Preenchido quando o trecho traz `fault` do operador. E anotacao a
    # conferir, nunca ordem: vale o que o modelo indica, e a divergencia vira
    # alerta na tela.
    rotulo_do_operador: str | None = None
    familia_do_operador: str | None = None

    alertas: list[str] = field(default_factory=list)

    @property
    def divergiu(self) -> bool:
        return (
            self.familia_do_operador is not None
            and self.familia is not None
            and self.familia_do_operador != self.familia
        )

    @property
    def confiavel(self) -> bool:
        return self.confianca >= CONFIANCA_MINIMA


def classificar_bloco(bloco: pd.DataFrame, modelo) -> Classificacao:
    """Classifica um trecho de leituras. `modelo` e um `Classificador` treinado.

    Aceita trecho curto — o `Classificador` resume o bloco inteiro quando ele
    nao cabe uma janela —, mas registra um alerta abaixo de
    `MINIMO_DE_LEITURAS`, porque ali a resposta deixa de ser confiavel.
    """
    if bloco is None or bloco.empty:
        return Classificacao(alertas=["Nenhuma leitura recebida."])

    ranking = modelo.consultar(bloco[modelo.colunas])
    topo = ranking.iloc[0]

    c = Classificacao(
        familia=str(topo["familia"]),
        confianca=round(float(topo["probabilidade"]), 3),
        n_leituras=len(bloco),
        n_janelas=max(1, (len(bloco) - modelo.tamanho) // max(1, modelo.tamanho // 2) + 1)
        if len(bloco) >= modelo.tamanho else 1,
        ranking=ranking,
    )

    if len(bloco) < MINIMO_DE_LEITURAS:
        c.alertas.append(
            f"Apenas {len(bloco)} leitura(s). Com menos de {MINIMO_DE_LEITURAS} "
            "quase nao ha variacao para medir, e o modelo tende a responder "
            "`motor_desligado`. Envie um trecho maior."
        )

    if (anotado := bloco.get(config.COLUNA_ROTULO)) is not None and len(anotado):
        c.rotulo_do_operador = str(anotado.iloc[0])
        c.familia_do_operador = familia_de(c.rotulo_do_operador)

    return c
