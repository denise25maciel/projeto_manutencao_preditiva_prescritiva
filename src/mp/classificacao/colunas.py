"""Quais colunas do sensor entram no modelo — e por que as outras saem.

Das 23 colunas numericas do arquivo, 16 sao medida fisica util. As outras 7
saem por tres motivos diferentes, e cada motivo importa.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["VAZAMENTO", "REDUNDANTES", "REGIME", "colunas_de_medida"]

# **Vazamento.** `id` e `created_at` crescem com o tempo, e o arquivo foi gravado
# em campanhas — uma falha por campanha. O modelo acertaria pela posicao no
# arquivo, sem aprender nada sobre vibracao, e erraria tudo numa maquina nova.
VAZAMENTO = ("id", "created_at", "evento", "evento_a", "evento_b", "sessao", "delta_s")

# **Duplicata de unidade.** `*_in_s` e a mesma velocidade em polegada (x 25,4) e
# `temperature_f` a mesma temperatura em Fahrenheit. Manter as duas conta a
# mesma informacao duas vezes e dobra o peso dela.
REDUNDANTES = (
    "z_rms_velocity_in_s", "z_peak_velocity_in_s",
    "x_rms_velocity_in_s", "x_peak_velocity_in_s",
    "temperature_f",
)

# **Regime, nao sintoma.** `rpm` e temperatura dizem em que condicao a maquina
# rodava, nao que defeito ela tem — o mesmo defeito a 500 e a 2000 rpm tem
# assinaturas diferentes. Alem disso o rpm ja define o evento
# (`config.COLUNAS_QUEBRA_EVENTO`), e o que separa os grupos nao deveria tambem
# prever a classe dentro deles.
REGIME = ("rpm", "temperature_c")


def colunas_de_medida(df: pd.DataFrame, incluir_regime: bool = False) -> list[str]:
    """As colunas que entram no modelo, na ordem.

    `incluir_regime=True` devolve `rpm` e temperatura junto — existe para o
    experimento da aba 2 poder medir o efeito delas em vez de supo-lo.
    """
    fora = set(VAZAMENTO) | set(REDUNDANTES)
    if not incluir_regime:
        fora |= set(REGIME)

    return [c for c in df.select_dtypes(include="number").columns if c not in fora]
