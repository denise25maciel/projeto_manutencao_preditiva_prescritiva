"""Quais colunas entram na comparacao, e em que escala.

Um evento de sensor e comparado com o historico por distancia. Duas decisoes
mandam no resultado, e as duas estao aqui.

**Quais colunas.** Nem toda coluna numerica e uma medida fisica:

- `id` e `created_at` sao identificadores temporais. O `id` cresce junto com o
  tempo, e o dataset foi gravado em campanhas — uma falha por campanha. Deixar o
  `id` entrar seria dar ao modelo a resposta: leituras de `id` proximo tem quase
  sempre o mesmo rotulo, e a acuracia subiria sem que nada tivesse sido
  aprendido sobre vibracao. E o vazamento classico.
- `*_in_s` e `temperature_f` sao a mesma medida em outra unidade
  (`mm_s` x 25,4; Fahrenheit x Celsius). Manter as duas conta a mesma
  informacao duas vezes e dobra o peso dela na distancia.

E a pendencia **P3** do projeto, aplicada **aqui e so aqui**: as colunas
continuam no banco e no CSV como metadado. O descarte e para o calculo de
similaridade.

**Em que escala.** `rpm` vale 500 a 2000; `z_kurtosis` fica perto de 2,5. Numa
distancia euclidiana sem padronizar, o rpm decide tudo sozinho e as medidas de
vibracao viram ruido. O `StandardScaler` poe todas em desvios-padrao.

**A rotacao e um caso a parte.** Ela nao e sintoma de defeito — e regime de
operacao. O mesmo defeito a 500 e a 2000 rpm tem assinaturas muito diferentes,
e por isso ela entra na identidade do evento (`config.COLUNAS_QUEBRA_EVENTO`).
Deixa-la na distancia faz o vizinho mais proximo ser "o ensaio na mesma
rotacao", nao "a mesma falha". Por isso `incluir_regime=False` e o padrao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mp import config

# Identificadores. Entram como atalho e o acerto vira artefato.
VAZAMENTO = ("id", "created_at", "evento", "evento_a", "evento_b", "sessao", "delta_s")

# Mesma grandeza em outra unidade. A primeira de cada par fica de fora.
REDUNDANTES = (
    "z_rms_velocity_in_s", "z_peak_velocity_in_s",
    "x_rms_velocity_in_s", "x_peak_velocity_in_s",
    "temperature_f",
)

# Regime de operacao, nao sintoma. Ver a docstring do modulo.
REGIME = ("rpm", "temperature_c")


def colunas_de_similaridade(df: pd.DataFrame, incluir_regime: bool = False) -> list[str]:
    """As colunas que entram na distancia, na ordem."""
    fora = set(VAZAMENTO) | set(REDUNDANTES)
    if not incluir_regime:
        fora |= set(REGIME)

    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in fora
    ]


class Preparador:
    """Converte leituras cruas na matriz que a busca compara.

    Guarda as colunas escolhidas e a escala aprendida. O mesmo objeto tem de
    servir para o historico e para o evento novo — se o evento fosse padronizado
    com outra media, a distancia nao significaria nada.
    """

    def __init__(self, incluir_regime: bool = False, suavizar: int = 0):
        self.incluir_regime = incluir_regime
        # Mediana movel curta. Kurtosis e crest factor disparam com um impacto
        # isolado; sem suavizar, uma leitura de pico vira um vizinho estranho.
        # 0 desliga — e o padrao, porque o evento novo chega sozinho e nao tem
        # vizinhanca para suavizar, e tratar os dois lados diferente enviesaria.
        self.suavizar = suavizar
        self.colunas: list[str] = []
        self.escala = StandardScaler()

    def ajustar(self, df: pd.DataFrame) -> "Preparador":
        self.colunas = colunas_de_similaridade(df, self.incluir_regime)
        self.escala.fit(self._matriz_crua(df))
        return self

    def _matriz_crua(self, df: pd.DataFrame) -> np.ndarray:
        bloco = df[self.colunas].astype("float64")
        if self.suavizar > 1:
            bloco = bloco.rolling(self.suavizar, min_periods=1, center=True).median()
        # Nulo vira a mediana da coluna: nao inventa sinal e nao quebra a conta.
        return bloco.fillna(bloco.median()).to_numpy()

    def transformar(self, df: pd.DataFrame) -> np.ndarray:
        return self.escala.transform(self._matriz_crua(df))

    def transformar_evento(self, evento: dict) -> np.ndarray:
        """Um JSON de sensor vira uma linha na mesma escala do historico.

        Campo ausente vira a media da coluna (zero, depois de padronizar) — o
        que equivale a dizer "sem informacao", e nao "valor baixo".
        """
        linha = [float(evento.get(c, np.nan)) for c in self.colunas]
        bruto = np.array([linha], dtype="float64")
        faltando = np.isnan(bruto)
        bruto[faltando] = np.take(self.escala.mean_, np.where(faltando)[1])
        return self.escala.transform(bruto)

    @property
    def n_colunas(self) -> int:
        return len(self.colunas)
