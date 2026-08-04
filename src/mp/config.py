"""Decisoes numericas e caminhos do projeto.

Regra do GUIA.md: nenhum limiar fica escondido no meio do codigo. Tudo que e
escolha (limiar, fator, tamanho de janela) mora aqui, com o notebook que justifica
citado em comentario.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------

# config.py esta em src/mp/, entao a raiz do projeto sobe tres niveis.
RAIZ = Path(__file__).resolve().parents[2]

DATA_DIR = RAIZ / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "mp.db"  # Parte 2
FAULT_MAP_PATH = DATA_DIR / "fault_map.yaml"  # Parte 1

# Markdown gerado a partir dos PDFs de procedimento. Fica fora do git: e
# conteudo da empresa, so em outro formato.
DOCS_MD_DIR = DATA_DIR / "processed" / "documentos_md"

# O CSV bruto trocou de lugar durante o desenvolvimento. Em vez de fixar um
# caminho e quebrar, procuramos nos lugares plausiveis, em ordem de preferencia.
# A variavel de ambiente MP_CSV tem prioridade sobre todos.
CANDIDATOS_CSV = (
    RAW_DIR / "banner.csv",
    RAIZ / "docs" / "banner.csv",
    RAIZ / "descricao_desafio" / "banner.csv",
    RAIZ / "teste_industria" / "banner.csv",
)


def caminho_csv() -> Path:
    """Resolve o caminho do CSV bruto. Levanta erro claro se nao achar."""
    if env := os.environ.get("MP_CSV"):
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"MP_CSV aponta para {p}, que nao existe.")
        return p

    for p in CANDIDATOS_CSV:
        if p.exists():
            return p

    tentados = "\n  ".join(str(p) for p in CANDIDATOS_CSV)
    raise FileNotFoundError(
        "banner.csv nao encontrado. Procurei em:\n  " + tentados +
        "\nDefina a variavel de ambiente MP_CSV com o caminho correto."
    )


# --------------------------------------------------------------------------
# Estrutura conhecida do CSV
# --------------------------------------------------------------------------

COLUNA_ROTULO = "fault"
COLUNA_TEMPO = "created_at"
COLUNA_ID = "id"

# Pares (redundante, canonica, fator) — confirmados no 01_eda.ipynb.
# Mantemos o SI: mm/s e Celsius. As colunas imperiais sao a mesma informacao
# multiplicada por uma constante; manter as duas so infla a dimensionalidade
# do kNN e faz o StandardScaler contar a mesma grandeza duas vezes.
PARES_REDUNDANTES = [
    ("z_rms_velocity_in_s", "z_rms_velocity_mm_s", 25.4),
    ("x_rms_velocity_in_s", "x_rms_velocity_mm_s", 25.4),
    ("z_peak_velocity_in_s", "z_peak_velocity_mm_s", 25.4),
    ("x_peak_velocity_in_s", "x_peak_velocity_mm_s", 25.4),
]

# temperature_f = temperature_c * 9/5 + 32 — tratada a parte por nao ser
# um fator multiplicativo simples.
PAR_TEMPERATURA = ("temperature_f", "temperature_c")

# Tolerancia ao comparar as duas unidades. Os dados vem arredondados no arquivo,
# entao a conversao nunca bate exatamente.
#   velocidade: in/s tem 4 casas -> erro de ate 0.00005 x 25.4 = 0.0013 mm/s,
#               mais o arredondamento do proprio mm/s: ~0.0025 observado.
#   temperatura: C tem 2 casas -> erro de ate 0.005 x 9/5 = 0.009 F, mais o
#               arredondamento de F: ~0.016 observado.
# 0.02 cobre os dois casos e ainda esta ordens de grandeza abaixo de qualquer
# diferenca que significasse "medida independente".
TOLERANCIA_REDUNDANCIA = 0.02

# `id` cresce com o tempo. Usar como feature vaza a ordem de coleta para o
# modelo: o kNN acertaria por proximidade de indice, nao por vibracao.
COLUNAS_VAZAMENTO = [COLUNA_ID, COLUNA_TEMPO]

# --------------------------------------------------------------------------
# Amostragem
# --------------------------------------------------------------------------

INTERVALO_ESPERADO_S = 2.0
TOLERANCIA_INTERVALO_S = 0.25  # +-0.25s ainda conta como cadencia nominal

# Acima disso consideramos que houve corte: fim de uma sessao e inicio de outra.
# Usado para quebrar episodios (Parte 1) e para nao medir "taxa de amostragem"
# atravessando a fronteira entre sessoes.
GAP_NOVA_SESSAO_S = 60.0

# --------------------------------------------------------------------------
# Qualidade / outliers
# --------------------------------------------------------------------------

# Tukey. 1.5 marca o outlier classico; 3.0 marca o extremo. Na Parte 0 apenas
# reportamos — nenhum valor e removido ou winsorizado.
IQR_FATOR_MODERADO = 1.5
IQR_FATOR_EXTREMO = 3.0

# Uma coluna com <= este numero de valores distintos e candidata a constante /
# categorica disfarcada de numerica (rpm tem 5 patamares, por exemplo).
MAX_DISTINTOS_QUASE_CONSTANTE = 1

# --------------------------------------------------------------------------
# Rotulos
# --------------------------------------------------------------------------

# Rotulos que descrevem ESTADO, nao defeito. O fluxo prescritivo (G2) para aqui.
# A lista final vira do fault_map.yaml na Parte 1; estes sao os radicais que
# reconhecemos ja na analise para nao poluir a tabela de assinaturas de falha.
RADICAIS_NAO_PROBLEMA = (
    "normal",
    "baseline",
    "teste",
    "acelerando",
    "motor_desligado",
    "mortor_desligado",  # typo presente nos dados
    "normla",            # typo presente nos dados
)

# Sufixos que marcam repeticao de sessao/montagem, nao um defeito diferente.
# Servem para SUGERIR familias na Parte 1 — aqui nao agrupam nada sozinhos.
SUFIXOS_DE_SESSAO = ("_pos_2", "_carga", "_novo", "_adxl", "_antigo", "_teste")
PREFIXOS_DE_SESSAO = ("new_",)

# --------------------------------------------------------------------------
# Assinaturas
# --------------------------------------------------------------------------

# Colunas que compoem a tabela de assinatura por rotulo. Escolhidas por
# corresponderem ao que os PDFs de procedimento descrevem: energia por eixo,
# impacto (kurtosis/crest), banda alta (rolamento) e regime (rpm).
COLUNAS_ASSINATURA = [
    "z_rms_velocity_mm_s",
    "x_rms_velocity_mm_s",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "temperature_c",
    "rpm",
]

# --------------------------------------------------------------------------
# Documentos de procedimento
# --------------------------------------------------------------------------

# Campos que um procedimento de manutencao deveria ter. A ordem e a do fluxo
# de trabalho real: entender -> diagnosticar -> corrigir -> validar -> registrar.
#
# Cada campo casa por regex contra o TITULO da secao numerada. Ausencia de um
# campo nao invalida o documento — vira "pendente" no relatorio, que e o
# insumo para decidir se vale pedir revisao do procedimento a engenharia.
#
# Os padroes rodam sobre o titulo com acentos removidos e em minusculas.
CAMPOS_CANONICOS = [
    ("objetivo", "Objetivo", r"objetivo"),
    ("introducao", "Introducao", r"introducao|conceitos fundamentais"),
    ("descricao_problema", "Descricao do problema",
     r"descricao do problema|caracterizacao da falha|componentes d"),
    ("causas", "Causas",
     r"causas|modos de falha|tipos de falha|tipos de desalinhamento|"
     r"principais tipos"),
    ("sintomas", "Sintomas", r"sintomas"),
    ("ferramentas", "Ferramentas", r"ferramentas|instrumentos"),
    ("seguranca", "Seguranca", r"seguranca"),
    ("inspecao_visual", "Inspecao visual", r"inspecao visual"),
    ("diagnostico_vibracao", "Diagnostico por vibracao",
     r"diagnostico por vibracao|diagnostico inicial|analise espectral|"
     r"medicao de vibracao|caracteristicas vibracionais|frequencias caracteristicas|"
     r"diagnostico por envelope|diagnostico de defeito"),
    ("correcao", "Correcao",
     r"correcao|balanceamento|substituicao|instalacao do novo|preparacao para"),
    ("validacao", "Validacao",
     r"validacao|verificacao final|verificacao dinamica|monitoramento pos"),
    ("criterios_aceitacao", "Criterios de aceitacao", r"criterios de aceitacao"),
    ("registro", "Registro", r"registro d"),
    ("preventivas", "Recomendacoes preventivas",
     r"preventiv|boas praticas|cuidados durante"),
    ("indicadores", "Indicadores de monitoramento", r"indicadores"),
]

# Familias de `fault` do banner.csv que cada documento cobre. CURADO A MAO —
# e o embriao do data/fault_map.yaml (Parte 1) e o que o guardrail G3 vai
# consultar. Nao inferir por similaridade semantica: G3 e um SELECT.
MAPA_DOC_FAMILIA = {
    "Doc1": ["rolamento_inner", "rolamento_outer", "rolamento_ball",
             "rolamento_combination"],
    "Doc2": ["desalinhamento"],
    "Doc3": ["desbalanceamento"],
    "Doc4": ["correia"],
    "Doc5": ["polia"],
    "Doc6": ["cocked_rotor"],
}

# Cobertura parcial: o documento fala do fenomeno, mas em outro componente.
# Nao entra no G3 como documento valido — vira pergunta para a engenharia.
# Doc5 secao 3.1 descreve excentricidade DE POLIA; `eccentric_rotor` no banner
# e excentricidade DE ROTOR. Mesmo nome, componente diferente.
COBERTURA_PARCIAL = {
    "Doc5": ["eccentric_rotor"],
}

# Razao entre eixos. Qual e axial e qual e radial depende da montagem do sensor,
# que o dataset nao informa — por isso o nome e neutro (x/z) e a leitura fica
# como pergunta para o cruzamento com os PDFs na Parte 1.
NUMERADOR_RAZAO = "x_rms_velocity_mm_s"
DENOMINADOR_RAZAO = "z_rms_velocity_mm_s"
NOME_RAZAO = "razao_x_z_rms_vel"
