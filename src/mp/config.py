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

# Acima disso consideramos que houve corte entre CAMPANHAS de coleta. Usado para
# nao medir "taxa de amostragem" atravessando a fronteira, e para quebrar a linha
# dos graficos de serie temporal.
GAP_NOVA_SESSAO_S = 60.0

# Colunas cuja mudanca encerra um evento.
#
# `fault` e obvia: outro defeito, outro evento.
#
# `rpm` entrou depois de uma verificacao. Com a quebra so por rotulo, **136 dos
# 205 eventos misturavam rotacoes** — 95% das leituras. A bancada rodava 500,
# 1000 e 2000 rpm em sequencia sem trocar o nome da falha, e os tres ensaios
# viravam um evento so.
#
# O caso extremo: o evento de `rolamento_combination_pos_2` tinha velocidade RMS
# de 3,5 em 500 rpm e 21,1 em 2000 rpm — seis vezes maior, dentro do "mesmo"
# evento. A mediana dele nao descrevia nenhum dos tres regimes.
#
# Incluir o rpm leva a dispersao interna tipica de 2,40 para 1,31 e zera os
# eventos com regime misturado.
COLUNAS_QUEBRA_EVENTO = (COLUNA_ROTULO, "rpm")

# Corte que separa um ENSAIO do seguinte — a fronteira do episodio (Parte 1).
# Menor que o de campanha de proposito: o operador leva de 20 a 45 s trocando o
# arranjo da bancada entre dois ensaios, e com 60 s dois ensaios consecutivos do
# mesmo rotulo ficariam colados.
#
# O numero nao e chute. Olhando os intervalos entre leituras do mesmo rotulo, os
# dados tem uma faixa COMPLETAMENTE VAZIA:
#
#     maior intervalo de coleta normal :  6,000 s
#     menor pausa de verdade           : 16,085 s
#     entre os dois: 0 ocorrencias em 166.591 intervalos
#
# Qualquer corte dentro dessa faixa produz os mesmos 570 episodios — testado com
# 8, 10, 12 e 15 s. Abaixo de 6 s a cadencia de 5,3 s se parte (11 mil episodios);
# acima de 16 s comecam a se fundir ensaios distintos (366 episodios com 60 s).
#
# Escolhemos 10 s por ser o centro da faixa vazia: maior margem dos dois lados se
# uma coleta futura tiver ritmo um pouco diferente.
# Justificativa completa na tela "Qualidade dos Dados", secao 2.
GAP_NOVO_EPISODIO_S = 10.0

# Teto de pontos enviados ao navegador num grafico de serie temporal. Acima
# disso a serie e reamostrada por blocos (mediana + faixa min-max), o que
# preserva os picos — que em vibracao sao o sinal, nao ruido.
MAX_PONTOS_SERIE = 3000

# Teto de pontos da PAGINA inteira, somando todos os graficos.
# A tela de analise plota as 23 colunas numericas empilhadas, para ate 4
# rotulos: sao ate 92 series simultaneas. Sem um teto global, o navegador
# receberia centenas de milhares de pontos e travaria. O orcamento por serie
# vira MAX_PONTOS_PAGINA / (rotulos x colunas), limitado a MAX_PONTOS_SERIE.
MAX_PONTOS_PAGINA = 30_000

# Piso por serie: abaixo disso a linha perde a forma e o grafico nao informa.
MIN_PONTOS_SERIE = 200

# --------------------------------------------------------------------------
# Conversa (Parte 5)
# --------------------------------------------------------------------------

# Trava de escopo (no 5 do grafo). O manual e fixado no turno 1 e nao muda; uma
# pergunta cuja MELHOR semelhanca dentro dele fica abaixo disto nao trata do
# assunto do manual e nao chega ao modelo.
#
# Medido contra o Doc2 com 8 perguntas dentro do assunto e 6 fora:
#
#   dentro   0.175 a 0.707   (mediana 0.538)
#   fora     0.088 a 0.196   (mediana 0.159)
#
# **As faixas se sobrepoem.** "Como sei que ficou bom?" (dentro) marca 0.175 e
# fica ABAIXO de "Qual o melhor carro para comprar?" (fora, 0.196). Nenhum
# limiar unico separa as duas listas — a pergunta curta e generica se parece
# pouco com qualquer texto, e a semelhanca de cosseno nao distingue "vago" de
# "off-topic".
#
# 0.17 e o valor que barra 5 das 6 perguntas de fora sem barrar nenhuma de
# dentro. **Com 14 perguntas, e ajuste a uma amostra pequena, nao calibracao.**
#
# Consequencia assumida: esta trava e um filtro barato, nao uma garantia. O que
# garante e a sequencia depois dela — G4 corta o trecho fraco, G5 exige citacao
# real, e a degradacao entrega o texto do manual quando nada disso se sustenta.
LIMIAR_ESCOPO = 0.17

# Quantos turnos anteriores entram no prompt. Historico longo demais dilui os
# trechos do manual, que sao o que importa.
MAX_TURNOS_NO_PROMPT = 4

# --------------------------------------------------------------------------
# Escolha do manual: quando a evidencia nao aponta um so
# --------------------------------------------------------------------------
#
# Os dois limiares abaixo decidem uma coisa so: **o sistema trava o manual
# sozinho, ou mostra a lista e deixa o tecnico escolher?** Nao existe terceiro
# desfecho — nao ha rodada de investigacao nem teto de tentativas.
#
# Isso muda o custo de errar para cada lado, e por isso os numeros podem ser
# conservadores sem prejuizo: travar errado contamina a conversa inteira, ja que
# o manual nao muda depois de fixado; mostrar a lista custa um clique a quem
# esta na maquina e costuma reconhecer o defeito de imediato. Na duvida,
# perguntar sai mais barato.

# Margem relativa entre o 1o e o 2o documento: (p1 - p2) / p1. Abaixo disto a
# sessao NAO trava — a lista de candidatos vai para a tela.
#
# Medido com 10 descricoes, somando o peso por documento:
#
#   ambiguas    1,2%   3,3%   5,7%   13,8%
#   decididas   46,2%  47,3%  50,8%  52,7%  62,2%  81,4%
#
# O vao entre 13,8% e 46,2% e largo, e 0,25 fica no meio dele.
#
# **A margem mede ambiguidade, nao vagueza** — e a distincao importa. Frase
# especifica pode ser ambigua: "ruido de impacto repetitivo no rolamento" da
# margem de 3,3% porque Doc1 e Doc3 descrevem isso quase igual, e ai perguntar
# mais e o certo. Frase vaga pode ser decidida: "barulho estranho" aponta Doc1
# com 50,8%. Gatilhar por ambiguidade e o comportamento desejado; gatilhar por
# vagueza seria adivinhar a intencao de quem escreveu.
#
# O que a margem **nao** garante: que o vencedor esta certo. Ela diz que a
# evidencia foi consistente, nao que foi boa. O G4 continua sendo quem barra
# evidencia fraca.
#
# 10 frases e amostra pequena. Como o LIMIAR_ESCOPO, isto e ajuste, nao
# calibracao — o que sustenta o fluxo e a escolha humana no fim, nao o numero.
MARGEM_MINIMA_DOCUMENTO = 0.25

# Fatia minima do peso total que o documento vencedor precisa concentrar.
#
# A margem so olha o 1o contra o 2o, e por isso deixa passar um caso que ela nao
# enxerga: pesos [1,0; 0,5; 0,5; 0,5; 0,5] dao margem de 50% — folgada —, mas o
# lider concentra apenas 33% da evidencia, e ha QUATRO outros manuais ainda
# plausiveis. E a diferenca entre "ganhou do segundo" e "ganhou de todos".
#
# Medido nas mesmas 10 descricoes:
#
#   ambiguas    24,7%  26,2%  26,2%  39,7%
#   decididas   49,1%  49,5%  49,8%  50,5%  61,7%  74,8%
#
# 0,45 fica no vao entre 39,7% e 49,1%. As duas condicoes valem juntas: passar
# na margem e nao concentrar evidencia continua sendo motivo para perguntar.
#
# **Limitacao conhecida do share.** Sendo `p1/total`, ele depende de quantos
# documentos aparecem no top-k: com 4 candidatos, dividir por igual da 25% e o
# minimo de 45% exige quase metade de tudo; com 2 candidatos, 45% e quase de
# graca. O limiar nao exige o mesmo esforco em situacoes diferentes. Com a lista
# na tela isso deixou de ser grave — o efeito e mostrar candidatos a mais, nao
# travar no manual errado —, mas continua sendo motivo para normalizar pelo
# numero de candidatos numa proxima rodada.
SHARE_MINIMO_DOCUMENTO = 0.45

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
