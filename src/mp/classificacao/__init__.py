"""Classificacao supervisionada da familia a partir de um trecho de leituras.

Vem do projeto irmao de classificacao (`prep.py`, `sistema.py`, `avaliacao.py`),
adaptado para usar as pecas deste aqui: o rotulo sai do `fault_map.yaml`, o
grupo e o evento de `ingestion.construir_eventos`, e as colunas sao as mesmas
que o motor de similaridade compara.

- `amostras`  — de leitura crua para a matriz (o que era `prep.py`)
- `modelo`    — a floresta e a consulta (o que era `sistema.py`)
- `validacao` — as duas estrategias de corte (o que era `avaliacao.py`)
- `execucao`  — os tres acima rodando em ordem, cronometrados, com relatorio

`execucao` **nao** e reexportado aqui de proposito: ele importa este pacote, e
puxa-lo de volta para o `__init__` fecharia um ciclo de import. Quem precisa
dele escreve `from mp.classificacao.execucao import executar_pipeline`, ou roda
`python -m mp.classificacao.execucao` para ver o relatorio no terminal.

Papel no sistema: **sinal auxiliar**, o `[R2]` que o GUIA.md previu. Nao decide
manual, nao entra no caminho do LLM, nao substitui guardrail. A familia que
autoriza a prescricao continua vindo do rotulo pelo catalogo.
"""

from mp.classificacao.amostras import (
    ESTATISTICAS,
    cobertura_dos_eventos,
    colunas_de_entrada,
    criar_amostras,
    matriz,
    matriz_legivel,
    nomes_das_features,
    preparar,
    resumir_bloco,
    tabela_de_estatisticas,
)
from mp.classificacao.modelo import Classificador, prever_evento_segurado, treinar
from mp.classificacao.validacao import (
    ESTRATEGIAS,
    acerto_por_familia,
    dividir_treino_teste,
    experimento_janela,
    experimento_regime,
    linha_de_base,
    matriz_de_confusao,
    validar,
)

__all__ = [
    "ESTATISTICAS",
    "ESTRATEGIAS",
    "Classificador",
    "acerto_por_familia",
    "cobertura_dos_eventos",
    "colunas_de_entrada",
    "criar_amostras",
    "dividir_treino_teste",
    "experimento_janela",
    "experimento_regime",
    "linha_de_base",
    "matriz",
    "matriz_de_confusao",
    "matriz_legivel",
    "nomes_das_features",
    "preparar",
    "prever_evento_segurado",
    "resumir_bloco",
    "tabela_de_estatisticas",
    "treinar",
    "validar",
]
