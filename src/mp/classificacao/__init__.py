"""Classificacao supervisionada da familia a partir de um trecho de leituras.

Vem do projeto irmao (`prep.py`, `sistema.py`, `avaliacao.py`), adaptado para
usar as pecas deste: o rotulo sai do `fault_map.yaml` e o grupo e o evento.

- `colunas`   — quais medidas entram, e por que as outras saem
- `amostras`  — de leitura crua para a matriz
- `modelo`    — a floresta e a consulta
- `consulta`  — o veredito sobre um trecho novo, para a conversa
- `validacao` — as duas estrategias de corte
- `execucao`  — os anteriores rodando em ordem, cronometrados

`execucao` **nao** e reexportado: ele importa este pacote, e puxa-lo de volta
fecharia um ciclo. Use `from mp.classificacao.execucao import ...`.
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
from mp.classificacao.colunas import colunas_de_medida
from mp.classificacao.consulta import Classificacao, classificar_bloco
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
    "Classificacao",
    "Classificador",
    "classificar_bloco",
    "colunas_de_medida",
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
