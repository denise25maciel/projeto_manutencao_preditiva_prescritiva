"""O projeto de classificacao supervisionada, trazido para dentro deste sistema.

Vem de um repositorio irmao, sobre **os mesmos dados**: um `prep.py` que
transforma o `banner.csv` em exemplos, e um par `sistema.py` / `avaliacao.py`
que treina uma floresta e mede o que ela vale. As duas abas desta tela sao esses
dois arquivos, adaptados.

**Adaptados, e nao copiados.** O algoritmo veio inteiro — mesmo recorte em
janelas, mesmas cinco estatisticas, mesma floresta, mesmas duas estrategias de
validacao. O que trocou foi de onde vem cada *decisao*, e sao tres:

    o rotulo    regras de normalizacao no codigo  ->  data/fault_map.yaml
    o grupo     mudou o texto de `fault`          ->  evento (`fault` + `rpm`)
    as colunas  lista propria, com rpm e temp.    ->  as mesmas do kNN

Cada troca elimina uma segunda fonte de verdade para uma pergunta que este
projeto ja respondia em outro lugar. O detalhe de cada uma esta na aba 1.

**Tres abas, e nao tres telas.** A pergunta que a aba 2 responde — "quanto vale
este numero?" — so faz sentido depois de saber o que e um exemplo, que e a aba
1. Separar em telas deixaria a acuracia acessivel sem a construcao que a explica,
e e justamente a construcao que mostra por que 92% e um numero falso.

    aba 1   preparacao   como uma leitura vira um exemplo
    aba 2   o modelo     o que a acuracia significa, e os experimentos
    aba 3   execucao     o pipeline rodando de verdade, e o laudo dos testes

As duas primeiras **argumentam** e leem de cache; a terceira **executa**, sem
cache nenhum, para o tempo que ela mostra ser o tempo que ela levou.

O codigo vive em `src/mp/classificacao/`; aqui so ha chamada e desenho.
"""

from __future__ import annotations

import streamlit as st

import _dados as D
import _secao_clf_execucao
import _secao_clf_modelo
import _secao_clf_preparo


D.configurar_pagina("Classificacao", "🌲")

st.title("🌲 Classificacao — prever a familia pelos numeros")
st.caption(
    "O mesmo arquivo, sem os manuais: da leitura crua ao palpite, e quanto esse "
    "palpite vale."
)

st.markdown(
    """
As outras telas partem de um evento **ja rotulado** e vao ate o procedimento.
Esta faz a pergunta anterior: **da para descobrir a familia so pelos numeros do
sensor, sem ninguem anotar?**

E um projeto que existia separado, sobre os mesmos dados, e que agora roda sobre
as pecas deste — o rotulo sai do `fault_map.yaml`, o grupo e o evento, as
colunas sao as mesmas que o motor de similaridade compara.
    """
)

aba_preparo, aba_modelo, aba_execucao = st.tabs(
    [
        "1 · Preparacao dos dados",
        "2 · O modelo e o que ele vale",
        "3 · Executar e ver o resultado",
    ]
)

with aba_preparo:
    _secao_clf_preparo.render()

with aba_modelo:
    _secao_clf_modelo.render()

with aba_execucao:
    _secao_clf_execucao.render()
