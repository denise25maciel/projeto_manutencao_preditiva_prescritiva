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
from mp import config


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

# ==========================================================================
# As duas bases, para baixar
# ==========================================================================
#
# Ficam aqui em cima, e nao so no passo 6 da aba 1, porque sao o entregavel que
# alguem de fora quer primeiro: abrir no Excel e conferir. Enterradas no meio de
# uma narrativa de seis passos, elas existiam sem serem achadas.
#
# A explicacao do corte continua no passo 6 — aqui e so o arquivo.
with st.container(border=True):
    st.markdown("#### 📥 Baixar as bases de treino e teste")

    _tamanho = config.CLF_JANELA_TAMANHO
    _divisao = D.r_clf_divisao(_tamanho, False, True, 0.2)
    _treino, _teste = _divisao["treino"], _divisao["teste"]

    st.caption(
        f"Cada linha e uma janela de {_tamanho} leituras resumida em "
        f"{_treino.shape[1] - 2} numeros. A **primeira coluna e `familia`** — a "
        "resposta certa, que vai junto no arquivo — e a segunda e `evento`, o "
        "grupo que separou o treino do teste. Nenhuma das duas entra no modelo "
        "como feature."
    )

    _c1, _c2 = st.columns(2)
    with _c1:
        st.metric(
            "TREINO",
            f"{len(_treino):,} linhas".replace(",", "."),
            f"{_divisao['eventos_treino']} eventos · "
            f"{_divisao['familias_treino']} familias",
            delta_color="off",
        )
        st.download_button(
            "Baixar treino (CSV)",
            data=D.r_clf_csv_base("treino", _tamanho, False, True),
            file_name=f"treino_janela_{_tamanho}.csv",
            mime="text/csv",
            width="stretch",
            key="dl_treino_topo",
        )
    with _c2:
        st.metric(
            "TESTE",
            f"{len(_teste):,} linhas".replace(",", "."),
            f"{_divisao['eventos_teste']} eventos · "
            f"{_divisao['familias_teste']} familias",
            delta_color="off",
        )
        st.download_button(
            "Baixar teste (CSV)",
            data=D.r_clf_csv_base("teste", _tamanho, False, True),
            file_name=f"teste_janela_{_tamanho}.csv",
            mime="text/csv",
            width="stretch",
            key="dl_teste_topo",
        )

    st.caption(
        f"O corte sorteia **eventos inteiros**: {_divisao['eventos_vazados']} "
        "eventos aparecem nos dois lados. Por isso a proporcao nao sai exata em "
        f"80/20 — deu {100 * (1 - _divisao['fracao_real']):.0f}/"
        f"{100 * _divisao['fracao_real']:.0f}. O porque disso esta na aba 1, "
        "passo 6, onde da para trocar o criterio e ver a diferenca."
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
