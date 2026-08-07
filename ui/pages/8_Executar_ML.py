"""Rodar o pipeline de classificacao, como item proprio do menu.

O mesmo painel existe como terceira aba da tela **Classificacao**. Aqui ele
ganha entrada propria na barra lateral, porque executar o modelo e uma acao que
se procura pelo menu — nao algo que se encontra navegando por abas dentro de
outra tela.

**Nao ha conteudo duplicado.** As duas entradas chamam o mesmo
`_secao_clf_execucao.render()`, e compartilham o `st.session_state`: rodar aqui
deixa o resultado visivel la, e vice-versa.

Rodar com:  streamlit run ui/app.py
"""

from __future__ import annotations

import streamlit as st

import _dados as D  # noqa: F401  — garante o sys.path de `src/` antes do resto
import _secao_clf_execucao


D.configurar_pagina("Executar ML", "▶️")

st.title("▶️ Executar o modelo de classificacao")
st.caption(
    "O pipeline inteiro, do CSV bruto ao laudo dos testes — cada etapa "
    "cronometrada, sem cache."
)

_secao_clf_execucao.render()
