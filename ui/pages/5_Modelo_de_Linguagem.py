"""Passo 5.1 — escolher o provedor e testar a conexao.

Primeira tela que fala com um modelo de linguagem. Aqui ele ainda nao participa
do pipeline: so provamos que da para trocar de provedor sem mexer no codigo.
"""

from __future__ import annotations

import streamlit as st

import _dados as D

D.configurar_pagina("Modelo de Linguagem", "🤖")

st.title("🤖 Modelo de Linguagem")
st.caption("Escolha o provedor, teste a conexao e converse para conferir.")

st.markdown(
    """
O projeto tem como meta um modelo **local**, rodando na estacao, sem API externa.
Durante o desenvolvimento usamos a API do ChatGPT porque o retorno e imediato — o
que permite fechar o fluxo antes de lidar com desempenho de GPU.

Para isso nao virar retrabalho, **quem escolhe o provedor e voce, aqui**. Trocar de
um para outro nao muda uma linha do resto do sistema.
"""
)

estado = D.r_provedores()

# ==========================================================================
# 1. Situacao de cada provedor
# ==========================================================================
st.subheader("Provedores")

colunas = st.columns(len(estado))
for coluna, (nome, info) in zip(colunas, estado.items()):
    with coluna:
        if info["pronto"]:
            st.success(f"**{nome}**\n\npronto para usar")
        else:
            st.warning(f"**{nome}**\n\n{info['motivo']}")
        st.caption(D.DESCRICAO_PROVEDOR.get(nome, ""))

prontos = [n for n, i in estado.items() if i["pronto"]]
if not prontos:
    st.error(
        "Nenhum provedor disponivel. Para usar a API do ChatGPT, crie um arquivo "
        "`.env` na raiz do projeto com `OPENAI_API_KEY=sk-...`. Para usar o modelo "
        "local, inicie o Ollama com `ollama serve`."
    )
    st.stop()

st.divider()

# ==========================================================================
# 2. Configuracao
# ==========================================================================
st.subheader("Configuracao")

c1, c2 = st.columns(2)
with c1:
    provedor = st.selectbox(
        "Provedor", prontos,
        help="So aparecem os que estao prontos para responder agora.",
    )
with c2:
    modelos = D.r_modelos_do_provedor(provedor)
    modelo = st.selectbox(
        "Modelo", modelos,
        help="No Ollama, sao os modelos realmente baixados na maquina.",
    )

c3, c4 = st.columns(2)
with c3:
    temperatura = st.slider(
        "Temperatura", 0.0, 1.0, 0.1, step=0.05,
        help="Quanto o modelo pode variar a resposta. Perto de zero para "
             "resposta tecnica, porque queremos o mesmo texto para a mesma "
             "pergunta.",
    )
with c4:
    max_tokens = st.slider(
        "Tamanho maximo da resposta", 100, 2000, 800, step=100,
        help="Em tokens — mais ou menos tres quartos disso em palavras.",
    )

st.session_state["llm_config"] = {
    "provedor": provedor, "modelo": modelo,
    "temperatura": temperatura, "max_tokens": max_tokens,
}

st.caption(
    "A escolha fica guardada na sessao e sera usada pelos proximos passos "
    "(resposta prescritiva, ancoragem, conversa)."
)

st.divider()

# ==========================================================================
# 3. Testar
# ==========================================================================
st.subheader("Testar a conexao")

if st.button("Testar", type="primary"):
    with st.spinner(f"Falando com {modelo}..."):
        ok, mensagem = D.testar_llm(provedor, modelo, temperatura, max_tokens)
    if ok:
        st.success(f"**Conectou.** {mensagem}")
    else:
        st.error(f"**Nao conectou.** {mensagem}")

st.divider()

# ==========================================================================
# 4. Conversar
# ==========================================================================
st.subheader("Conversar")

st.markdown(
    """
Esta conversa **nao passa pelos guardrails** — e um teste direto com o modelo,
para voce comparar provedores.

E justamente por isso ela serve de contraste: pergunte algo que o manual da empresa
nao cobre e veja o modelo responder mesmo assim, com confianca. E esse
comportamento que os guardrails vao barrar no passo seguinte.
"""
)

sugestoes = [
    "Qual lubrificante usar em rolamento de motor eletrico?",
    "Como corrigir desalinhamento?",
    "Quanto de torque aplicar nos parafusos da base?",
]
escolha = st.selectbox("Sugestoes", ["(escrever a minha)"] + sugestoes)
pergunta = st.text_area(
    "Pergunta",
    value="" if escolha.startswith("(") else escolha,
    placeholder="Escreva a pergunta...",
    height=90,
)

if st.button("Enviar") and pergunta.strip():
    with st.spinner(f"{modelo} pensando..."):
        try:
            r = D.conversar_llm(provedor, modelo, temperatura, max_tokens, pergunta)
        except Exception as e:  # noqa: BLE001 — a mensagem vai para a tela
            st.error(f"{type(e).__name__}: {e}")
            r = None

    if r is not None:
        st.markdown("**Resposta**")
        st.info(r["texto"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Provedor", r["provedor"])
        m2.metric("Modelo", r["modelo"][:22])
        m3.metric("Tempo", f"{r['segundos']} s")
        m4.metric("Tokens", f"{r['tokens_entrada']}+{r['tokens_saida']}")

        st.warning(
            "**Repare:** o modelo respondeu com seguranca, mesmo sem consultar "
            "manual nenhum. Se a resposta citar uma norma, um valor de torque ou um "
            "tipo de graxa, isso veio do treinamento dele — nao do procedimento da "
            "empresa.\n\nE exatamente o que o sistema final precisa impedir."
        )

st.divider()

with st.expander("Como o cliente e plugavel"):
    st.markdown(
        """
Os tres provedores cumprem o mesmo contrato:

```python
cliente = criar("openai", modelo="gpt-4o-mini")
resposta = cliente.gerar([Mensagem("user", "ola")])
```

Trocar `"openai"` por `"ollama"` nao muda mais nada. Cada classe resolve suas
particularidades por dentro — a API da Anthropic, por exemplo, trata o `system`
como parametro separado em vez de mensagem, e a conversao acontece dentro do
proprio cliente.

O que muda entre eles, e que a tela mostra:

| | OpenAI | Ollama |
|---|---|---|
| Onde roda | servidor da empresa | sua maquina |
| Chave | `.env` | nao precisa |
| Custo | por token | zero |
| Velocidade | ~1 s | ~20 s |
| Dado sai da maquina | sim | nao |

A ultima linha e a que decide para a entrega: o procedimento da empresa nao deveria
sair da rede dela.
"""
    )
