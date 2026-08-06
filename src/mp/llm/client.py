"""Cliente de modelo de linguagem, com provedores intercambiaveis.

O projeto tem como meta um modelo **local** (Ollama). Durante o desenvolvimento
usamos a API do ChatGPT, porque a chave ja existe e o retorno e imediato — o que
permite fechar o fluxo antes de lidar com download de modelo e GPU.

Para isso funcionar sem retrabalho, o contrato e o mesmo em todos:

    cliente = criar("openai", modelo="gpt-4o-mini")
    resposta = cliente.gerar([Mensagem("user", "ola")])

Trocar `"openai"` por `"ollama"` nao muda mais nada. **Quem escolhe e o usuario,
na interface** — nao uma constante no codigo.

A chave nunca aparece aqui. Vem do `.env`, que esta no `.gitignore`.

Onde o LangChain entra — e onde nao entra
-----------------------------------------
Ele entra **neste arquivo e so neste arquivo**, por dois motivos concretos:

1. **Mensagens tipadas.** `SystemMessage`, `HumanMessage` e `AIMessage` sao o
   vocabulario comum dos tres provedores. Antes, cada um recebia um `dict` e a
   API da Anthropic exigia extrair o `system` na mao — agora e o adaptador que
   resolve.
2. **Saida estruturada.** `with_structured_output` funciona igual nos tres. A
   mao, seriam tres formatos diferentes de *tool call* (OpenAI *functions*,
   Anthropic *tool_use*, Ollama). E dessa uniformidade que depende o
   `estruturar`, usado para separar os sintomas do tecnico.

Ele **nao** entra no RAG nem no grafo. A busca em dois estagios (`SELECT` e
depois cosseno) e a ordem dos nos sao a tese do projeto; um `retriever` generico
faria funcionar e apagaria o argumento. O framework aqui resolve heterogeneidade
de provedor, nunca decisao.

Como consequencia disso, `gerar` ficou **uma implementacao so**, na classe base:
o que cada provedor ainda tem de proprio e como se conecta e como traduz as
opcoes por chamada.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Carrega o `.env` uma vez, na importacao. `override=False` respeita variaveis
# ja definidas no ambiente — util em container, onde a chave vem de fora.
try:
    from dotenv import load_dotenv

    from mp import config

    load_dotenv(config.RAIZ / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------
# Mensagens
# --------------------------------------------------------------------------

# Os tres tipos de fala, pelos nomes que o projeto ja usava. `user` e `human`
# apontam para a mesma classe porque a API fala "user" e o LangChain fala
# "human" — quem escreve o codigo nao deveria ter de saber disso.
_CLASSES = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
}

# Rotulos para exibir na auditoria. Sao os `type` do LangChain, nao os papeis
# de entrada: e o que a mensagem virou, que e o que interessa conferir.
ROTULO_DA_MENSAGEM = {"system": "SYSTEM", "human": "HUMAN", "ai": "AI"}


def Mensagem(papel: str, conteudo: str) -> BaseMessage:  # noqa: N802
    """`Mensagem("system", ...)` -> `SystemMessage(...)`.

    **E uma fabrica, nao uma classe** — dai o nome em maiuscula. O projeto ja
    escrevia `Mensagem("user", pergunta)` em varios lugares, e o valor de
    trocar isso por tres imports diferentes seria zero; o valor esta em o que
    sai daqui ser a mensagem tipada do LangChain, que os tres provedores
    entendem sem conversao.
    """
    classe = _CLASSES.get(papel.lower())
    if classe is None:
        raise ValueError(
            f"papel '{papel}' desconhecido. Use: {', '.join(sorted(set(_CLASSES)))}"
        )
    return classe(content=conteudo)


def como_texto(mensagens: list[BaseMessage]) -> str:
    """As mensagens como um texto so, com o tipo de cada uma na frente.

    E o que a tela de auditoria mostra: o `[SYSTEM]` / `[HUMAN]` / `[AI]` na
    frente de cada bloco prova que o historico chegou **estruturado**, e nao
    achatado dentro de um paragrafo.
    """
    partes = []
    for m in mensagens:
        rotulo = ROTULO_DA_MENSAGEM.get(m.type, m.type.upper())
        partes.append(f"[{rotulo}]\n{_conteudo_em_texto(m.content)}")
    return "\n\n".join(partes)


def _conteudo_em_texto(conteudo: Any) -> str:
    """O texto de uma mensagem.

    O conteudo nem sempre e `str`: a Anthropic devolve uma lista de blocos, e
    so os de tipo `text` interessam aqui.
    """
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        return "".join(
            bloco.get("text", "") if isinstance(bloco, dict) else str(bloco)
            for bloco in conteudo
        )
    return str(conteudo)


@dataclass
class Resposta:
    """O que voltou do modelo, mais o que custou.

    Guardar tokens e tempo nao e enfeite: e o que permite comparar provedores e
    justificar a escolha do modelo local depois.
    """

    texto: str
    provedor: str
    modelo: str
    tokens_entrada: int = 0
    tokens_saida: int = 0
    segundos: float = 0.0
    bruto: dict = field(default_factory=dict, repr=False)

    @property
    def tokens_total(self) -> int:
        return self.tokens_entrada + self.tokens_saida


class Cliente(Protocol):
    """O contrato. Quem consome nao sabe qual provedor esta por tras."""

    provedor: str
    modelo: str

    def gerar(self, mensagens: list[BaseMessage], **kwargs) -> Resposta: ...
    def estruturar(self, mensagens: list[BaseMessage], esquema): ...
    def testar(self) -> tuple[bool, str]: ...
    def modelos(self) -> list[str]: ...


# --------------------------------------------------------------------------
# Base comum
# --------------------------------------------------------------------------


class _ClienteLangChain:
    """O que os tres provedores tem em comum — que hoje e quase tudo.

    Cada subclasse informa tres coisas: como construir o chat model
    (`_construir`), como se chama (`provedor`) e como traduzir as opcoes por
    chamada (`TRADUCAO`). O resto — gerar, medir, estruturar, testar — e daqui.
    """

    provedor = ""
    MODELOS: list[str] = []

    # `nome nosso -> nome do provedor`. Existe porque "tamanho maximo da
    # resposta" se chama `max_tokens` em dois deles e `num_predict` no Ollama, e
    # esconder isso e justamente o trabalho do adaptador.
    TRADUCAO = {
        "temperatura": "temperature",
        "max_tokens": "max_tokens",
        "modelo": "model",
    }

    def __init__(self, modelo: str = "", temperatura: float = 0.1,
                 max_tokens: int = 800, timeout: float = 60.0):
        self.modelo = modelo
        self.temperatura = temperatura
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._lc = None

    # --- o que cada provedor precisa dizer ---------------------------------

    def _construir(self):
        """O chat model do LangChain, ja configurado. Implementado na subclasse."""
        raise NotImplementedError

    def _modelo_lc(self):
        """Constroi na primeira vez e reaproveita — conectar custa."""
        if self._lc is None:
            self._lc = self._construir()
        return self._lc

    def _opcoes(self, kwargs: dict) -> dict:
        """As opcoes desta chamada, com os nomes que o provedor espera."""
        return {
            self.TRADUCAO[chave]: valor
            for chave, valor in kwargs.items()
            if chave in self.TRADUCAO and valor is not None
        }

    # --- o contrato ---------------------------------------------------------

    def modelos(self) -> list[str]:
        return list(self.MODELOS)

    def gerar(self, mensagens: list[BaseMessage], **kwargs) -> Resposta:
        """Manda as mensagens e devolve texto, tokens e tempo.

        Uma implementacao para os tres. O `.bind()` aplica so o que veio nesta
        chamada; sem `kwargs`, usa o que foi configurado na criacao.
        """
        lc = self._modelo_lc()
        if opcoes := self._opcoes(kwargs):
            lc = lc.bind(**opcoes)

        t0 = time.time()
        r = lc.invoke(list(mensagens))

        # `usage_metadata` e o formato unico do LangChain. Antes, o Ollama exigia
        # ler `prompt_eval_count` e a OpenAI `prompt_tokens` — dois nomes para a
        # mesma contagem, cada um lido num lugar diferente do codigo.
        uso = getattr(r, "usage_metadata", None) or {}
        return Resposta(
            texto=_conteudo_em_texto(r.content),
            provedor=self.provedor,
            modelo=self._modelo_respondido(r),
            tokens_entrada=int(uso.get("input_tokens", 0) or 0),
            tokens_saida=int(uso.get("output_tokens", 0) or 0),
            segundos=round(time.time() - t0, 2),
        )

    def estruturar(self, mensagens: list[BaseMessage], esquema):
        """Devolve uma instancia de `esquema` (classe Pydantic), nao texto livre.

        Por baixo e *tool calling*: o esquema vira a assinatura de uma ferramenta,
        o modelo preenche os campos e o LangChain valida o retorno. Cada provedor
        tem um formato proprio para isso; aqui a chamada e a mesma.

        **Isto nao devolve decisao ao modelo.** A unica ferramenta que existe e a
        de preencher um formulario, e o formulario e nosso. Ele nao escolhe se
        chama, nao escolhe qual, e nao ha ramo do fluxo dependendo do que ele
        responder — quem confere o resultado e codigo, do lado de fora.
        """
        return self._modelo_lc().with_structured_output(esquema).invoke(list(mensagens))

    def testar(self) -> tuple[bool, str]:
        try:
            r = self.gerar([Mensagem("user", "Responda apenas: ok")], max_tokens=10)
            return True, f"{r.modelo} respondeu em {r.segundos}s: {r.texto.strip()[:40]}"
        except Exception as e:  # noqa: BLE001 — a mensagem vai para a tela
            return False, f"{type(e).__name__}: {e}"

    # --- detalhes -----------------------------------------------------------

    def _modelo_respondido(self, r) -> str:
        """O modelo que de fato respondeu, que nem sempre e o que foi pedido.

        `gpt-4o-mini` volta como `gpt-4o-mini-2024-07-18`, e essa versao exata e
        o que se quer registrar no turno. Cada provedor guarda isso sob uma
        chave diferente do `response_metadata`.
        """
        meta = getattr(r, "response_metadata", None) or {}
        for chave in ("model_name", "model", "model_id"):
            if valor := meta.get(chave):
                return str(valor)
        return self.modelo


# --------------------------------------------------------------------------
# OpenAI — para desenvolver
# --------------------------------------------------------------------------


class ClienteOpenAI(_ClienteLangChain):
    """API do ChatGPT. Exige `OPENAI_API_KEY` no `.env`."""

    provedor = "openai"

    # Modelos baratos e rapidos primeiro: o uso aqui e redacao curta sobre
    # trechos ja recuperados, nao raciocinio longo.
    MODELOS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

    def __init__(self, modelo: str = "gpt-4o-mini", **kwargs):
        super().__init__(modelo=modelo, **kwargs)

    def _construir(self):
        from langchain_openai import ChatOpenAI

        chave = os.environ.get("OPENAI_API_KEY")
        if not chave:
            raise RuntimeError(
                "OPENAI_API_KEY nao encontrada. Crie um arquivo `.env` na raiz "
                "do projeto com a linha `OPENAI_API_KEY=sk-...`."
            )
        return ChatOpenAI(
            model=self.modelo, api_key=chave, temperature=self.temperatura,
            max_tokens=self.max_tokens, timeout=self.timeout,
        )


# --------------------------------------------------------------------------
# Ollama — a meta
# --------------------------------------------------------------------------


class ClienteOllama(_ClienteLangChain):
    """Modelo local via Ollama, no servico em `localhost:11434`.

    O `instalados()` continua sendo uma chamada REST direta: e a lista de
    modelos baixados na maquina, que nao e conversa com o modelo e nao tem
    equivalente no adaptador.
    """

    provedor = "ollama"

    MODELOS = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "gemma2:9b"]

    # No Ollama, "tamanho maximo da resposta" e `num_predict`.
    TRADUCAO = {
        "temperatura": "temperature",
        "max_tokens": "num_predict",
        "modelo": "model",
    }

    def __init__(self, modelo: str | None = None, temperatura: float = 0.1,
                 max_tokens: int = 800, url: str | None = None,
                 timeout: float = 180.0):
        # Modelo local em CPU pode demorar bem mais que uma API.
        super().__init__(modelo=modelo or "", temperatura=temperatura,
                         max_tokens=max_tokens, timeout=timeout)
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        # Sem modelo escolhido, usa o primeiro que estiver baixado na maquina.
        # Fixar um nome popular daria 404 sempre que ele nao estivesse la.
        if not self.modelo:
            baixados = self.instalados()
            self.modelo = baixados[0] if baixados else self.MODELOS[0]

    def _construir(self):
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self.modelo, base_url=self.url, temperature=self.temperatura,
            num_predict=self.max_tokens,
        )

    def modelos(self) -> list[str]:
        """Os modelos realmente baixados. Cai na lista sugerida se o servico nao responde."""
        return self.instalados() or list(self.MODELOS)

    def instalados(self) -> list[str]:
        """So os modelos realmente baixados. Lista vazia se o servico nao responde."""
        import httpx

        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=5.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return []

    def testar(self) -> tuple[bool, str]:
        baixados = self.instalados()
        if not baixados:
            return False, (
                f"Ollama nao respondeu em {self.url}. O servico esta rodando? "
                "Inicie com `ollama serve`."
            )

        # O Ollama devolve 404 quando o modelo pedido nao esta baixado, e a
        # mensagem crua nao explica isso. Conferimos antes para o erro ser util.
        if self.modelo not in baixados:
            return False, (
                f"O modelo '{self.modelo}' nao esta baixado. "
                f"Disponiveis: {', '.join(baixados)}. "
                f"Para baixar: `ollama pull {self.modelo}`."
            )

        return super().testar()


# --------------------------------------------------------------------------
# Anthropic — opcional
# --------------------------------------------------------------------------


class ClienteAnthropic(_ClienteLangChain):
    """API da Anthropic. Exige `ANTHROPIC_API_KEY` no `.env`.

    Aqui so para provar que o contrato aguenta um terceiro provedor sem remendo.
    O `system` e um parametro separado nessa API, e nao uma mensagem — a
    conversao era feita a mao dentro de `gerar` e hoje e o adaptador que faz.
    """

    provedor = "anthropic"

    MODELOS = ["claude-sonnet-4-5", "claude-haiku-4-5"]

    def __init__(self, modelo: str = "claude-haiku-4-5", **kwargs):
        super().__init__(modelo=modelo, **kwargs)

    def _construir(self):
        from langchain_anthropic import ChatAnthropic

        chave = os.environ.get("ANTHROPIC_API_KEY")
        if not chave:
            raise RuntimeError("ANTHROPIC_API_KEY nao encontrada no `.env`.")
        return ChatAnthropic(
            model=self.modelo, api_key=chave, temperature=self.temperatura,
            max_tokens=self.max_tokens, timeout=self.timeout,
        )


# --------------------------------------------------------------------------
# Fabrica
# --------------------------------------------------------------------------

PROVEDORES = {
    "openai": ClienteOpenAI,
    "ollama": ClienteOllama,
    "anthropic": ClienteAnthropic,
}

# Como cada provedor se identifica na tela, e o que ele exige para funcionar.
DESCRICAO = {
    "openai": "API do ChatGPT — para desenvolver. Exige OPENAI_API_KEY no .env.",
    "ollama": "Modelo local — a meta do projeto. Exige o servico Ollama rodando.",
    "anthropic": "API da Anthropic — opcional. Exige ANTHROPIC_API_KEY no .env.",
}


def criar(provedor: str = "openai", **kwargs) -> Cliente:
    """Devolve o cliente do provedor pedido."""
    if provedor not in PROVEDORES:
        raise ValueError(
            f"provedor '{provedor}' desconhecido. Use: {', '.join(PROVEDORES)}"
        )
    return PROVEDORES[provedor](**kwargs)


def provedores_disponiveis() -> dict:
    """Quais provedores dao para usar agora, e por que nao os outros.

    Nao chama a API — so verifica se o adaptador esta instalado e se a chave
    existe. A tela usa isso para nao oferecer o que vai falhar.
    """
    estado = {}

    def _tem(pacote: str) -> bool:
        try:
            __import__(pacote)
            return True
        except ImportError:
            return False

    for nome, pacote, variavel in (
        ("openai", "langchain_openai", "OPENAI_API_KEY"),
        ("anthropic", "langchain_anthropic", "ANTHROPIC_API_KEY"),
    ):
        tem_pacote = _tem(pacote)
        tem_chave = bool(os.environ.get(variavel))
        estado[nome] = {
            "pronto": tem_pacote and tem_chave,
            "motivo": "" if tem_pacote and tem_chave
            else (f"pacote `{pacote}` nao instalado" if not tem_pacote
                  else f"{variavel} ausente no .env"),
        }

    # O Ollama nao tem chave: ou o servico responde, ou nao ha o que oferecer.
    try:
        import httpx

        httpx.get(
            (os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
            + "/api/tags",
            timeout=2.0,
        ).raise_for_status()
        estado["ollama"] = {"pronto": _tem("langchain_ollama"), "motivo":
                            "" if _tem("langchain_ollama")
                            else "pacote `langchain_ollama` nao instalado"}
    except Exception:  # noqa: BLE001
        estado["ollama"] = {"pronto": False, "motivo": "servico Ollama nao respondeu"}

    # A ordem importa para a tela: os cartoes saem nesta sequencia.
    return {n: estado[n] for n in ("openai", "ollama", "anthropic")}
