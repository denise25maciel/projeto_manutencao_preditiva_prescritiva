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
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

# Carrega o `.env` uma vez, na importacao. `override=False` respeita variaveis
# ja definidas no ambiente — util em container, onde a chave vem de fora.
try:
    from dotenv import load_dotenv

    from mp import config

    load_dotenv(config.RAIZ / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


@dataclass
class Mensagem:
    """Uma fala. `papel` e 'system', 'user' ou 'assistant'."""

    papel: str
    conteudo: str

    def como_dict(self) -> dict:
        return {"role": self.papel, "content": self.conteudo}


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

    def gerar(self, mensagens: list[Mensagem], **kwargs) -> Resposta: ...
    def testar(self) -> tuple[bool, str]: ...
    def modelos(self) -> list[str]: ...


# --------------------------------------------------------------------------
# OpenAI — para desenvolver
# --------------------------------------------------------------------------


class ClienteOpenAI:
    """API do ChatGPT. Exige `OPENAI_API_KEY` no `.env`."""

    provedor = "openai"

    # Modelos baratos e rapidos primeiro: o uso aqui e redacao curta sobre
    # trechos ja recuperados, nao raciocinio longo.
    MODELOS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

    def __init__(self, modelo: str = "gpt-4o-mini", temperatura: float = 0.1,
                 max_tokens: int = 800, timeout: float = 60.0):
        self.modelo = modelo
        self.temperatura = temperatura
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._cliente = None

    def _conectar(self):
        if self._cliente is None:
            from openai import OpenAI

            chave = os.environ.get("OPENAI_API_KEY")
            if not chave:
                raise RuntimeError(
                    "OPENAI_API_KEY nao encontrada. Crie um arquivo `.env` na raiz "
                    "do projeto com a linha `OPENAI_API_KEY=sk-...`."
                )
            self._cliente = OpenAI(api_key=chave, timeout=self.timeout)
        return self._cliente

    def modelos(self) -> list[str]:
        return list(self.MODELOS)

    def gerar(self, mensagens: list[Mensagem], **kwargs) -> Resposta:
        cliente = self._conectar()
        t0 = time.time()
        r = cliente.chat.completions.create(
            model=kwargs.get("modelo", self.modelo),
            messages=[m.como_dict() for m in mensagens],
            temperature=kwargs.get("temperatura", self.temperatura),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        uso = r.usage
        return Resposta(
            texto=r.choices[0].message.content or "",
            provedor=self.provedor,
            modelo=r.model,
            tokens_entrada=getattr(uso, "prompt_tokens", 0) or 0,
            tokens_saida=getattr(uso, "completion_tokens", 0) or 0,
            segundos=round(time.time() - t0, 2),
        )

    def testar(self) -> tuple[bool, str]:
        try:
            r = self.gerar([Mensagem("user", "Responda apenas: ok")], max_tokens=10)
            return True, f"{r.modelo} respondeu em {r.segundos}s: {r.texto.strip()[:40]}"
        except Exception as e:  # noqa: BLE001 — a mensagem vai para a tela
            return False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Ollama — a meta
# --------------------------------------------------------------------------


class ClienteOllama:
    """Modelo local via Ollama. Fala HTTP com o servico em `localhost:11434`.

    Sem SDK de proposito: e uma chamada REST, e `httpx` ja esta no projeto.
    Menos uma dependencia para instalar na estacao de producao.
    """

    provedor = "ollama"

    MODELOS = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "gemma2:9b"]

    def __init__(self, modelo: str | None = None, temperatura: float = 0.1,
                 max_tokens: int = 800, url: str | None = None,
                 timeout: float = 180.0):
        # Sem modelo escolhido, usa o primeiro que estiver baixado na maquina.
        # Fixar um nome popular daria 404 sempre que ele nao estivesse la.
        self.modelo = modelo or ""
        self.temperatura = temperatura
        self.max_tokens = max_tokens
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        # Modelo local em CPU pode demorar bem mais que uma API.
        self.timeout = timeout
        if not self.modelo:
            baixados = self.instalados()
            self.modelo = baixados[0] if baixados else self.MODELOS[0]

    def modelos(self) -> list[str]:
        """Os modelos realmente baixados. Cai na lista sugerida se o servico nao responde."""
        return self.instalados() or list(self.MODELOS)

    def gerar(self, mensagens: list[Mensagem], **kwargs) -> Resposta:
        import httpx

        t0 = time.time()
        r = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": kwargs.get("modelo", self.modelo),
                "messages": [m.como_dict() for m in mensagens],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperatura", self.temperatura),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            },
            timeout=kwargs.get("timeout", self.timeout),
        )
        r.raise_for_status()
        dados = r.json()
        return Resposta(
            texto=dados.get("message", {}).get("content", ""),
            provedor=self.provedor,
            modelo=dados.get("model", self.modelo),
            tokens_entrada=dados.get("prompt_eval_count", 0) or 0,
            tokens_saida=dados.get("eval_count", 0) or 0,
            segundos=round(time.time() - t0, 2),
        )

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

        try:
            r = self.gerar([Mensagem("user", "Responda apenas: ok")], max_tokens=10)
            return True, f"{r.modelo} respondeu em {r.segundos}s: {r.texto.strip()[:40]}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Anthropic — opcional
# --------------------------------------------------------------------------


class ClienteAnthropic:
    """API da Anthropic. Exige `ANTHROPIC_API_KEY` no `.env`.

    Aqui so para provar que o contrato aguenta um terceiro provedor sem
    remendo. O `system` e um parametro separado nessa API, e nao uma mensagem —
    a conversao acontece dentro de `gerar`, e quem chama nao percebe.
    """

    provedor = "anthropic"

    MODELOS = ["claude-sonnet-4-5", "claude-haiku-4-5"]

    def __init__(self, modelo: str = "claude-haiku-4-5", temperatura: float = 0.1,
                 max_tokens: int = 800, timeout: float = 60.0):
        self.modelo = modelo
        self.temperatura = temperatura
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._cliente = None

    def _conectar(self):
        if self._cliente is None:
            import anthropic

            chave = os.environ.get("ANTHROPIC_API_KEY")
            if not chave:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY nao encontrada no `.env`."
                )
            self._cliente = anthropic.Anthropic(api_key=chave, timeout=self.timeout)
        return self._cliente

    def modelos(self) -> list[str]:
        return list(self.MODELOS)

    def gerar(self, mensagens: list[Mensagem], **kwargs) -> Resposta:
        cliente = self._conectar()
        sistema = "\n\n".join(m.conteudo for m in mensagens if m.papel == "system")
        conversa = [m.como_dict() for m in mensagens if m.papel != "system"]

        t0 = time.time()
        r = cliente.messages.create(
            model=kwargs.get("modelo", self.modelo),
            system=sistema or "",
            messages=conversa,
            temperature=kwargs.get("temperatura", self.temperatura),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return Resposta(
            texto="".join(b.text for b in r.content if b.type == "text"),
            provedor=self.provedor,
            modelo=r.model,
            tokens_entrada=r.usage.input_tokens,
            tokens_saida=r.usage.output_tokens,
            segundos=round(time.time() - t0, 2),
        )

    def testar(self) -> tuple[bool, str]:
        try:
            r = self.gerar([Mensagem("user", "Responda apenas: ok")], max_tokens=10)
            return True, f"{r.modelo} respondeu em {r.segundos}s: {r.texto.strip()[:40]}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"


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

    Nao chama a API — so verifica se o pacote esta instalado e se a chave existe.
    A tela usa isso para nao oferecer o que vai falhar.
    """
    estado = {}

    try:
        import openai  # noqa: F401
        tem_pacote = True
    except ImportError:
        tem_pacote = False
    tem_chave = bool(os.environ.get("OPENAI_API_KEY"))
    estado["openai"] = {
        "pronto": tem_pacote and tem_chave,
        "motivo": "" if tem_pacote and tem_chave
        else ("pacote `openai` nao instalado" if not tem_pacote
              else "OPENAI_API_KEY ausente no .env"),
    }

    try:
        import httpx

        httpx.get(
            (os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
            + "/api/tags",
            timeout=2.0,
        ).raise_for_status()
        estado["ollama"] = {"pronto": True, "motivo": ""}
    except Exception:  # noqa: BLE001
        estado["ollama"] = {"pronto": False, "motivo": "servico Ollama nao respondeu"}

    try:
        import anthropic  # noqa: F401
        tem_pacote = True
    except ImportError:
        tem_pacote = False
    tem_chave = bool(os.environ.get("ANTHROPIC_API_KEY"))
    estado["anthropic"] = {
        "pronto": tem_pacote and tem_chave,
        "motivo": "" if tem_pacote and tem_chave
        else ("pacote `anthropic` nao instalado" if not tem_pacote
              else "ANTHROPIC_API_KEY ausente no .env"),
    }

    return estado
