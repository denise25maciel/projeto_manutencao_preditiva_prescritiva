"""O estado de uma conversa.

Uma sessao nao e uma sequencia de perguntas soltas: tem um defeito fixado, um
manual autorizado e um historico. Este modulo guarda essas tres coisas.

**Duas regras moram aqui, e as duas so existem por causa do multi-turno.**

1. **O manual e fixado no turno 1 e nao muda.** Quem decide qual manual vale e o
   catalogo, uma vez, no inicio. Turno nenhum pode trocar de documento no meio da
   conversa — senao a pergunta 3 seria respondida com o manual de outro defeito.

2. **O historico guarda so o que foi verificado.** Quando a resposta nao passa na
   ancoragem, o que entra no historico e o trecho do manual, nao a prosa. Sem
   isso, um deslize no turno 2 vira contexto no turno 3 e o modelo o trata como
   fato ja estabelecido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Turno:
    """Uma volta da conversa: o que foi perguntado e o que ficou registrado."""

    pergunta: str
    resposta: str = ""

    # Os trechos que sustentaram esta resposta.
    trechos: list = field(default_factory=list)
    vereditos: list = field(default_factory=list)

    # Passou no G5? Determina o que vai para o historico.
    verificada: bool = False
    degradou: bool = False
    recusado: bool = False
    motivo: str = ""

    usou_llm: bool = False
    provedor: str | None = None
    modelo: str | None = None
    tentativas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    segundos: float = 0.0
    prompt: str = ""

    quando: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def texto_para_historico(self) -> str:
        """O que os proximos turnos verao.

        Resposta verificada entra como esta. Resposta reprovada **nao entra** —
        no lugar dela vai o texto do manual, que e o que se pode garantir.
        """
        if self.verificada and not self.degradou:
            return self.resposta
        if self.trechos:
            return "\n\n".join(
                f"[{t.referencia}] {t.texto.strip()}" for t in self.trechos[:2]
            )
        return self.motivo or "(sem conteudo verificado neste turno)"

    @property
    def referencias(self) -> list[str]:
        """Documento, secao e pagina de cada trecho usado."""
        return [t.referencia for t in self.trechos]


@dataclass
class Sessao:
    """Uma conversa sobre um defeito. O defeito nao muda; as perguntas mudam."""

    rotulo: str
    familia: str | None = None
    documentos: list[str] = field(default_factory=list)

    # O que a similaridade viu. `None` quando a sessao foi aberta por rotulo,
    # sem passar pelo kNN.
    diagnostico: object | None = None

    # --- quando a sessao nasce de uma descricao escrita --------------------
    # Nesse caminho nao ha evento de sensor: o texto do tecnico e a entrada, e o
    # documento aparece antes da familia.
    descricao: str = ""
    familias_do_documento: list[str] = field(default_factory=list)
    peso_documento: float = 0.0
    trechos_de_abertura: list = field(default_factory=list)

    # --- investigacao: quando a evidencia nao aponta um manual so ----------
    # Cada sintoma e guardado SEPARADO, nao concatenado numa frase. A busca
    # codifica um por um e fica com o maior score por trecho; juntar tudo num
    # texto so tiraria a media dos sintomas e diluiria justamente o que
    # discrimina. Ver `rag.buscar_por_sintomas`.
    sintomas: list[str] = field(default_factory=list)
    investigando: bool = False
    rodadas: int = 0
    candidatos: list[tuple[str, float]] = field(default_factory=list)
    pergunta_investigacao: str = ""
    # O historico das perguntas feitas ao tecnico, uma por rodada. A conversa
    # precisa poder ser relida inteira: sem isto so a ultima pergunta sobrevive
    # e a tela mostraria respostas soltas, sem o que as motivou.
    perguntas_investigacao: list[str] = field(default_factory=list)
    # Preenchido quando o teto de rodadas se esgota: a escolha passa a ser do
    # tecnico, e a interface precisa saber disso para oferecer a lista.
    aguardando_escolha: bool = False

    @property
    def situacao(self) -> str:
        """`investigando` | `aberta` | `encerrada` — o que a tela deve mostrar."""
        if self.aberta:
            return "aberta"
        if self.investigando or self.aguardando_escolha:
            return "investigando"
        return "encerrada"

    @property
    def descricao_completa(self) -> str:
        """Tudo que o tecnico contou ate agora, para exibir (nao para embedar)."""
        return " ".join(self.sintomas) if self.sintomas else self.descricao

    @property
    def origem(self) -> str:
        """Como esta sessao comecou — muda o que se pode garantir dela."""
        if self.diagnostico is not None:
            return "sensor"
        if self.descricao:
            return "texto"
        return "rotulo"

    # Fechada = o catalogo recusou no turno 1. Nenhuma pergunta sera respondida.
    aberta: bool = False
    motivo: str = ""
    vereditos_abertura: list = field(default_factory=list)

    turnos: list[Turno] = field(default_factory=list)
    criada_em: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def manual(self) -> str:
        return ", ".join(self.documentos) if self.documentos else "(nenhum)"

    def historico_para_prompt(self, maximo: int = 4) -> list[tuple[str, str]]:
        """Os ultimos turnos como `(pergunta, texto_verificado)`.

        So entram turnos que produziram conteudo — recusa nao vira contexto.
        """
        uteis = [t for t in self.turnos if not t.recusado]
        return [(t.pergunta, t.texto_para_historico) for t in uteis[-maximo:]]

    @property
    def n_verificados(self) -> int:
        return sum(1 for t in self.turnos if t.verificada and not t.degradou)
