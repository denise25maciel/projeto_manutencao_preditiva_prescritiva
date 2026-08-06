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

    # Turno que o SISTEMA fez a si mesmo ao fixar o manual, para ja abrir a
    # conversa com problema, sintomas e correcao. A tela desenha sem o balao do
    # usuario — a pergunta nao foi do tecnico. Fora isso e um turno igual aos
    # outros: passou por G4, redacao e G5, e entra no historico verificado.
    abertura: bool = False

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
    # So preenchida quando o tipo foi **apurado**: pelo kNN, ou por o manual
    # travado cobrir uma familia so. Manual que cobre varias deixa isto `None` —
    # o Doc1 atende quatro tipos de rolamento e a busca por texto nao separa
    # entre eles. Quem quer nomear o assunto usa `familias`.
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

    # --- escolha: quando a evidencia nao aponta um manual so ---------------
    # Cada sintoma e guardado SEPARADO, nao concatenado numa frase. A busca
    # codifica um por um e fica com o maior score por trecho; juntar tudo num
    # texto so tiraria a media dos sintomas e diluiria justamente o que
    # discrimina. Ver `rag.buscar_por_sintomas`.
    sintomas: list[str] = field(default_factory=list)
    candidatos: list[tuple[str, float]] = field(default_factory=list)
    # `Doc2` nao diz nada a quem esta na maquina; `desalinhamento` diz. O mapa
    # e montado no grafo, que e quem tem acesso ao catalogo — o estado nao
    # consulta banco nem YAML, so guarda o que ja foi resolvido.
    nomes_candidatos: dict[str, str] = field(default_factory=dict)
    # A evidencia nao separou os candidatos: a lista vai para a tela e **o
    # tecnico** decide entre seguir com um ou detalhar mais. Nao ha teto de
    # tentativas — quem decide quando parar de detalhar e ele, nao um contador.
    aguardando_escolha: bool = False

    @property
    def situacao(self) -> str:
        """`escolhendo` | `aberta` | `encerrada` — o que a tela deve mostrar."""
        if self.aberta:
            return "aberta"
        if self.aguardando_escolha:
            return "escolhendo"
        return "encerrada"

    @property
    def melhor_trecho_por_documento(self) -> dict[str, object]:
        """O trecho de maior score de cada documento candidato.

        E a resposta a *"por que este manual apareceu na lista?"*. Mostrar o
        trecho ao lado do nome e o que torna a escolha do tecnico informada em
        vez de um chute entre seis codigos: ele le o pedaco do manual que se
        pareceu com o que descreveu e reconhece — ou nao — a propria maquina.

        Sai dos trechos que a busca **ja** devolveu; nao ha consulta nova.
        """
        melhor: dict[str, object] = {}
        for t in self.trechos_de_abertura:
            atual = melhor.get(t.documento_id)
            if atual is None or t.score > atual.score:
                melhor[t.documento_id] = t
        return melhor

    @property
    def aviso_de_pouca_informacao(self) -> str:
        """Em portugues claro: por que a escolha esta voltando para o tecnico.

        O `motivo` diz "margem de 14%, minimo 25%" — verdade, e ilegivel para
        quem esta com a maquina parada. O numero desce para a legenda.

        E **codigo, nao modelo**: dependendo do LLM, bastaria ele cair ou
        redigir mal para o sistema nao avisar nada, e o tecnico ficaria
        esperando uma resposta que nunca vem.
        """
        # Os sintomas entram na propria frase, entre colchetes. O tecnico ve o
        # que o sistema realmente registrou — se ele escreveu tres coisas e so
        # duas aparecem, o erro fica visivel na hora, em vez de virar uma busca
        # silenciosamente errada.
        ditos = ", ".join(s.strip() for s in self.sintomas if s.strip())
        descrito = f"O que voce descreveu [{ditos}]" if ditos else "O que voce descreveu"

        n = len(self.candidatos)
        if n <= 1:
            return (
                f"{descrito} ainda nao aponta um procedimento com folga. "
                "Siga com o candidato abaixo se ele for o certo, ou **conte "
                "mais** sobre o que esta acontecendo."
            )

        # Os NOMES dos candidatos, nao os sintomas deles. A diferenca e
        # deliberada: mostrar "desalinhamento, polia, correia" torna a duvida
        # visivel; mostrar o que cada um descreve faria o tecnico repetir de
        # volta o que acabou de ler, e a evidencia viraria eco em vez de
        # observacao.
        quais = ", ".join(
            self.nomes_candidatos.get(doc, doc) for doc, _ in self.candidatos
        )
        return (
            f"{descrito} combina com **{n} procedimentos** ({quais}) e a "
            "evidencia nao separa um deles. **Escolha** o que descreve a sua "
            "maquina, ou **conte mais** para eu refazer a busca."
        )

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

    @property
    def familias(self) -> list[str]:
        """Todas as familias que o manual desta conversa cobre.

        Pelo sensor, a familia foi apurada e a lista tem uma so. Por texto, ela
        vem do documento travado e pode ter varias. E a lista, nao um item dela,
        o que se pode afirmar dessa sessao.
        """
        if self.familias_do_documento:
            return list(self.familias_do_documento)
        return [self.familia] if self.familia else []

    @property
    def assunto(self) -> str:
        """O nome do assunto para exibir — `rolamento ×4` quando o tipo nao foi apurado.

        Le o nome que o grafo ja resolveu em `nomes_candidatos`; a regra que o
        monta mora la, junto do catalogo, e nao e reescrita aqui.
        """
        if self.familia:
            return self.familia
        for documento in self.documentos:
            if nome := self.nomes_candidatos.get(documento):
                return nome
        return self.manual

    def historico_para_prompt(self, maximo: int = 4) -> list[tuple[str, str]]:
        """Os ultimos turnos como `(pergunta, texto_verificado)`.

        So entram turnos que produziram conteudo — recusa nao vira contexto.
        """
        uteis = [t for t in self.turnos if not t.recusado]
        return [(t.pergunta, t.texto_para_historico) for t in uteis[-maximo:]]

    @property
    def n_verificados(self) -> int:
        return sum(1 for t in self.turnos if t.verificada and not t.degradou)
