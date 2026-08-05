"""Esquema do banco: quatro tabelas.

O caminho de uma pergunta atravessa todas elas:

    leitura -> evento -> rotulo -> familia -> documento -> trecho

Cada seta e uma consulta exata. Nenhuma e semelhanca.

A ligacao **familia -> documento** e a unica que nao mora aqui: vive no
`data/fault_map.yaml`, versionado no Git. E decisao curada, e no Git cada mudanca
tem autor, data e motivo; dentro do banco viraria um UPDATE sem rastro.

Por que quatro tabelas e nao uma
--------------------------------
Numa tabela so, cada uma das 166.796 leituras carregaria os dados do seu evento
(repetidos centenas de vezes) e o texto inteiro do procedimento (repetido dezenas
de milhares de vezes). Corrigir o titulo de um manual exigiria corrigir 27 mil
linhas.

Tipos
-----
`Float` em vez de `Numeric`: os valores vem do sensor com 4 casas e nao ha conta
financeira aqui. `LargeBinary` para o embedding — no SQLite vira BLOB, e a
comparacao por cosseno roda em memoria com numpy (sao poucas centenas de trechos).

Fuso horario
------------
`DateTime(timezone=True)` e o que o esquema pede, mas **o SQLite nao armazena
fuso** — ele guarda o texto da data e devolve um datetime "ingenuo", sem `+00:00`.
O instante esta correto; so a etiqueta se perde.

Consequencia pratica: comparar uma data lida do banco com uma data com fuso
levanta `TypeError`. Todo `created_at` aqui e UTC por construcao (o CSV vem com
`+00:00` e o loader converte). Ao comparar com valores externos, usar
`.replace(tzinfo=timezone.utc)` no que veio do banco.

Isso desaparece sozinho na migracao para PostgreSQL, que guarda o fuso de verdade.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# readings
# --------------------------------------------------------------------------


class Leitura(Base):
    """Uma medicao do sensor. E o dado bruto, sem transformacao.

    `evento_a` e `evento_b` sao as duas versoes de agrupamento, guardadas lado a
    lado de proposito: a mesma leitura pode pertencer a eventos diferentes
    conforme a ordem das operacoes, e o projeto trata isso como experimento
    documentado, nao como detalhe interno.
    """

    __tablename__ = "readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fault: Mapped[str] = mapped_column(String(64), index=True)
    rpm: Mapped[float] = mapped_column(Float)

    # A qual evento esta leitura pertence, em cada versao.
    evento_a: Mapped[int | None] = mapped_column(Integer, index=True)
    evento_b: Mapped[int | None] = mapped_column(Integer, index=True)

    # --- medidas de vibracao, eixo z ---
    z_rms_velocity_in_s: Mapped[float] = mapped_column(Float)
    z_rms_velocity_mm_s: Mapped[float] = mapped_column(Float)
    z_peak_velocity_in_s: Mapped[float] = mapped_column(Float)
    z_peak_velocity_mm_s: Mapped[float] = mapped_column(Float)
    z_rms_acceleration_g: Mapped[float] = mapped_column(Float)
    z_peak_acceleration_g: Mapped[float] = mapped_column(Float)
    z_high_freq_rms_accel_g: Mapped[float] = mapped_column(Float)
    z_peak_vel_comp_freq_hz: Mapped[float] = mapped_column(Float)
    z_kurtosis: Mapped[float] = mapped_column(Float)
    z_crest_factor: Mapped[float] = mapped_column(Float)

    # --- medidas de vibracao, eixo x ---
    x_rms_velocity_in_s: Mapped[float] = mapped_column(Float)
    x_rms_velocity_mm_s: Mapped[float] = mapped_column(Float)
    x_peak_velocity_in_s: Mapped[float] = mapped_column(Float)
    x_peak_velocity_mm_s: Mapped[float] = mapped_column(Float)
    x_rms_acceleration_g: Mapped[float] = mapped_column(Float)
    x_peak_acceleration_g: Mapped[float] = mapped_column(Float)
    x_high_freq_rms_accel_g: Mapped[float] = mapped_column(Float)
    x_peak_vel_comp_freq_hz: Mapped[float] = mapped_column(Float)
    x_kurtosis: Mapped[float] = mapped_column(Float)
    x_crest_factor: Mapped[float] = mapped_column(Float)

    # --- temperatura ---
    temperature_c: Mapped[float] = mapped_column(Float)
    temperature_f: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        # As consultas mais frequentes filtram por rotulo e ordenam por tempo.
        Index("ix_readings_fault_tempo", "fault", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Leitura {self.id} {self.fault} {self.created_at:%d/%m %H:%M:%S}>"


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------


class Episodio(Base):
    """Uma vez em que a maquina foi medida com o mesmo defeito.

    A chave e o par **(versao, numero)**: o evento 2 da versao A e o evento 2 da
    versao B sao coisas diferentes.

    Guardar as duas numa tabela so — em vez de duas tabelas — deixa a comparacao
    ser um `GROUP BY versao` em vez de um `UNION` em toda consulta.
    """

    __tablename__ = "episodes"

    versao: Mapped[str] = mapped_column(String(1), primary_key=True)
    numero: Mapped[int] = mapped_column(Integer, primary_key=True)

    fault: Mapped[str] = mapped_column(String(64), index=True)
    # Resolvida a partir do fault_map.yaml no momento da ingestao. Fica aqui
    # copiada para a consulta ser direta; a fonte da verdade continua no arquivo.
    familia: Mapped[str | None] = mapped_column(String(64), index=True)

    # A rotacao faz parte da identidade do evento: ela encerra um e comeca outro
    # (`config.COLUNAS_QUEBRA_EVENTO`), entao e constante aqui por construcao.
    # Sem ela a tabela nao diria em que regime o ensaio foi feito — e o mesmo
    # defeito a 500 e a 2000 rpm tem assinaturas muito diferentes.
    rpm: Mapped[float | None] = mapped_column(Float, index=True)

    n_leituras: Mapped[int] = mapped_column(Integer)
    inicio: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fim: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duracao_s: Mapped[float] = mapped_column(Float)

    # O quanto as leituras de dentro do evento se parecem entre si. Menor e melhor.
    dispersao: Mapped[float | None] = mapped_column(Float)
    # Maior interrupcao sem leitura dentro do evento. Denuncia agrupamento largo.
    maior_buraco_s: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_episodes_versao_familia", "versao", "familia"),
    )

    def __repr__(self) -> str:
        return f"<Episodio {self.versao}{self.numero} {self.fault} n={self.n_leituras}>"


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------


class Documento(Base):
    """Um procedimento de manutencao.

    A tabela descreve o arquivo. **Qual familia ele atende nao esta aqui** — essa
    ligacao e muitos-para-muitos (o manual de rolamentos cobre quatro familias) e
    e decisao curada, entao vive no `fault_map.yaml`.

    O guardrail G3 pergunta ao catalogo, nao a esta tabela.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    titulo: Mapped[str] = mapped_column(Text)
    arquivo_md: Mapped[str] = mapped_column(String(255))
    pdf_origem: Mapped[str | None] = mapped_column(String(255))

    # `pdf` (camada de texto do arquivo) ou `sidecar` (transcricao manual de um
    # PDF digitalizado). Registrado para ninguem confundir os dois.
    origem_texto: Mapped[str] = mapped_column(String(16))
    n_secoes: Mapped[int] = mapped_column(Integer)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="documento", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Documento {self.id} {self.titulo[:40]}>"


# --------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------


class Chunk(Base):
    """Uma secao numerada de um procedimento.

    A unidade de recuperacao do RAG. O corte e por secao — e nao por numero de
    caracteres — porque a resposta precisa citar um endereco verificavel:
    "Doc2, secao 9". Um pedaco cortado no meio de uma frase nao tem endereco.

    `embedding` fica vazio ate a Parte 4. E a representacao numerica do texto,
    guardada como BLOB; a comparacao por cosseno roda em memoria com numpy.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    documento_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    numero: Mapped[str] = mapped_column(String(16))
    titulo: Mapped[str] = mapped_column(Text)
    nivel: Mapped[int] = mapped_column(Integer)

    # `sintomas`, `diagnostico_vibracao`, `correcao`, `validacao`, `seguranca`...
    # Vazio quando o titulo da secao nao casa com nenhum campo conhecido.
    campo: Mapped[str | None] = mapped_column(String(32), index=True)

    texto: Mapped[str] = mapped_column(Text)
    n_caracteres: Mapped[int] = mapped_column(Integer)

    # Pagina do PDF de origem. Nula quando o texto veio de transcricao manual
    # (Doc1 e digitalizado e nao tem camada de texto). A pagina e METADADO,
    # lido do banco — nunca escrito pelo modelo de linguagem, que so citaria de
    # memoria e acertaria por acaso.
    pagina_inicio: Mapped[int | None] = mapped_column(Integer)
    pagina_fim: Mapped[int | None] = mapped_column(Integer)

    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    embedding_modelo: Mapped[str | None] = mapped_column(String(64))

    documento: Mapped["Documento"] = relationship(back_populates="chunks")

    __table_args__ = (
        # A busca da Parte 4 filtra por documento e tipo antes de comparar vetores.
        Index("ix_chunks_documento_campo", "documento_id", "campo"),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.documento_id} sec {self.numero} [{self.campo}]>"
