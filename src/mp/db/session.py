"""Conexao com o banco.

SQLite e um **arquivo**, nao um servico. Nao ha servidor para subir, senha para
configurar nem container para orquestrar — o projeto roda a partir de um clone
limpo.

O esquema esta escrito em SQLAlchemy, que fica acima do banco. Migrar para
PostgreSQL depois e trocar a string de conexao; nenhuma consulta precisa ser
reescrita.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from mp import config
from mp.db.models import Base


def url_do_banco(caminho: str | Path | None = None) -> str:
    """String de conexao. Trocar esta linha e o que migra para PostgreSQL."""
    caminho = Path(caminho) if caminho else config.DB_PATH
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{caminho}"


def criar_engine(caminho: str | Path | None = None, echo: bool = False):
    motor = create_engine(url_do_banco(caminho), echo=echo, future=True)

    @event.listens_for(motor, "connect")
    def _ajustes_sqlite(conexao, _registro):
        cursor = conexao.cursor()
        # O SQLite ignora chave estrangeira por padrao. Sem isto, um chunk
        # poderia apontar para um documento que nao existe.
        cursor.execute("PRAGMA foreign_keys=ON")
        # WAL deixa leitura e escrita conviverem — a UI consulta enquanto a
        # ingestao escreve.
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return motor


engine = criar_engine()
_Sessao = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@contextmanager
def sessao(motor=None):
    """Sessao com commit no fim e rollback se algo falhar.

    Uso:

        with sessao() as s:
            s.add(objeto)
    """
    fabrica = sessionmaker(bind=motor, expire_on_commit=False) if motor else _Sessao
    s = fabrica()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def criar_esquema(motor=None, apagar_antes: bool = False) -> None:
    """Cria as tabelas. `apagar_antes=True` recria o banco do zero.

    A ingestao usa `apagar_antes=True` para ser repetivel: rodar duas vezes
    produz exatamente o mesmo banco, sem duplicar nada.
    """
    motor = motor or engine
    if apagar_antes:
        Base.metadata.drop_all(motor)
    Base.metadata.create_all(motor)
