"""Banco de dados — SQLite via SQLAlchemy.

- `models`  — o esquema: quatro tabelas
- `session` — conexao e criacao do arquivo
- `ingest`  — popula o banco a partir do CSV bruto e dos manuais
"""

from mp.db.models import Base, Chunk, Documento, Episodio, Leitura
from mp.db.session import criar_esquema, engine, sessao

__all__ = [
    "Base",
    "Leitura",
    "Episodio",
    "Documento",
    "Chunk",
    "engine",
    "sessao",
    "criar_esquema",
]
