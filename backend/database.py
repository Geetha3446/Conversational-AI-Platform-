"""
SQLite + SQLAlchemy plumbing.

Two important details for this project:

1. `check_same_thread=False` - FastAPI serves sync endpoints from a thread pool,
   so the same connection can legitimately be touched by different threads.

2. WAL journal mode + a generous busy timeout - the LangGraph checkpointer writes
   to the *same* SQLite file as the ORM. WAL lets readers and a writer coexist,
   and the timeout makes concurrent writers wait politely instead of raising
   "database is locked".
"""

from __future__ import annotations

import sqlite3
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.sqlalchemy_url,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record):
    """Run once per new physical connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")     # concurrent read + write
    cursor.execute("PRAGMA synchronous=NORMAL;")   # good durability / speed trade-off
    cursor.execute("PRAGMA foreign_keys=ON;")      # enforce FK constraints
    cursor.execute("PRAGMA busy_timeout=30000;")   # wait up to 30s for a lock
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class every ORM model inherits from."""


# ---------------------------------------------------------------------------
# Dependency used by FastAPI routes
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """Yield a session and guarantee it is closed, even if the route raises."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Safe to call on every startup (it is a no-op if they exist)."""
    from backend import models  # noqa: F401  (import registers the models on Base)

    Base.metadata.create_all(bind=engine)


def raw_connection() -> sqlite3.Connection:
    """
    A plain sqlite3 connection to the same file, for LangGraph's SqliteSaver
    which expects a DB-API connection rather than a SQLAlchemy engine.
    """
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn
