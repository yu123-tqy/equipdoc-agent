from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer(path: Path) -> SqliteSaver:
    """Create a process-wide durable checkpointer suitable for the Gradio app."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(path),
        check_same_thread=False,
        timeout=30,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


def create_postgres_checkpointer(database_url: str):
    """Create a pooled production checkpointer without leaking connection lifetime."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL checkpoints require the 'postgres' project extra."
        ) from exc
    pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=8,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=True,
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    checkpointer.connection_pool = pool
    return checkpointer


def create_checkpointer(database: Path | str) -> Any:
    raw = str(database)
    if raw.startswith(("postgresql://", "postgresql+psycopg://")):
        return create_postgres_checkpointer(raw.replace("postgresql+psycopg://", "postgresql://", 1))
    if raw.startswith("sqlite:///"):
        raw = raw.removeprefix("sqlite:///")
    return create_sqlite_checkpointer(Path(raw))


def delete_checkpoint_threads(checkpointer: Any, thread_ids: list[str]) -> int:
    deleted = 0
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if not callable(delete_thread):
        return deleted
    for thread_id in dict.fromkeys(thread_ids):
        delete_thread(thread_id)
        deleted += 1
    return deleted
