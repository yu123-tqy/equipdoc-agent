from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool


ACTIVE_RUN_STATUSES = ("running", "awaiting_review")
SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _title_from_message(message: str) -> str:
    compact = " ".join(str(message).split())
    return compact[:32] or "新对话"


def _normalize_owner_id(owner_id: str) -> str:
    value = " ".join(str(owner_id).split())
    if not value or len(value) > 128:
        raise ValueError("owner_id must contain 1 to 128 characters")
    return value


def _database_url(database: Path | str) -> tuple[str, Path | None]:
    raw = str(database)
    if "://" in raw:
        return raw, None
    path = Path(database).expanduser().resolve()
    return f"sqlite:///{path.as_posix()}", path


class ConversationStore:
    """Durable, owner-scoped conversation storage for SQLite or PostgreSQL."""

    def __init__(
        self,
        database: Path | str,
        *,
        owner_id: str = "local",
        recent_context_messages: int = 8,
        summary_max_chars: int = 2400,
        compaction_turns: int = 12,
        _engine: Engine | None = None,
    ) -> None:
        self.database_url, self.path = _database_url(database)
        self.owner_id = _normalize_owner_id(owner_id)
        self.recent_context_messages = max(2, int(recent_context_messages))
        self.summary_max_chars = max(400, int(summary_max_chars))
        self.compaction_turns = max(2, int(compaction_turns))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = _engine or self._create_engine(self.database_url)
        if _engine is None:
            self._initialize()

    @staticmethod
    def _create_engine(database_url: str) -> Engine:
        is_sqlite = database_url.startswith("sqlite:")
        engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args={"timeout": 30, "check_same_thread": False} if is_sqlite else {},
            poolclass=NullPool if is_sqlite else None,
        )
        if is_sqlite:
            @event.listens_for(engine, "connect")
            def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()
        return engine

    def for_owner(self, owner_id: str) -> "ConversationStore":
        """Return a lightweight owner-scoped view sharing the same connection pool."""
        return ConversationStore(
            self.database_url,
            owner_id=owner_id,
            recent_context_messages=self.recent_context_messages,
            summary_max_chars=self.summary_max_chars,
            compaction_turns=self.compaction_turns,
            _engine=self.engine,
        )

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        with self.engine.begin() as connection:
            yield connection

    def _initialize(self) -> None:
        if self.database_url.startswith("sqlite:"):
            with self.engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        statements = (
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                agent_thread_id TEXT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                summary TEXT NOT NULL DEFAULT '',
                next_sequence_no INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT,
                deleted_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                sequence_no INTEGER NOT NULL,
                role TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'text',
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                status TEXT NOT NULL,
                pending_review_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_threads (
                conversation_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (conversation_id, thread_id),
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """,
        )
        with self._connection() as connection:
            for statement in statements:
                connection.execute(text(statement))
        self._migrate_existing_schema()
        index_statements = (
            "CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated "
            "ON conversations(owner_id, archived_at, deleted_at, updated_at DESC)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_sequence "
            "ON messages(conversation_id, sequence_no)",
            "CREATE INDEX IF NOT EXISTS idx_runs_conversation_status "
            "ON runs(conversation_id, status, started_at DESC)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_one_active "
            "ON runs(conversation_id) WHERE status IN ('running', 'awaiting_review')",
        )
        with self._connection() as connection:
            for statement in index_statements:
                connection.execute(text(statement))
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, applied_at) "
                    "SELECT :version, :now WHERE NOT EXISTS "
                    "(SELECT 1 FROM schema_migrations WHERE version = :version)"
                ),
                {"version": SCHEMA_VERSION, "now": _utc_now()},
            )

    def _migrate_existing_schema(self) -> None:
        columns = {item["name"] for item in inspect(self.engine).get_columns("conversations")}
        additions = {
            "agent_thread_id": "TEXT",
            "summary": "TEXT NOT NULL DEFAULT ''",
            "next_sequence_no": "INTEGER NOT NULL DEFAULT 1",
            "deleted_at": "TEXT",
        }
        with self._connection() as connection:
            for column, definition in additions.items():
                if column not in columns:
                    connection.execute(
                        text(f"ALTER TABLE conversations ADD COLUMN {column} {definition}")
                    )
            connection.execute(
                text("UPDATE conversations SET agent_thread_id = id WHERE agent_thread_id IS NULL")
            )
            connection.execute(
                text(
                    """
                    UPDATE conversations SET next_sequence_no = (
                        SELECT COALESCE(MAX(messages.sequence_no), 0) + 1
                        FROM messages WHERE messages.conversation_id = conversations.id
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO agent_threads(conversation_id, thread_id, created_at)
                    SELECT id, agent_thread_id, created_at FROM conversations
                    WHERE agent_thread_id IS NOT NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM agent_threads
                        WHERE agent_threads.conversation_id = conversations.id
                          AND agent_threads.thread_id = conversations.agent_thread_id
                      )
                    """
                )
            )

    def schema_version(self) -> int:
        with self._connection() as connection:
            value = connection.execute(text("SELECT MAX(version) FROM schema_migrations")).scalar()
        return int(value or 0)

    def create_conversation(
        self,
        *,
        title: str = "新对话",
        conversation_id: str | None = None,
    ) -> str:
        conversation_id = conversation_id or str(uuid4())
        now = _utc_now()
        with self._connection() as connection:
            existing = connection.execute(
                text("SELECT owner_id FROM conversations WHERE id = :id"),
                {"id": conversation_id},
            ).mappings().first()
            if existing:
                if existing["owner_id"] != self.owner_id:
                    raise PermissionError("Conversation belongs to another owner")
                return conversation_id
            connection.execute(
                text(
                    """
                    INSERT INTO conversations
                        (id, owner_id, agent_thread_id, title, status, created_at, updated_at)
                    VALUES (:id, :owner_id, :thread_id, :title, 'active', :now, :now)
                    """
                ),
                {
                    "id": conversation_id,
                    "owner_id": self.owner_id,
                    "thread_id": conversation_id,
                    "title": _title_from_message(title),
                    "now": now,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO agent_threads(conversation_id, thread_id, created_at) "
                    "VALUES (:conversation_id, :thread_id, :now)"
                ),
                {"conversation_id": conversation_id, "thread_id": conversation_id, "now": now},
            )
        return conversation_id

    def ensure_conversation(self, conversation_id: str) -> None:
        self.create_conversation(conversation_id=conversation_id)

    def get_agent_thread_id(self, conversation_id: str) -> str:
        with self._connection() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT agent_thread_id FROM conversations
                    WHERE id = :id AND owner_id = :owner_id
                      AND archived_at IS NULL AND deleted_at IS NULL
                    """
                ),
                {"id": conversation_id, "owner_id": self.owner_id},
            ).mappings().first()
        if not row:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        return str(row["agent_thread_id"] or conversation_id)

    def list_conversations(
        self,
        *,
        search: str = "",
        limit: int | None = None,
        offset: int = 0,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"owner_id": self.owner_id, "offset": max(0, offset)}
        filters = ["c.owner_id = :owner_id", "c.deleted_at IS NULL"]
        if not include_archived:
            filters.append("c.archived_at IS NULL")
        clean_search = " ".join(str(search).split())
        if clean_search:
            params["pattern"] = f"%{clean_search.lower()}%"
            filters.append(
                "(LOWER(c.title) LIKE :pattern OR EXISTS ("
                "SELECT 1 FROM messages m WHERE m.conversation_id = c.id "
                "AND LOWER(m.content) LIKE :pattern))"
            )
        limit_clause = ""
        if limit is not None:
            params["limit"] = max(1, min(100, int(limit)))
            limit_clause = " LIMIT :limit OFFSET :offset"
        query = (
            "SELECT c.id, c.title, c.status, c.created_at, c.updated_at, c.archived_at, "
            "c.summary FROM conversations c WHERE "
            + " AND ".join(filters)
            + " ORDER BY c.updated_at DESC"
            + limit_clause
        )
        with self._connection() as connection:
            rows = connection.execute(text(query), params).mappings().all()
        return [dict(row) for row in rows]

    def count_conversations(self, *, search: str = "", include_archived: bool = False) -> int:
        params: dict[str, Any] = {"owner_id": self.owner_id}
        filters = ["c.owner_id = :owner_id", "c.deleted_at IS NULL"]
        if not include_archived:
            filters.append("c.archived_at IS NULL")
        clean_search = " ".join(str(search).split())
        if clean_search:
            params["pattern"] = f"%{clean_search.lower()}%"
            filters.append(
                "(LOWER(c.title) LIKE :pattern OR EXISTS ("
                "SELECT 1 FROM messages m WHERE m.conversation_id = c.id "
                "AND LOWER(m.content) LIKE :pattern))"
            )
        with self._connection() as connection:
            value = connection.execute(
                text("SELECT COUNT(*) FROM conversations c WHERE " + " AND ".join(filters)),
                params,
            ).scalar_one()
        return int(value)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT id, title, status, summary, created_at, updated_at,
                           archived_at, deleted_at
                    FROM conversations
                    WHERE id = :id AND owner_id = :owner_id AND deleted_at IS NULL
                    """
                ),
                {"id": conversation_id, "owner_id": self.owner_id},
            ).mappings().first()
        return dict(row) if row else None

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE conversations SET title = :title, updated_at = :now
                    WHERE id = :id AND owner_id = :owner_id
                      AND archived_at IS NULL AND deleted_at IS NULL
                    """
                ),
                {
                    "title": _title_from_message(title),
                    "now": _utc_now(),
                    "id": conversation_id,
                    "owner_id": self.owner_id,
                },
            )
        return result.rowcount == 1

    def archive_conversation(self, conversation_id: str) -> bool:
        now = _utc_now()
        with self._connection() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE conversations
                    SET status = 'archived', archived_at = :now, updated_at = :now
                    WHERE id = :id AND owner_id = :owner_id
                      AND archived_at IS NULL AND deleted_at IS NULL
                    """
                ),
                {"now": now, "id": conversation_id, "owner_id": self.owner_id},
            )
        return result.rowcount == 1

    def restore_conversation(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE conversations
                    SET status = 'active', archived_at = NULL, updated_at = :now
                    WHERE id = :id AND owner_id = :owner_id
                      AND archived_at IS NOT NULL AND deleted_at IS NULL
                    """
                ),
                {"now": _utc_now(), "id": conversation_id, "owner_id": self.owner_id},
            )
        return result.rowcount == 1

    def delete_conversation(self, conversation_id: str) -> list[str]:
        """Permanently delete an idle conversation and return checkpoint thread ids."""
        with self._connection() as connection:
            owned = connection.execute(
                text(
                    "SELECT 1 FROM conversations WHERE id = :id AND owner_id = :owner_id"
                ),
                {"id": conversation_id, "owner_id": self.owner_id},
            ).first()
            if not owned:
                return []
            active = connection.execute(
                text(
                    "SELECT 1 FROM runs WHERE conversation_id = :id "
                    "AND status IN ('running', 'awaiting_review')"
                ),
                {"id": conversation_id},
            ).first()
            if active:
                raise RuntimeError("请先完成或停止当前任务再永久删除。")
            thread_ids = [
                str(row[0])
                for row in connection.execute(
                    text("SELECT thread_id FROM agent_threads WHERE conversation_id = :id"),
                    {"id": conversation_id},
                ).all()
            ]
            for table in ("messages", "runs", "agent_threads"):
                connection.execute(
                    text(f"DELETE FROM {table} WHERE conversation_id = :id"),
                    {"id": conversation_id},
                )
            connection.execute(
                text("DELETE FROM conversations WHERE id = :id AND owner_id = :owner_id"),
                {"id": conversation_id, "owner_id": self.owner_id},
            )
        return thread_ids

    def purge_archived(self, retention_days: int) -> dict[str, Any]:
        days = max(1, int(retention_days))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="milliseconds"
        )
        rows = self.list_conversations(include_archived=True)
        candidates = [
            item for item in rows
            if item.get("archived_at") and str(item["archived_at"]) < cutoff
        ]
        deleted_threads: list[str] = []
        for item in candidates:
            deleted_threads.extend(self.delete_conversation(str(item["id"])))
        return {"conversations": len(candidates), "thread_ids": deleted_threads}

    def _next_sequence(self, connection: Connection, conversation_id: str) -> int:
        row = connection.execute(
            text(
                """
                UPDATE conversations
                SET next_sequence_no = next_sequence_no + 1
                WHERE id = :id AND owner_id = :owner_id AND deleted_at IS NULL
                RETURNING next_sequence_no - 1 AS sequence_no
                """
            ),
            {"id": conversation_id, "owner_id": self.owner_id},
        ).mappings().first()
        if not row:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        return int(row["sequence_no"])

    def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        turn_id: str,
        kind: str = "text",
        metadata: dict[str, Any] | None = None,
        connection: Connection | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported conversation role: {role}")
        if connection is None:
            self.ensure_conversation(conversation_id)
            with self._connection() as db:
                return self.add_message(
                    conversation_id,
                    role=role,
                    content=content,
                    turn_id=turn_id,
                    kind=kind,
                    metadata=metadata,
                    connection=db,
                )
        sequence_no = self._next_sequence(connection, conversation_id)
        message_id = str(uuid4())
        connection.execute(
            text(
                """
                INSERT INTO messages
                    (id, conversation_id, turn_id, sequence_no, role, kind, content,
                     metadata_json, created_at)
                VALUES (:id, :conversation_id, :turn_id, :sequence_no, :role, :kind,
                        :content, :metadata_json, :created_at)
                """
            ),
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "sequence_no": sequence_no,
                "role": role,
                "kind": kind,
                "content": str(content),
                "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
                "created_at": _utc_now(),
            },
        )
        return message_id

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT m.id, m.turn_id, m.sequence_no, m.role, m.kind, m.content,
                           m.metadata_json, m.created_at
                    FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.conversation_id = :id AND c.owner_id = :owner_id
                      AND c.deleted_at IS NULL
                    ORDER BY m.sequence_no
                    """
                ),
                {"id": conversation_id, "owner_id": self.owner_id},
            ).mappings().all()
        messages: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except (TypeError, json.JSONDecodeError):
                item["metadata"] = {}
                item.pop("metadata_json", None)
            messages.append(item)
        return messages

    def conversation_context(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id) or {}
        messages = self.list_messages(conversation_id)
        recent = messages[-self.recent_context_messages :]
        return {
            "summary": str(conversation.get("summary") or ""),
            "recent_messages": [
                {"role": item["role"], "content": str(item["content"])[:600]}
                for item in recent
            ],
        }

    def latest_agent_memory(self, conversation_id: str) -> dict[str, Any]:
        for item in reversed(self.list_messages(conversation_id)):
            if item["role"] != "assistant":
                continue
            memory = (item.get("metadata") or {}).get("agent_memory")
            return dict(memory) if isinstance(memory, dict) else {}
        return {}

    def _refresh_summary(self, connection: Connection, conversation_id: str) -> None:
        rows = connection.execute(
            text(
                """
                SELECT role, content FROM messages WHERE conversation_id = :id
                ORDER BY sequence_no
                """
            ),
            {"id": conversation_id},
        ).mappings().all()
        older = rows[: -self.recent_context_messages]
        if not older:
            return
        lines = []
        for item in older:
            role = "用户" if item["role"] == "user" else "助手"
            compact = " ".join(str(item["content"]).split())[:300]
            lines.append(f"{role}：{compact}")
        summary = "\n".join(lines)
        if len(summary) > self.summary_max_chars:
            summary = "…\n" + summary[-self.summary_max_chars :]
        connection.execute(
            text("UPDATE conversations SET summary = :summary WHERE id = :id"),
            {"summary": summary, "id": conversation_id},
        )

    def _rotate_agent_thread(self, connection: Connection, conversation_id: str) -> str:
        thread_id = str(uuid4())
        now = _utc_now()
        connection.execute(
            text("UPDATE conversations SET agent_thread_id = :thread_id WHERE id = :id"),
            {"thread_id": thread_id, "id": conversation_id},
        )
        connection.execute(
            text(
                "INSERT INTO agent_threads(conversation_id, thread_id, created_at) "
                "VALUES (:conversation_id, :thread_id, :created_at)"
            ),
            {"conversation_id": conversation_id, "thread_id": thread_id, "created_at": now},
        )
        return thread_id

    def start_run(
        self,
        conversation_id: str,
        *,
        user_content: str,
        message_metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        self.ensure_conversation(conversation_id)
        run_id = str(uuid4())
        turn_id = str(uuid4())
        now = _utc_now()
        try:
            with self._connection() as connection:
                conversation = connection.execute(
                    text(
                        """
                        SELECT title FROM conversations
                        WHERE id = :id AND owner_id = :owner_id
                          AND archived_at IS NULL AND deleted_at IS NULL
                        """
                    ),
                    {"id": conversation_id, "owner_id": self.owner_id},
                ).mappings().first()
                if not conversation:
                    raise KeyError(f"Unknown conversation: {conversation_id}")
                connection.execute(
                    text(
                        """
                        INSERT INTO runs(id, conversation_id, turn_id, status, started_at)
                        VALUES (:id, :conversation_id, :turn_id, 'running', :started_at)
                        """
                    ),
                    {
                        "id": run_id,
                        "conversation_id": conversation_id,
                        "turn_id": turn_id,
                        "started_at": now,
                    },
                )
                if conversation["title"] == "新对话":
                    connection.execute(
                        text("UPDATE conversations SET title = :title WHERE id = :id"),
                        {"title": _title_from_message(user_content), "id": conversation_id},
                    )
                self.add_message(
                    conversation_id,
                    role="user",
                    content=user_content,
                    turn_id=turn_id,
                    metadata=message_metadata,
                    connection=connection,
                )
                connection.execute(
                    text(
                        "UPDATE conversations SET status = 'active', updated_at = :now "
                        "WHERE id = :id"
                    ),
                    {"now": now, "id": conversation_id},
                )
        except IntegrityError as exc:
            raise RuntimeError("当前对话已有正在运行或等待审核的任务。") from exc
        return {"run_id": run_id, "turn_id": turn_id}

    def mark_awaiting_review(self, run_id: str, payload: dict[str, Any]) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT r.conversation_id FROM runs r
                    JOIN conversations c ON c.id = r.conversation_id
                    WHERE r.id = :id AND r.status = 'running' AND c.owner_id = :owner_id
                    """
                ),
                {"id": run_id, "owner_id": self.owner_id},
            ).mappings().first()
            if not row:
                return False
            connection.execute(
                text(
                    "UPDATE runs SET status = 'awaiting_review', pending_review_json = :payload "
                    "WHERE id = :id"
                ),
                {"payload": json.dumps(payload, ensure_ascii=False), "id": run_id},
            )
            connection.execute(
                text(
                    "UPDATE conversations SET status = 'awaiting_review', updated_at = :now "
                    "WHERE id = :id"
                ),
                {"now": _utc_now(), "id": row["conversation_id"]},
            )
        return True

    def begin_review_resume(self, run_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE runs SET status = 'running'
                    WHERE id = :id AND status = 'awaiting_review'
                      AND conversation_id IN (
                        SELECT id FROM conversations WHERE owner_id = :owner_id
                      )
                    """
                ),
                {"id": run_id, "owner_id": self.owner_id},
            )
        return result.rowcount == 1

    def complete_run(
        self,
        run_id: str,
        *,
        assistant_content: str,
        metadata: dict[str, Any] | None = None,
        kind: str = "text",
    ) -> bool:
        now = _utc_now()
        with self._connection() as connection:
            run = connection.execute(
                text(
                    """
                    SELECT r.conversation_id, r.turn_id, r.status FROM runs r
                    JOIN conversations c ON c.id = r.conversation_id
                    WHERE r.id = :id AND c.owner_id = :owner_id
                    """
                ),
                {"id": run_id, "owner_id": self.owner_id},
            ).mappings().first()
            if not run or run["status"] not in ACTIVE_RUN_STATUSES:
                return False
            self.add_message(
                run["conversation_id"],
                role="assistant",
                content=assistant_content,
                turn_id=run["turn_id"],
                kind=kind,
                metadata=metadata,
                connection=connection,
            )
            connection.execute(
                text(
                    """
                    UPDATE runs SET status = 'completed', pending_review_json = '{}',
                                    finished_at = :now WHERE id = :id
                    """
                ),
                {"now": now, "id": run_id},
            )
            connection.execute(
                text(
                    "UPDATE conversations SET status = 'active', updated_at = :now "
                    "WHERE id = :id"
                ),
                {"now": now, "id": run["conversation_id"]},
            )
            self._refresh_summary(connection, str(run["conversation_id"]))
            completed_turns = connection.execute(
                text(
                    "SELECT COUNT(*) FROM runs WHERE conversation_id = :id "
                    "AND status = 'completed'"
                ),
                {"id": run["conversation_id"]},
            ).scalar_one()
            if int(completed_turns) % self.compaction_turns == 0:
                self._rotate_agent_thread(connection, str(run["conversation_id"]))
        return True

    def fail_run(self, run_id: str, public_message: str) -> bool:
        return self.complete_run(
            run_id,
            assistant_content=public_message,
            metadata={"failed": True},
            kind="error",
        )

    def cancel_active_run(self, conversation_id: str) -> bool:
        now = _utc_now()
        with self._connection() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE runs SET status = 'canceled', pending_review_json = '{}',
                                    finished_at = :now
                    WHERE conversation_id = :id
                      AND status IN ('running', 'awaiting_review')
                      AND conversation_id IN (
                        SELECT id FROM conversations WHERE owner_id = :owner_id
                      )
                    """
                ),
                {"now": now, "id": conversation_id, "owner_id": self.owner_id},
            )
            if result.rowcount:
                self._rotate_agent_thread(connection, conversation_id)
                connection.execute(
                    text(
                        "UPDATE conversations SET status = 'active', updated_at = :now "
                        "WHERE id = :id"
                    ),
                    {"now": now, "id": conversation_id},
                )
        return result.rowcount > 0

    def recover_interrupted_runs(self, *, all_owners: bool = False) -> int:
        now = _utc_now()
        owner_filter = "" if all_owners else " AND c.owner_id = :owner_id"
        params = {"now": now, "owner_id": self.owner_id}
        with self._connection() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT r.id, r.conversation_id, r.turn_id FROM runs r
                    JOIN conversations c ON c.id = r.conversation_id
                    WHERE r.status = 'running'
                    """ + owner_filter
                ),
                params,
            ).mappings().all()
            for row in rows:
                result = connection.execute(
                    text(
                        "UPDATE runs SET status = 'failed', finished_at = :now "
                        "WHERE id = :id AND status = 'running'"
                    ),
                    {"now": now, "id": row["id"]},
                )
                if not result.rowcount:
                    continue
                owner = connection.execute(
                    text("SELECT owner_id FROM conversations WHERE id = :id"),
                    {"id": row["conversation_id"]},
                ).scalar_one()
                owner_store = self if owner == self.owner_id else self.for_owner(str(owner))
                owner_store.add_message(
                    row["conversation_id"],
                    role="assistant",
                    content="服务在任务执行期间重启，本轮未完成。请重新提交该问题。",
                    turn_id=row["turn_id"],
                    kind="error",
                    metadata={"recovered_after_restart": True},
                    connection=connection,
                )
                self._rotate_agent_thread(connection, str(row["conversation_id"]))
                connection.execute(
                    text(
                        "UPDATE conversations SET status = 'active', updated_at = :now "
                        "WHERE id = :id"
                    ),
                    {"now": now, "id": row["conversation_id"]},
                )
        return len(rows)

    def get_active_run(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT r.id, r.turn_id, r.status, r.pending_review_json, r.started_at
                    FROM runs r
                    JOIN conversations c ON c.id = r.conversation_id
                    WHERE r.conversation_id = :id AND c.owner_id = :owner_id
                      AND r.status IN ('running', 'awaiting_review')
                    ORDER BY r.started_at DESC LIMIT 1
                    """
                ),
                {"id": conversation_id, "owner_id": self.owner_id},
            ).mappings().first()
        if not row:
            return None
        result = dict(row)
        try:
            result["pending_review"] = json.loads(result.pop("pending_review_json"))
        except (TypeError, json.JSONDecodeError):
            result["pending_review"] = {}
            result.pop("pending_review_json", None)
        return result
