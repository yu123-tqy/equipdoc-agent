from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_root / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _auth_users(raw: str) -> tuple[tuple[str, str], ...]:
    users: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        username, separator, password = item.partition(":")
        username = username.strip()
        if not separator or not username or not password:
            raise ValueError(
                "EQUIPDOC_AUTH_USERS must use username:password pairs separated by commas"
            )
        if username in seen:
            raise ValueError(f"Duplicate EQUIPDOC_AUTH_USERS username: {username}")
        seen.add(username)
        users.append((username, password))
    return tuple(users)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    demo_mode: bool
    agentic_mode: bool
    agent_max_steps: int
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_seconds: float
    bearing_model_path: Path
    bearing_norm_path: Path
    sample_root: Path
    upload_root: Path
    conversation_db_path: Path
    checkpoint_db_path: Path
    conversation_database_url: str
    checkpoint_database_url: str
    auth_users: tuple[tuple[str, str], ...]
    history_page_size: int
    conversation_retention_days: int
    recent_context_messages: int
    conversation_summary_max_chars: int
    memory_compaction_turns: int
    max_upload_bytes: int
    upload_ttl_seconds: int
    rag_enabled: bool
    rag_chunks_path: Path
    rag_db_dir: Path
    rag_collection: str
    embedding_model: str
    rag_top_k: int
    server_host: str
    server_port: int
    gradio_share: bool

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or PROJECT_ROOT).resolve()
        _load_dotenv(root)
        max_upload_mb = max(1, int(os.getenv("EQUIPDOC_MAX_UPLOAD_MB", "8")))
        upload_ttl_hours = max(1.0, float(os.getenv("EQUIPDOC_UPLOAD_TTL_HOURS", "24")))
        return cls(
            project_root=root,
            demo_mode=_env_bool("EQUIPDOC_DEMO_MODE", True),
            agentic_mode=_env_bool("EQUIPDOC_AGENTIC_MODE", False),
            agent_max_steps=max(1, min(4, int(os.getenv("EQUIPDOC_AGENT_MAX_STEPS", "3")))),
            llm_base_url=os.getenv("EQUIPDOC_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/"),
            llm_model=os.getenv("EQUIPDOC_LLM_MODEL", "qwen-equipdoc"),
            llm_api_key=os.getenv("EQUIPDOC_LLM_API_KEY", "EMPTY"),
            llm_timeout_seconds=float(os.getenv("EQUIPDOC_LLM_TIMEOUT_SECONDS", "120")),
            bearing_model_path=_resolve(root, os.getenv("EQUIPDOC_BEARING_MODEL_PATH", "models/bearing_cnn.pth")),
            bearing_norm_path=_resolve(root, os.getenv("EQUIPDOC_BEARING_NORM_PATH", "data/processed/norm.npy")),
            sample_root=_resolve(root, os.getenv("EQUIPDOC_SAMPLE_ROOT", "data/samples")),
            upload_root=_resolve(root, os.getenv("EQUIPDOC_UPLOAD_ROOT", "runtime/uploads")),
            conversation_db_path=_resolve(
                root,
                os.getenv(
                    "EQUIPDOC_CONVERSATION_DB_PATH",
                    "runtime/equipdoc_conversations.sqlite3",
                ),
            ),
            checkpoint_db_path=_resolve(
                root,
                os.getenv(
                    "EQUIPDOC_CHECKPOINT_DB_PATH",
                    "runtime/langgraph_checkpoints.sqlite3",
                ),
            ),
            conversation_database_url=os.getenv("EQUIPDOC_DATABASE_URL", "").strip(),
            checkpoint_database_url=os.getenv(
                "EQUIPDOC_CHECKPOINT_DATABASE_URL", ""
            ).strip(),
            auth_users=_auth_users(os.getenv("EQUIPDOC_AUTH_USERS", "")),
            history_page_size=max(
                5, min(50, int(os.getenv("EQUIPDOC_HISTORY_PAGE_SIZE", "12")))
            ),
            conversation_retention_days=max(
                1, int(os.getenv("EQUIPDOC_CONVERSATION_RETENTION_DAYS", "90"))
            ),
            recent_context_messages=max(
                2, min(30, int(os.getenv("EQUIPDOC_RECENT_CONTEXT_MESSAGES", "8")))
            ),
            conversation_summary_max_chars=max(
                400,
                min(8000, int(os.getenv("EQUIPDOC_CONVERSATION_SUMMARY_MAX_CHARS", "2400"))),
            ),
            memory_compaction_turns=max(
                2, min(100, int(os.getenv("EQUIPDOC_MEMORY_COMPACTION_TURNS", "12")))
            ),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            upload_ttl_seconds=int(upload_ttl_hours * 3600),
            rag_enabled=_env_bool("EQUIPDOC_RAG_ENABLED", True),
            rag_chunks_path=_resolve(root, os.getenv("EQUIPDOC_RAG_CHUNKS_PATH", "data/knowledge_chunks.jsonl")),
            rag_db_dir=_resolve(root, os.getenv("EQUIPDOC_RAG_DB_DIR", "vector_db/chroma_equipdoc")),
            rag_collection=os.getenv("EQUIPDOC_RAG_COLLECTION", "equipdoc_rag"),
            embedding_model=os.getenv("EQUIPDOC_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            rag_top_k=max(1, int(os.getenv("EQUIPDOC_RAG_TOP_K", "5"))),
            server_host=os.getenv("EQUIPDOC_SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("EQUIPDOC_SERVER_PORT", "7860")),
            gradio_share=_env_bool("EQUIPDOC_GRADIO_SHARE", False),
        )

    def ensure_runtime_dirs(self) -> None:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.conversation_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def mode_name(self) -> str:
        if self.demo_mode:
            return "demo"
        return "full_agentic" if self.agentic_mode else "full"

    @property
    def conversation_database(self) -> Path | str:
        return self.conversation_database_url or self.conversation_db_path

    @property
    def checkpoint_database(self) -> Path | str:
        return self.checkpoint_database_url or self.checkpoint_db_path

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_users)
