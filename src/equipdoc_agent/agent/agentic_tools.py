from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings
from ..privacy import redact_sensitive_text
from ..rag import KnowledgeRetriever
from ..tools import analyze_bearing_signal, inspect_bearing_signal
from .planning import ALLOWED_EQUIPMENT, ALLOWED_FAULT_TYPES, ALLOWED_TOOLS


def _validate_search_arguments(
    query: Any,
    equipment: Any,
    fault_type: Any,
    top_k: Any,
) -> tuple[str, dict[str, str] | None, int]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Knowledge search query must be a non-empty string.")
    normalized_query = query.strip()
    if len(normalized_query) > 500:
        raise ValueError("Knowledge search query exceeds 500 characters.")
    if equipment not in {None, ""} and equipment not in ALLOWED_EQUIPMENT:
        raise ValueError(f"Unsupported equipment filter: {equipment}")
    if fault_type not in {None, ""} and fault_type not in ALLOWED_FAULT_TYPES:
        raise ValueError(f"Unsupported fault_type filter: {fault_type}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        raise ValueError("Knowledge search top_k must be an integer between 1 and 5.")
    filters = {
        key: value
        for key, value in {
            "equipment": equipment,
            "fault_type": fault_type,
        }.items()
        if value not in {None, "", "general"}
    }
    return normalized_query, filters or None, top_k


def search_maintenance_knowledge(
    retriever: KnowledgeRetriever | None,
    *,
    query: str,
    equipment: str | None = None,
    fault_type: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Search the project knowledge base and return citation-ready hits."""
    normalized_query, filters, final_k = _validate_search_arguments(
        query,
        equipment,
        fault_type,
        top_k,
    )
    if retriever is None:
        return {
            "_tool_name": "search_maintenance_knowledge",
            "status": "error",
            "error": "Knowledge index is unavailable.",
            "query": normalized_query,
            "filters": filters or {},
            "hits": [],
        }
    raw_hits = retriever.search(normalized_query, filters=filters, top_k=final_k)
    hits = []
    for item in raw_hits:
        doc_id = str(item.get("doc_id", "unknown"))
        chunk_id = str(item.get("chunk_id", "unknown"))
        hits.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "citation": f"{doc_id}#{chunk_id}",
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
                "source_priority": float(item.get("source_priority", 0.0)),
                "rrf_score": float(item.get("rrf_score", 0.0)),
                "lexical_score": (
                    float(item["lexical_score"])
                    if item.get("lexical_score") is not None
                    else None
                ),
                "dense_score": (
                    float(item["dense_score"])
                    if item.get("dense_score") is not None
                    else None
                ),
            }
        )
    return {
        "_tool_name": "search_maintenance_knowledge",
        "status": "ok",
        "query": normalized_query,
        "filters": filters or {},
        "hits": hits,
        "warnings": list(getattr(retriever, "warnings", [])),
    }


def execute_agentic_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    signal_path: str | None,
    settings: Settings,
    retriever: KnowledgeRetriever | None,
) -> dict[str, Any]:
    """Execute one validated tool without accepting a model-provided file path."""
    if tool_name not in ALLOWED_TOOLS:
        return {
            "_tool_name": str(tool_name),
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
        }
    safe_arguments = dict(arguments or {})
    safe_arguments.pop("signal_path", None)
    try:
        if tool_name == "search_maintenance_knowledge":
            return search_maintenance_knowledge(retriever, **safe_arguments)
        if safe_arguments:
            raise ValueError(
                f"{tool_name} does not accept model-provided arguments: "
                f"{sorted(safe_arguments)}"
            )
        if not signal_path:
            raise ValueError("A signal file is required.")
        if tool_name == "inspect_signal":
            result = inspect_bearing_signal(signal_path, settings)
        else:
            result = analyze_bearing_signal(signal_path, settings)
        return {
            "_tool_name": tool_name,
            **result,
            "signal_file": Path(signal_path).name,
        }
    except Exception as exc:
        return {
            "_tool_name": tool_name,
            "status": "error",
            "error": redact_sensitive_text(
                f"{type(exc).__name__}: {exc}",
                project_root=settings.project_root,
            ),
        }
