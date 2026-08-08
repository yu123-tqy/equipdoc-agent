from __future__ import annotations

import html
from typing import Any, Iterable


DISPLAY_LIMIT = 5
SCORE_FIELDS = ("rank_score", "rrf_score", "lexical_score", "dense_score")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def sanitize_retrieval_hits(
    hits: Iterable[dict[str, Any]] | None,
    *,
    limit: int = DISPLAY_LIMIT,
) -> list[dict[str, Any]]:
    """Return a small JSON-safe snapshot for the browser-side Gradio state.

    Only public retrieval fields are retained.  In particular, filesystem paths,
    model objects, embeddings, and arbitrary metadata never reach the UI.
    """
    safe_hits: list[dict[str, Any]] = []
    for item in list(hits or [])[: max(0, limit)]:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        safe_item: dict[str, Any] = {
            "doc_id": str(item.get("doc_id", "unknown")),
            "chunk_id": str(item.get("chunk_id", "unknown")),
            "title": str(item.get("title", "未命名文档")),
            "heading_path": str(
                item.get("heading_path")
                or metadata.get("heading_path")
                or metadata.get("section")
                or "正文"
            ),
            "text": str(item.get("text", "")),
        }
        for field in SCORE_FIELDS:
            number = _finite_float(item.get(field))
            if number is not None:
                safe_item[field] = number
        priority = _finite_float(item.get("source_priority", metadata.get("source_priority")))
        if priority is not None:
            safe_item["source_priority"] = priority
        safe_hits.append(safe_item)
    return safe_hits


def render_retrieval_hits_markdown(hits: Iterable[dict[str, Any]] | None) -> str:
    """Render a sanitized Top-K retrieval trace for the demo page."""
    safe_hits = sanitize_retrieval_hits(hits)
    if not safe_hits:
        return "当前回答没有可展示的知识库召回片段。"

    lines = [
        f"## 本次知识库召回 Top {len(safe_hits)}",
        "",
        "> 以下内容是生成本次回答时保存的检索快照；结果已执行混合检索与文档多样性重排，未重新检索。",
    ]
    for rank, item in enumerate(safe_hits, start=1):
        title = html.escape(item["title"])
        doc_id = html.escape(item["doc_id"])
        chunk_id = html.escape(item["chunk_id"])
        heading_path = html.escape(item["heading_path"])
        score_parts = []
        score_labels = {
            "rank_score": "融合分数",
            "rrf_score": "RRF",
            "lexical_score": "词法",
            "dense_score": "向量",
        }
        for field in SCORE_FIELDS:
            if field in item:
                score_parts.append(f"{score_labels[field]} {item[field]:.4f}")
        if "source_priority" in item:
            score_parts.append(f"来源优先级 {item['source_priority']:.0f}")
        scores = "；".join(score_parts) or "当前后端未返回可展示分数"
        text_lines = html.escape(item["text"]).splitlines() or [""]
        lines.extend(
            [
                "",
                f"### {rank}. {title}",
                "",
                f"- 文档：`{doc_id}`",
                f"- Chunk：`{chunk_id}`",
                f"- 章节：{heading_path}",
                f"- 排序信息：{scores}",
                "",
                *[f"> {line}" if line else ">" for line in text_lines],
            ]
        )
    return "\n".join(lines)
