from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_CHARS = 420
DEFAULT_MAX_CHARS = 500
DEFAULT_OVERLAP_CHARS = 80


def _parse_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Missing frontmatter in {path}")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError(f"Unclosed frontmatter in {path}") from exc

    metadata: dict[str, Any] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid frontmatter line in {path}: {line}")
        clean_value = value.strip()
        metadata[key.strip()] = (
            [item.strip() for item in clean_value.split(",") if item.strip()]
            if key.strip() == "keywords"
            else clean_value
        )
    return metadata, "\n".join(lines[end + 1 :]).strip()


def _sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    active_path = "正文"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((active_path, text))
        buffer.clear()

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            buffer.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading = match.group(2).strip()
        del heading_stack[level - 1 :]
        while len(heading_stack) < level - 1:
            heading_stack.append("未命名章节")
        heading_stack.append(heading)
        active_path = " > ".join(heading_stack)
    flush()
    return sections


def _source_priority(metadata: dict[str, Any]) -> int:
    raw = metadata.get("source_priority")
    if raw not in {None, ""}:
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source_priority must be an integer, got {raw!r}") from exc
    authority = str(metadata.get("source_authority", ""))
    source_type = str(metadata.get("source_type", ""))
    if authority == "experiment_plan_primary":
        return 100
    if authority == "contract_supporting_spec":
        return 70
    if source_type == "authoritative_summary":
        return 60
    if source_type == "self_written_note":
        return 40
    return 30


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [piece for piece in re.split(r"(?<=[。！？；;])|\n+", text) if piece]
    if not sentences:
        sentences = [text]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(sentence):
                end = min(len(sentence), start + max_chars)
                chunks.append(sentence[start:end])
                if end == len(sentence):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = overlap + sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _load_overrides(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_chunk_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id", "")).strip()
            doc_id = str(item.get("doc_id", "")).strip()
            text = str(item.get("text", ""))
            if not chunk_id or not doc_id or not text:
                raise ValueError(f"Invalid chunk override at {path}:{line_number}")
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate override chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            grouped.setdefault(doc_id, []).append(item)
    return grouped


def _override_chunks(
    items: list[dict[str, Any]],
    doc_metadata: dict[str, Any],
    title: str,
    source_path: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    chunks = []
    priority = _source_priority(doc_metadata)
    for item in items:
        text = str(item["text"])
        if len(text) > max_chars:
            raise ValueError(
                f"Override {item['chunk_id']} has {len(text)} chars; maximum is {max_chars}"
            )
        heading_path = str(item.get("heading_path", "正文"))
        chunk_metadata = {
            **doc_metadata,
            **(item.get("metadata") or {}),
            "source_path": source_path,
            "heading_path": heading_path,
            "chunk_id": item["chunk_id"],
            "source_priority": priority,
            "char_count": len(text),
        }
        chunks.append(
            {
                "chunk_id": item["chunk_id"],
                "doc_id": item["doc_id"],
                "title": str(item.get("title") or title),
                "heading_path": heading_path,
                "text": text,
                "metadata": chunk_metadata,
            }
        )
    return chunks


def _markdown_chunks(
    doc_id: str,
    title: str,
    metadata: dict[str, Any],
    body: str,
    source_path: str,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    priority = _source_priority(metadata)
    for heading_path, section_text in _sections(body):
        prefix = f"文档：{title}\n章节：{heading_path}\n\n"
        body_limit = max(160, max_chars - len(prefix))
        for segment in _split_text(section_text, body_limit, overlap_chars):
            text = prefix + segment
            if len(text) > max_chars:
                raise ValueError(f"Chunk exceeds maximum length in {doc_id}: {len(text)}")
            chunk_id = f"{doc_id}_c{len(chunks) + 1:03d}"
            chunk_metadata = {
                **metadata,
                "source_path": source_path,
                "heading_path": heading_path,
                "chunk_id": chunk_id,
                "source_priority": priority,
                "chunk_target_chars": target_chars,
                "chunk_max_chars": max_chars,
                "chunk_overlap_chars": overlap_chars,
                "char_count": len(text),
            }
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "heading_path": heading_path,
                    "text": text,
                    "metadata": chunk_metadata,
                }
            )
    return chunks


def build_chunks(
    knowledge_dir: Path,
    chunk_overrides_path: Path | None = None,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    if not 0 < target_chars <= max_chars:
        raise ValueError("target_chars must be positive and no larger than max_chars")

    overrides = _load_overrides(chunk_overrides_path)
    chunks: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    known_doc_ids: set[str] = set()
    for path in sorted(knowledge_dir.glob("*.md")):
        metadata, body = _parse_frontmatter(path.read_text(encoding="utf-8"), path)
        doc_id = str(metadata.get("doc_id", "")).strip()
        title = str(metadata.get("title", "")).strip()
        if not doc_id or not title:
            raise ValueError(f"doc_id and title are required in {path}")
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen_doc_ids.add(doc_id)
        known_doc_ids.add(doc_id)
        source_path = path.relative_to(ROOT).as_posix()
        doc_chunks = (
            _override_chunks(overrides[doc_id], metadata, title, source_path, max_chars)
            if doc_id in overrides
            else _markdown_chunks(
                doc_id,
                title,
                metadata,
                body,
                source_path,
                target_chars,
                max_chars,
                overlap_chars,
            )
        )
        for item in doc_chunks:
            if item["chunk_id"] in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {item['chunk_id']}")
            seen_chunk_ids.add(item["chunk_id"])
            chunks.append(item)

    unknown_override_docs = set(overrides) - known_doc_ids
    if unknown_override_docs:
        raise ValueError(f"Overrides reference unknown doc_ids: {sorted(unknown_override_docs)}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic lexical knowledge chunks.")
    parser.add_argument("--knowledge-dir", type=Path, default=ROOT / "data/knowledge")
    parser.add_argument("--output", type=Path, default=ROOT / "data/knowledge_chunks.jsonl")
    parser.add_argument(
        "--chunk-overrides",
        type=Path,
        default=ROOT / "data/knowledge_chunk_overrides.jsonl",
        help="Optional pre-chunked JSONL for documents that require exact source-page anchors.",
    )
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--check", action="store_true", help="Fail if output is not up to date.")
    args = parser.parse_args()

    chunks = build_chunks(
        args.knowledge_dir.resolve(),
        args.chunk_overrides.resolve() if args.chunk_overrides else None,
        args.target_chars,
        args.max_chars,
        args.overlap_chars,
    )
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in chunks) + "\n"
    doc_count = len({item["doc_id"] for item in chunks})
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != payload:
            raise SystemExit(
                f"Knowledge chunks are stale: run {Path(__file__).name} and commit the output."
            )
        print(f"Verified {len(chunks)} chunks from {doc_count} documents: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Built {len(chunks)} chunks from {doc_count} documents: {args.output}")


if __name__ == "__main__":
    main()
