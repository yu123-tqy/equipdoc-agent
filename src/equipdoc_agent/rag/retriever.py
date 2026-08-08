from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import Settings
from .index_manifest import expected_index_manifest, read_index_manifest


def _tokenize(text: str) -> list[str]:
    """Return deterministic Chinese/ASCII tokens for model-free retrieval.

    Character bi/tri-grams make the fallback useful for domain terms such as
    ``齿面磨损`` and ``包络解调`` without requiring jieba.  When jieba is
    installed, its word tokens are additive rather than replacing the stable
    fallback, so the public evaluation does not change completely by environment.
    """
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", normalized)
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        if len(segment) >= 3:
            tokens.extend(segment[index : index + 3] for index in range(len(segment) - 2))
    try:
        import jieba

        tokens.extend(token.strip().lower() for token in jieba.lcut(text) if token.strip())
    except ImportError:
        pass
    return tokens


def _searchable_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    keywords = metadata.get("keywords") or []
    return "\n".join(
        str(value)
        for value in (
            item.get("title", ""),
            item.get("heading_path", ""),
            " ".join(str(keyword) for keyword in keywords),
            item.get("text", ""),
        )
        if value
    )


def _matches_filters(item: dict[str, Any], filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    metadata = item.get("metadata") or {}
    for key, expected in filters.items():
        if expected in {"", "general", None}:
            continue
        if str(metadata.get(key)) != str(expected):
            return False
    return True


def _source_priority(item: dict[str, Any]) -> float:
    """Return the explicit source precedence used for near-tie reranking."""
    raw = (item.get("metadata") or {}).get("source_priority", 0)
    try:
        return max(0.0, min(100.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class KnowledgeRetriever:
    """Lazy, degradable retrieval layer.

    Lexical retrieval works with the base installation. Dense retrieval is added
    only when the optional RAG dependencies and a Chroma collection are present.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = self._load_chunks(settings.rag_chunks_path)
        self.chunk_by_id = {item["chunk_id"]: item for item in self.chunks}
        self.tokenized = [_tokenize(_searchable_text(item)) for item in self.chunks]
        self._bm25 = None
        self._dense_collection = None
        self._embedding_model = None
        self.warnings: list[str] = []
        self._init_bm25()
        self._init_dense_if_available()

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _init_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.warnings.append("rank-bm25 unavailable; using token-overlap retrieval")
            return
        self._bm25 = BM25Okapi(self.tokenized)

    def _init_dense_if_available(self) -> None:
        if not self.settings.rag_db_dir.exists():
            self.warnings.append("vector DB unavailable; dense retrieval disabled")
            return
        if not self.settings.rag_chunks_path.is_file():
            self.warnings.append("knowledge chunks unavailable; dense retrieval disabled")
            return
        expected_manifest = expected_index_manifest(self.settings, len(self.chunks))
        actual_manifest = read_index_manifest(self.settings.rag_db_dir)
        if actual_manifest != expected_manifest:
            self.warnings.append(
                "vector DB manifest is missing or stale; dense retrieval disabled until rebuild"
            )
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(path=str(self.settings.rag_db_dir))
            self._dense_collection = client.get_collection(self.settings.rag_collection)
            self._embedding_model = SentenceTransformer(
                self.settings.embedding_model,
                trust_remote_code=True,
            )
        except Exception as exc:
            self.warnings.append(f"dense retrieval disabled: {type(exc).__name__}")

    def _lexical_search(
        self,
        query: str,
        filters: dict[str, str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        query_tokens = _tokenize(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
        else:
            query_set = set(query_tokens)
            scores = [
                len(query_set.intersection(tokens)) / max(1, len(query_set))
                for tokens in self.tokenized
            ]
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        hits = []
        for index, score in ranked:
            if float(score) <= 0:
                continue
            chunk = self.chunks[index]
            if not _matches_filters(chunk, filters):
                continue
            hits.append({**chunk, "lexical_score": float(score)})
            if len(hits) >= limit:
                break
        return hits

    def _dense_search(
        self,
        query: str,
        filters: dict[str, str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if self._dense_collection is None or self._embedding_model is None:
            return []
        embedding = self._embedding_model.encode([query], normalize_embeddings=True).tolist()[0]
        raw_filters = {
            key: value
            for key, value in (filters or {}).items()
            if value not in {None, "", "general"}
        }
        where = None
        if len(raw_filters) == 1:
            where = raw_filters
        elif len(raw_filters) > 1:
            where = {"$and": [{key: {"$eq": value}} for key, value in raw_filters.items()]}
        result = self._dense_collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        hits = []
        for chunk_id, distance in zip(
            result.get("ids", [[]])[0],
            result.get("distances", [[]])[0],
        ):
            chunk = self.chunk_by_id.get(chunk_id)
            if chunk:
                hits.append({**chunk, "dense_score": 1.0 / (1.0 + float(distance))})
        return hits

    def search(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        final_k = top_k or self.settings.rag_top_k
        pool_size = max(20, final_k)
        lexical = self._lexical_search(query, filters, pool_size)
        dense = self._dense_search(query, filters, pool_size)
        scores = _rrf(
            [
                [item["chunk_id"] for item in lexical],
                [item["chunk_id"] for item in dense],
            ]
        )
        merged: dict[str, dict[str, Any]] = {}
        for item in lexical + dense:
            chunk_id = item["chunk_id"]
            merged.setdefault(chunk_id, {}).update(item)
            merged[chunk_id]["rrf_score"] = scores.get(chunk_id, 0.0)
            priority = _source_priority(item)
            merged[chunk_id]["source_priority"] = priority
            # Priority is deliberately only a near-tie breaker.  With RRF k=60,
            # dividing by one million cannot jump an otherwise relevant result
            # several lexical ranks merely because its source is more authoritative.
            merged[chunk_id]["rank_score"] = scores.get(chunk_id, 0.0) + priority / 1_000_000.0
        ranked = sorted(
            merged.values(),
            key=lambda item: item.get("rank_score", 0.0),
            reverse=True,
        )
        # The corpus contains several chunks per document.  Reserve at least 80%
        # of top-k for distinct documents, then let the remaining slots recover
        # complementary chunks from the same source (for example, mechanism and
        # field-review evidence stored under separate headings).
        selected = []
        selected_docs: set[str] = set()
        diversity_target = max(1, (final_k * 4 + 4) // 5)
        for item in ranked:
            doc_id = str(item.get("doc_id", ""))
            if doc_id in selected_docs:
                continue
            selected.append(item)
            selected_docs.add(doc_id)
            if len(selected) >= diversity_target:
                break
        if len(selected) < final_k:
            selected_ids = {item.get("chunk_id") for item in selected}
            selected.extend(item for item in ranked if item.get("chunk_id") not in selected_ids)
        return selected[:final_k]
