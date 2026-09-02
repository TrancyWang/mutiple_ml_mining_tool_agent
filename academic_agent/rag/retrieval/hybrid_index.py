from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..providers.embedding import OpenAICompatibleClient, cosine
from ..common.schemas import SearchHit, SearchResponse
from ..storage.filesystem import RAGStorage


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text.lower())


def lexical_score(query: str, text: str) -> float:
    q = Counter(tokens(query))
    d = Counter(tokens(text))
    if not q or not d:
        return 0.0
    matched = sum(min(count, d[token]) for token, count in q.items())
    coverage = matched / max(1, sum(q.values()))
    phrase = 0.15 if query.lower() in text.lower() else 0.0
    return min(1.0, coverage + phrase)


class LocalHybridIndex:
    def __init__(self, storage: RAGStorage, llm: OpenAICompatibleClient | None = None):
        self.storage = storage
        self.llm = llm or OpenAICompatibleClient()

    def build(self, dataset_id: str, document_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        vectors = self.llm.embed_many([c["text"] for c in chunks])
        indexed = []
        for chunk, vector in zip(chunks, vectors):
            indexed.append({**chunk, "vector": vector, "token_count": len(tokens(chunk["text"]))})
        path = self.storage.document_dir(dataset_id, document_id) / "index.json"
        self.storage.write_json(path, {"version": 1, "chunks": indexed})
        return {"chunks": len(indexed), "path": str(path)}

    def _load_indexes(self, dataset_id: str, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        allowed = set(document_ids or [])
        rows: list[dict[str, Any]] = []
        for doc in self.storage.list_documents(dataset_id):
            if not doc or (allowed and doc.get("id") not in allowed):
                continue
            path = self.storage.document_dir(dataset_id, doc["id"]) / "index.json"
            payload = self.storage.read_json(path, {})
            rows.extend(payload.get("chunks", []))
        return rows

    def search(self, dataset_id: str, query: str, top_k: int = 6, document_ids: list[str] | None = None, filters: dict[str, Any] | None = None) -> SearchResponse:
        started = time.monotonic()
        filters = filters or {}
        query_vector = self.llm.embed_many([query])[0]
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._load_indexes(dataset_id, document_ids):
            metadata = {**chunk.get("metadata", {}), "page": chunk.get("page"), "block_type": chunk.get("block_type"), "heading_path": chunk.get("heading_path", [])}
            if any(str(metadata.get(key)) != str(value) for key, value in filters.items()):
                continue
            lex = lexical_score(query, chunk.get("text", ""))
            vec = max(0.0, cosine(query_vector, chunk.get("vector", [])))
            score = 0.55 * lex + 0.45 * vec
            if score > 0:
                scored.append((score, {**chunk, "lexical_score": lex, "vector_score": vec}))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = []
        for score, chunk in scored[:top_k]:
            hits.append(SearchHit(chunk_id=chunk["id"], document_id=chunk["document_id"], document_name=chunk["document_name"], text=chunk["text"], score=round(score, 6), lexical_score=round(chunk["lexical_score"], 6), vector_score=round(chunk["vector_score"], 6), metadata={k: v for k, v in chunk.items() if k in {"block_id", "block_type", "page", "bbox", "heading_path", "keywords", "chunk_index", "is_atomic", "metadata"}}))
        return SearchResponse(query=query, hits=hits, took_ms=int((time.monotonic() - started) * 1000))
