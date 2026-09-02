from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    lexical_score: float
    vector_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    hits: list[SearchHit]
    took_ms: int


@dataclass
class Citation:
    rank: int
    document_id: str
    document_name: str
    page: int | None
    block_type: str
    snippet: str
    score: float
    heading_path: list[str] = field(default_factory=list)


@dataclass
class RAGAnswer:
    answer: str
    query: str
    citations: list[Citation] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    used_retrieval: bool = False
