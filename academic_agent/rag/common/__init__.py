"""Shared configuration and data contracts for the RAG subsystem."""

from .config import settings
from .schemas import Citation, RAGAnswer, SearchHit, SearchResponse

__all__ = ["settings", "Citation", "RAGAnswer", "SearchHit", "SearchResponse"]
