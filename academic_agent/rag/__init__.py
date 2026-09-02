"""Structure-aware document ingestion and Agentic RAG for Academic Agent."""

from .application.service import DocumentRAGService, get_document_rag_service

__all__ = ["DocumentRAGService", "get_document_rag_service"]
