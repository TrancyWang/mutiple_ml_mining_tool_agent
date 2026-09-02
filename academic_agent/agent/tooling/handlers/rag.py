"""Document RAG tool bridge for the Academic Agent runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from academic_agent.infrastructure.workspace_manager import workspace_manager


def search_documents(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.rag.application.service import get_document_rag_service

    document_rag_service = get_document_rag_service()
    query = str(kwargs.get("query", "")).strip()
    if not query:
        return {"success": False, "error": "文档检索需要 query 参数。"}
    explicit = kwargs.get("file_paths") or []
    session_files = kwargs.get("_session_files") or []
    paths = explicit or session_files
    if not paths:
        paths = [
            str(workspace_manager.root / relative)
            for relative in workspace_manager.list_files()
            if Path(relative).suffix.lower().lstrip(".") in document_rag_service.SUPPORTED
        ][:200]
    if not paths:
        return {"success": False, "error": "请先上传文档，或在当前工作区放入可解析的文档。"}
    return document_rag_service.answer(
        kwargs.get("_workspace_path") or str(workspace_manager.root),
        query,
        [Path(path) for path in paths],
        top_k=int(kwargs.get("top_k", 6)),
    )
