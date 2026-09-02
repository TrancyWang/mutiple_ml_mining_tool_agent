from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from ..ingestion.assets import render_pdf_assets
from ..ingestion.chunker import build_chunks
from ..providers.embedding import OpenAICompatibleClient
from ..ingestion.parser import DocumentParser, add_transformer_fields
from ..retrieval.hybrid_index import LocalHybridIndex
from ..storage.filesystem import RAGStorage


class DocumentRAGService:
    """目标项目内的文档 RAG 服务。

    它把用户上传文件/当前工作区文件转换为结构化文档索引，再把检索结果交给
    Academic Agent 原有的 Qwen-Agent 负责最终回答。这样不会启动第二个服务，
    也不会替换目标项目现有的数据分析和机器学习工具链。
    """

    SUPPORTED = {"pdf", "md", "markdown", "txt", "log", "csv", "json", "jsonl", "docx", "png", "jpg", "jpeg", "webp"}

    def __init__(self, root: Path | None = None):
        self.storage = RAGStorage(root)
        self.parser = DocumentParser()
        self.llm = OpenAICompatibleClient()
        self.index = LocalHybridIndex(self.storage, self.llm)

    @staticmethod
    def _document_id(path: Path) -> str:
        return "doc_" + hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _candidates(query: str) -> list[str]:
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{2,}", query)
        compact = " ".join(dict.fromkeys(terms))
        candidates = [query]
        if compact and compact != query:
            candidates.append(compact)
        if any(word in query for word in ("怎么", "如何", "配置", "步骤", "方法")):
            candidates.append(f"{compact} 配置 步骤 方法")
        return list(dict.fromkeys(candidates))

    def _dataset_id(self, workspace_path: str | None) -> str:
        return self.storage.dataset_key(workspace_path)

    def _normalize_paths(self, paths: Iterable[str | Path]) -> list[Path]:
        output: list[Path] = []
        seen: set[str] = set()
        for value in paths:
            path = Path(value).expanduser().resolve()
            if not path.is_file() or path.suffix.lower().lstrip(".") not in self.SUPPORTED:
                continue
            key = str(path)
            if key not in seen:
                seen.add(key)
                output.append(path)
        return output

    def ingest(self, workspace_path: str | None, paths: Iterable[str | Path]) -> dict[str, Any]:
        dataset_id = self._dataset_id(workspace_path)
        indexed: list[dict[str, Any]] = []
        skipped: list[str] = []
        errors: list[dict[str, str]] = []
        for path in self._normalize_paths(paths):
            document_id = self._document_id(path)
            stat = path.stat()
            signature = f"{stat.st_size}:{stat.st_mtime_ns}"
            old = self.storage.get_metadata(dataset_id, document_id)
            if old and old.get("signature") == signature and (self.storage.document_dir(dataset_id, document_id) / "index.json").exists():
                skipped.append(path.name)
                continue
            try:
                local_source = self.storage.save_source(dataset_id, document_id, path)
                parsed = self.parser.parse(local_source, document_id)
                blocks = parsed["blocks"]
                if local_source.suffix.lower() == ".pdf":
                    blocks.extend(render_pdf_assets(local_source, self.storage.document_dir(dataset_id, document_id), document_id, blocks))
                blocks = add_transformer_fields(blocks)
                document_dir = self.storage.document_dir(dataset_id, document_id)
                self.storage.write_json(document_dir / "blocks.json", blocks)
                (document_dir / "output.md").write_text("\n\n".join(block.get("text", "") for block in blocks), encoding="utf-8")
                chunks = build_chunks(blocks, document_id, path.name)
                self.storage.write_json(document_dir / "chunks.json", chunks)
                result = self.index.build(dataset_id, document_id, chunks)
                self.storage.save_metadata(dataset_id, document_id, {
                    "id": document_id,
                    "dataset_id": dataset_id,
                    "name": path.name,
                    "source_path": str(path),
                    "extension": path.suffix.lower().lstrip("."),
                    "size": stat.st_size,
                    "signature": signature,
                    "status": "ready",
                    "block_count": len(blocks),
                    "chunk_count": result["chunks"],
                })
                indexed.append({"document_id": document_id, "name": path.name, "chunks": result["chunks"]})
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return {"dataset_id": dataset_id, "indexed": indexed, "skipped": skipped, "errors": errors}

    def search(self, workspace_path: str | None, query: str, paths: Iterable[str | Path], top_k: int = 6) -> dict[str, Any]:
        dataset_id = self._dataset_id(workspace_path)
        ingest_result = self.ingest(workspace_path, paths)
        candidates = self._candidates(query)
        all_hits = {}
        chosen = query
        for candidate in candidates:
            result = self.index.search(dataset_id, candidate, top_k=top_k)
            for hit in result.hits:
                all_hits[hit.chunk_id] = hit
            current = sorted(all_hits.values(), key=lambda item: item.score, reverse=True)
            if current and (current[0].score >= 0.16 or current[0].lexical_score >= 0.34):
                chosen = candidate
                break
        hits = sorted(all_hits.values(), key=lambda item: item.score, reverse=True)[:top_k]
        return {
            "success": True,
            "query": query,
            "rewritten_queries": [candidate for candidate in candidates if candidate != query and candidate == chosen],
            "used_retrieval": bool(hits),
            "ingest": ingest_result,
            "hits": [
                {
                    "rank": index,
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "document_name": hit.document_name,
                    "score": hit.score,
                    "page": hit.metadata.get("page"),
                    "block_type": hit.metadata.get("block_type", "text"),
                    "heading_path": hit.metadata.get("heading_path", []),
                    "text": hit.text[:4000],
                    "citation": f"【{index}】",
                }
                for index, hit in enumerate(hits, 1)
            ],
        }

    def answer(self, workspace_path: str | None, query: str, paths: Iterable[str | Path], top_k: int = 6) -> dict[str, Any]:
        result = self.search(workspace_path, query, paths, top_k)
        if not result["hits"]:
            result["answer"] = "当前文档中没有找到足够相关的证据。"
            return result
        result["answer"] = "\n\n".join(f"{hit['text']} {hit['citation']}" for hit in result["hits"][:3])
        return result


_document_rag_service: DocumentRAGService | None = None


def get_document_rag_service() -> DocumentRAGService:
    """Lazily create storage so importing Academic Agent never writes to disk."""
    global _document_rag_service
    if _document_rag_service is None:
        _document_rag_service = DocumentRAGService()
    return _document_rag_service
