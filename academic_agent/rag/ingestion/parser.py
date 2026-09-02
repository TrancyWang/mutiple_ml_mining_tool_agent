from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any


def _block(block_id: str, block_type: str, text: str, page: int | None = None, **meta: Any) -> dict[str, Any]:
    value = {
        "id": block_id,
        "block_type": block_type,
        "text": text.strip(),
        "page": page,
        "bbox": meta.pop("bbox", None),
        "metadata": meta,
    }
    return value


class DocumentParser:
    """Parse common document formats into a stable block contract.

    Heavy OCR engines stay behind adapters. This lets the API work locally and
    makes MinerU/PaddleOCR/DeepSeek replaceable deployment choices.
    """

    def parse(self, path: Path, document_id: str) -> dict[str, Any]:
        ext = path.suffix.lower().lstrip(".")
        if ext == "pdf":
            blocks = self._pdf(path, document_id)
        elif ext in {"md", "markdown"}:
            blocks = self._markdown(path.read_text(encoding="utf-8", errors="ignore"), document_id)
        elif ext in {"txt", "log"}:
            blocks = self._plain(path.read_text(encoding="utf-8", errors="ignore"), document_id)
        elif ext == "csv":
            blocks = self._csv(path, document_id)
        elif ext in {"json", "jsonl"}:
            blocks = self._json(path, document_id)
        elif ext == "docx":
            blocks = self._docx(path, document_id)
        elif ext in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
            blocks = [_block(f"{document_id}:image:0", "image", f"Image document: {path.name}", 1, image_path=str(path))]
        else:
            raise ValueError(f"unsupported extension: {ext}")
        blocks = [b for b in blocks if b["text"] or b["block_type"] == "image"]
        return {"document_id": document_id, "source": path.name, "blocks": blocks}

    def _plain(self, text: str, document_id: str) -> list[dict[str, Any]]:
        return [_block(f"{document_id}:text:{i}", "text", part, page=None) for i, part in enumerate(re.split(r"\n\s*\n", text)) if part.strip()]

    def _markdown(self, text: str, document_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        current: list[str] = []
        heading_path: list[str] = []
        block_index = 0
        in_fence = False

        def flush() -> None:
            nonlocal block_index, current
            value = "\n".join(current).strip()
            if value:
                block_type = "formula" if value.startswith("$$") and value.endswith("$$") else "text"
                blocks.append(_block(f"{document_id}:md:{block_index}", block_type, value, None, heading_path=heading_path.copy()))
                block_index += 1
            current = []

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                in_fence = not in_fence
                current.append(line)
                i += 1
                continue
            if not in_fence:
                heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
                if heading:
                    flush()
                    level, title = len(heading.group(1)), heading.group(2)
                    heading_path = heading_path[: level - 1] + [title]
                    blocks.append(_block(f"{document_id}:heading:{block_index}", "heading", title, None, level=level, heading_path=heading_path.copy()))
                    block_index += 1
                    i += 1
                    continue
                if re.match(r"^\s*\|.*\|\s*$", line) and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
                    flush()
                    table = [line, lines[i + 1]]
                    i += 2
                    while i < len(lines) and "|" in lines[i] and lines[i].strip():
                        table.append(lines[i])
                        i += 1
                    blocks.append(_block(f"{document_id}:table:{block_index}", "table", "\n".join(table), None, heading_path=heading_path.copy()))
                    block_index += 1
                    continue
                image = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line.strip())
                if image:
                    flush()
                    blocks.append(_block(f"{document_id}:image:{block_index}", "image", image.group(1) or image.group(2), None, image_path=image.group(2), heading_path=heading_path.copy()))
                    block_index += 1
                    i += 1
                    continue
                if line.strip().startswith("$$"):
                    flush()
                    formula = [line]
                    i += 1
                    while i < len(lines):
                        formula.append(lines[i])
                        if lines[i].strip().endswith("$$"):
                            i += 1
                            break
                        i += 1
                    blocks.append(_block(f"{document_id}:formula:{block_index}", "formula", "\n".join(formula), None, heading_path=heading_path.copy()))
                    block_index += 1
                    continue
            if not line.strip():
                flush()
            else:
                current.append(line)
            i += 1
        flush()
        return blocks

    def _csv(self, path: Path, document_id: str) -> list[dict[str, Any]]:
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig", errors="ignore"))))
        if not rows:
            return []
        header = rows[0]
        body = rows[1:]
        text = "\n".join(" | ".join(row) for row in [header, *body])
        return [_block(f"{document_id}:table:0", "table", text, None, columns=header, row_count=len(body))]

    def _json(self, path: Path, document_id: str) -> list[dict[str, Any]]:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".jsonl":
            value = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            value = json.loads(raw)
        text = json.dumps(value, ensure_ascii=False, indent=2)
        return [_block(f"{document_id}:json:0", "text", text, None, format="json")]

    def _docx(self, path: Path, document_id: str) -> list[dict[str, Any]]:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, re.DOTALL)
        blocks = []
        for i, paragraph in enumerate(paragraphs):
            text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", paragraph, re.DOTALL))
            text = re.sub(r"<[^>]+>", "", text).strip()
            if text:
                blocks.append(_block(f"{document_id}:docx:{i}", "text", text, None))
        return blocks

    def _pdf(self, path: Path, document_id: str) -> list[dict[str, Any]]:
        try:
            import fitz  # type: ignore
        except ImportError:
            remote_blocks = self._remote_ocr(path, document_id)
            if remote_blocks:
                return remote_blocks
            raise RuntimeError("PDF parser unavailable; install PyMuPDF or configure MINERU_API_URL/PADDLEOCR_API_URL/DEEPSEEK_OCR_API_URL")
        blocks: list[dict[str, Any]] = []
        with fitz.open(path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                page_blocks = page.get_text("blocks")
                if page_blocks:
                    for i, item in enumerate(page_blocks):
                        x0, y0, x1, y1, text = item[:5]
                        if text and text.strip():
                            blocks.append(_block(f"{document_id}:pdf:{page_no}:{i}", "text", text, page_no, bbox=[x0, y0, x1, y1], parser="pymupdf"))
        if not blocks:
            remote_blocks = self._remote_ocr(path, document_id)
            if remote_blocks:
                return remote_blocks
            raise RuntimeError("PDF contains no extractable text; configure a remote OCR adapter for scanned PDFs")
        return blocks

    def _remote_ocr(self, path: Path, document_id: str) -> list[dict[str, Any]]:
        url = os.getenv("MINERU_API_URL") or os.getenv("PADDLEOCR_API_URL") or os.getenv("DEEPSEEK_OCR_API_URL")
        if not url:
            return []
        boundary = f"----agentic-rag-{uuid.uuid4().hex}"
        body = b"--" + boundary.encode() + b"\r\n"
        body += f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
        body += b"Content-Type: application/pdf\r\n\r\n" + path.read_bytes() + b"\r\n"
        body += b"--" + boundary.encode() + b"--\r\n"
        request = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=int(os.getenv("OCR_TIMEOUT", "600"))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        return self._remote_blocks(payload, document_id)

    def _remote_blocks(self, payload: dict[str, Any], document_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        content_list = payload.get("content_list") or payload.get("blocks") or []
        pages = payload.get("pages") or []
        if not content_list and pages:
            for page_index, page in enumerate(pages, start=1):
                content_list.extend(page.get("parsing_res_list", []) if isinstance(page, dict) else [])
        for index, item in enumerate(content_list):
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("type") or item.get("block_label") or item.get("label") or "text").lower()
            block_type = "table" if "table" in raw_type else "formula" if "formula" in raw_type or "equation" in raw_type else "image" if "image" in raw_type or "figure" in raw_type else "heading" if "title" in raw_type or "header" in raw_type else "text"
            text = item.get("text") or item.get("block_content") or item.get("content") or item.get("latex") or ""
            page = item.get("page_idx", item.get("page_index", item.get("page_number")))
            if isinstance(page, int) and page == 0:
                page = 1
            blocks.append(_block(f"{document_id}:ocr:{index}", block_type, str(text), page, bbox=item.get("bbox") or item.get("block_bbox"), source_type=raw_type, image_path=item.get("img_path")))
        if not blocks:
            markdown = payload.get("content_md") or payload.get("markdown") or payload.get("content") or ""
            if isinstance(markdown, str) and markdown.strip():
                blocks = self._markdown(markdown, document_id)
        return blocks


def add_transformer_fields(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add lightweight deterministic metadata; an LLM transformer can replace this later."""
    for block in blocks:
        text = block.get("text", "")
        words = re.findall(r"[A-Za-z0-9_\-]{3,}|[\u4e00-\u9fff]{2,}", text)
        seen: list[str] = []
        for word in words:
            if word not in seen:
                seen.append(word)
        block["keywords"] = seen[:12]
        block["summary"] = re.sub(r"\s+", " ", text)[:240]
        block["heading_path"] = block.get("metadata", {}).get("heading_path", [])
    return blocks
