from __future__ import annotations

import re
from typing import Any


ATOMIC_TYPES = {"table", "formula", "image"}


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[。！？.!?；;])\s+|\n+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = current[-overlap:] + " " if overlap else ""
        current += sentence + " "
    if current.strip():
        pieces.append(current.strip())
    return pieces


def build_chunks(blocks: list[dict[str, Any]], document_id: str, document_name: str, max_chars: int = 1200, overlap: int = 160) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        text = (block.get("text") or "").strip()
        if not text and block.get("block_type") != "image":
            continue
        block_type = block.get("block_type", "text")
        pieces = [text] if block_type in ATOMIC_TYPES else _split_text(text, max_chars, overlap)
        for piece_index, piece in enumerate(pieces):
            above = ""
            below = ""
            if block_type in ATOMIC_TYPES:
                if block_index > 0:
                    above = (blocks[block_index - 1].get("text") or "")[-240:]
                if block_index + 1 < len(blocks):
                    below = (blocks[block_index + 1].get("text") or "")[:240]
            effective_text = "\n".join(part for part in [above, piece, below] if part).strip()
            chunks.append({
                "id": f"{document_id}:chunk:{len(chunks)}",
                "document_id": document_id,
                "document_name": document_name,
                "text": effective_text,
                "source_text": piece,
                "block_id": block.get("id"),
                "block_type": block_type,
                "page": block.get("page"),
                "bbox": block.get("bbox"),
                "heading_path": block.get("heading_path", []),
                "keywords": block.get("keywords", []),
                "summary": block.get("summary", ""),
                "chunk_index": piece_index,
                "total_chunks_in_block": len(pieces),
                "is_atomic": block_type in ATOMIC_TYPES,
                "metadata": block.get("metadata", {}),
            })
    return chunks

