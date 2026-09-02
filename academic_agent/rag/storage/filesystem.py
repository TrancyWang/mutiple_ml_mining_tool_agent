from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.config import settings


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def safe_name(name: str) -> str:
    value = Path(name or "document").name
    value = re.sub(r"[^\w.()\-\u4e00-\u9fff ]+", "_", value).strip(" .")
    return value[:180] or "document"


class RAGStorage:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or settings.datasets_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def dataset_key(self, workspace_path: str | None) -> str:
        raw = str(Path(workspace_path).resolve()) if workspace_path else "default"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[-120:] or "default"

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.root / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def document_dir(self, dataset_id: str, document_id: str) -> Path:
        path = self.dataset_dir(dataset_id) / "documents" / document_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_source(self, dataset_id: str, document_id: str, source: Path) -> Path:
        destination = self.document_dir(dataset_id, document_id) / safe_name(source.name)
        if destination.resolve() != source.resolve():
            destination.write_bytes(source.read_bytes())
        return destination

    def find_source(self, dataset_id: str, document_id: str) -> Path | None:
        directory = self.document_dir(dataset_id, document_id)
        ignored = {"document.json", "blocks.json", "chunks.json", "index.json"}
        for path in directory.iterdir():
            if path.is_file() and path.name not in ignored:
                return path
        return None

    def metadata_path(self, dataset_id: str, document_id: str) -> Path:
        return self.document_dir(dataset_id, document_id) / "document.json"

    def save_metadata(self, dataset_id: str, document_id: str, value: dict[str, Any]) -> None:
        self.write_json(self.metadata_path(dataset_id, document_id), value)

    def get_metadata(self, dataset_id: str, document_id: str) -> dict[str, Any] | None:
        value = self.read_json(self.metadata_path(dataset_id, document_id))
        return value if isinstance(value, dict) else None

    def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        documents_dir = self.dataset_dir(dataset_id) / "documents"
        return [
            value for path in sorted(documents_dir.glob("*/document.json"))
            if isinstance(value := self.read_json(path), dict)
        ]
