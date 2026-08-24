"""Session state shared by the natural-language and button entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import threading
import uuid


@dataclass
class SessionContext:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    uploaded_files: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    schema: dict[str, Any] = field(default_factory=dict)
    selected_columns: dict[str, str] = field(default_factory=dict)
    completed_tasks: list[dict[str, Any]] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)

    @property
    def current_file(self) -> Path | None:
        return Path(self.uploaded_files[-1]) if self.uploaded_files else None

    def attach_file(self, path: str, schema: dict[str, Any] | None = None) -> None:
        path = str(Path(path).resolve())
        if path not in self.uploaded_files:
            self.uploaded_files.append(path)
        if schema is not None:
            self.schema = schema

    def record_task(self, name: str, result: dict[str, Any]) -> None:
        self.completed_tasks.append({"name": name, "success": result.get("success", False), "result": result})
        for key in ("output_file", "image_path", "image_url", "visualization"):
            value = result.get(key)
            if value and str(value) not in self.artifacts:
                self.artifacts.append(str(value))
        for key in ("output_files", "artifacts"):
            values = result.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if value and str(value) not in self.artifacts:
                    self.artifacts.append(str(value))


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str = "default") -> SessionContext:
        with self._lock:
            return self._sessions.setdefault(session_id, SessionContext(session_id=session_id))

    def clear(self, session_id: str = "default") -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
