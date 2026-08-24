"""会话存储端口及 SQLite 适配器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from academic_agent.repositories.sqlite_store import SQLiteConversationStore, sqlite_memory


class ConversationRepository(Protocol):
    def list_chat_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]: ...
    def list_work_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]: ...
    def load_messages(self, session: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]: ...
    def delete_session(self, session: dict[str, Any]) -> int: ...


@dataclass(slots=True)
class SQLiteConversationRepository:
    """屏蔽 SQLite 查询参数，避免它们散落在 Qt 事件代码中。"""

    store: SQLiteConversationStore = field(default_factory=lambda: sqlite_memory)

    def list_chat_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.get_recent_sessions(
            user_id,
            limit=limit,
            workspace_path="",
            include_unbound=False,
        )

    def list_work_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.get_recent_sessions(
            user_id,
            limit=limit,
            workspace_path=None,
            bound_only=True,
        )

    def load_messages(self, session: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
        return self.store.get_recent_messages(
            str(session.get("user_id") or "guest_user"),
            limit=limit,
            session_id=session.get("session_id"),
            workspace_path=session.get("workspace_path") or None,
        )

    def delete_session(self, session: dict[str, Any]) -> int:
        return self.store.delete_session(
            str(session.get("user_id") or "guest_user"),
            str(session.get("session_id") or ""),
            workspace_path=session.get("workspace_path") or None,
        )
