"""会话用例：新建、筛选和恢复会话。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from academic_agent.models import AgentMode, ApplicationState
from academic_agent.repositories.conversations import ConversationRepository


@dataclass(slots=True)
class SessionController:
    state: ApplicationState
    repository: ConversationRepository

    def new_session(self) -> str:
        self.state.reset_conversation()
        return self.state.session_id

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.state.auth_session is None:
            return []
        if self.state.mode is AgentMode.CHAT:
            return self.repository.list_chat_sessions(self.state.user_id, limit)
        return self.repository.list_work_sessions(self.state.user_id, limit)

    def load_session(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        messages = self.repository.load_messages(session, limit=200)
        if not messages:
            return []
        self.state.session_id = str(session.get("session_id") or self.state.session_id)
        self.state.messages = [
            {
                "role": message["role"],
                "content": message["content"],
                "_image_paths": [],
                "_show_images": False,
            }
            for message in messages
            if message.get("role") in {"user", "assistant", "system"}
        ]
        self.state.current_assistant = ""
        self.state.confirmed_operations.clear()
        return self.state.messages

    def delete_session(self, session: dict[str, Any]) -> int:
        session_id = str(session.get("session_id") or "")
        if not session_id:
            raise ValueError("会话标识为空，无法删除")
        deleted = self.repository.delete_session(session)
        from academic_agent.agent.session import session_store

        session_store.clear(session_id)
        try:
            from academic_agent.agent.memory.manager import memory_manager

            memory_manager.delete_session(self.state.user_id, session_id)
        except Exception:
            pass
        if session_id == self.state.session_id:
            self.state.reset_conversation()
        return deleted
