"""短期、长期与语义记忆的统一协调器。"""

from __future__ import annotations

import threading
from typing import Any


class MemoryCoordinator:
    """为 Runtime 提供稳定接口，隐藏 SQLite/ES/Milvus/Chroma 细节。"""

    def __init__(self, manager: Any | None = None) -> None:
        self._manager = manager

    @property
    def manager(self):
        if self._manager is None:
            from academic_agent.agent.memory.manager import memory_manager

            self._manager = memory_manager
        return self._manager

    def recall(
        self,
        user_id: str,
        query: str,
        session_id: str,
        workspace_path: str | None,
    ) -> str:
        if not query:
            return ""
        return self.manager.build_context_with_memory(
            user_id=user_id,
            current_query=query,
            session_id=session_id,
            workspace_path=workspace_path,
            use_short_term=True,
            use_long_term=True,
        )

    def commit(
        self,
        user_id: str,
        user_text: str,
        assistant_text: str,
        session_id: str,
        workspace_path: str | None,
    ) -> None:
        if not user_text or not assistant_text:
            return
        self.manager.add_to_short_term(
            user_id,
            "user",
            user_text,
            session_id=session_id,
            workspace_path=workspace_path,
        )
        self.manager.add_to_short_term(
            user_id,
            "assistant",
            assistant_text,
            session_id=session_id,
            workspace_path=workspace_path,
        )
        threading.Thread(
            target=self.manager.process_and_store_memory,
            kwargs={
                "user_id": user_id,
                "user_text": user_text,
                "assistant_response": assistant_text,
                "session_id": session_id,
                "workspace_path": workspace_path,
            },
            daemon=True,
            name=f"agent-memory-{session_id[:8]}",
        ).start()
