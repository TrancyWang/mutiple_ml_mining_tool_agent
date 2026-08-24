"""项目用例：项目注册、枚举和切换。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from academic_agent.models import AgentMode, ApplicationState
from academic_agent.repositories.projects import ProjectRepository


@dataclass(slots=True)
class ProjectController:
    state: ApplicationState
    repository: ProjectRepository

    @property
    def current_root(self) -> Path:
        return self.repository.current_root

    def list_projects(self, limit: int = 30) -> list[dict[str, Any]]:
        if self.state.auth_session is None:
            return []
        if self.state.mode is AgentMode.WORK:
            self.repository.register_current(self.state.user_id)
        return self.repository.list_projects(self.state.user_id, limit)

    def activate(self, target: str | Path) -> Path:
        path = self.repository.activate(self.state.user_id, target)
        self.state.reset_conversation()
        return path

    def delete(self, project: dict[str, Any]) -> dict[str, Any]:
        target = project.get("workspace_path")
        if not target:
            raise ValueError("项目路径为空，无法删除")
        result = self.repository.delete(self.state.user_id, target)
        from academic_agent.agent.session import session_store

        for session_id in result.get("session_ids", []):
            session_store.clear(str(session_id))
            try:
                from academic_agent.agent.memory.manager import memory_manager

                memory_manager.delete_session(self.state.user_id, str(session_id))
            except Exception:
                pass
        self.state.reset_conversation()
        return result
