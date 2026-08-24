"""项目工作区存储端口及默认适配器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from academic_agent.infrastructure.workspace_manager import WorkspaceManager, workspace_manager
from academic_agent.infrastructure.runtime_paths import app_data_root
from academic_agent.repositories.sqlite_store import SQLiteConversationStore, sqlite_memory


class ProjectRepository(Protocol):
    @property
    def current_root(self) -> Path: ...
    def list_projects(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]: ...
    def register_current(self, user_id: str) -> None: ...
    def activate(self, user_id: str, target: str | Path) -> Path: ...
    def delete(self, user_id: str, target: str | Path) -> dict[str, Any]: ...


@dataclass(slots=True)
class WorkspaceProjectRepository:
    manager: WorkspaceManager = field(default_factory=lambda: workspace_manager)
    store: SQLiteConversationStore = field(default_factory=lambda: sqlite_memory)

    @property
    def current_root(self) -> Path:
        return self.manager.root

    def register_current(self, user_id: str) -> None:
        self.store.register_workspace(user_id, str(self.current_root))

    def list_projects(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        projects = self.store.get_recent_projects(user_id, limit=limit)
        current_path = str(self.current_root)
        if current_path not in {item["workspace_path"] for item in projects}:
            projects.insert(0, {
                "workspace_path": current_path,
                "name": self.current_root.name,
                "session_count": 0,
            })
        return projects

    def activate(self, user_id: str, target: str | Path) -> Path:
        path = Path(target).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"项目目录不存在：{path}")
        self.manager.set_root(path)
        self.store.register_workspace(user_id, str(path))
        return path

    def delete(self, user_id: str, target: str | Path) -> dict[str, Any]:
        """Remove a project from Agent metadata without touching its physical directory."""
        path = Path(target).expanduser().resolve()

        was_current = path == self.current_root
        if was_current:
            fallback = (Path.home() / "Documents" / "AcademicAgent" / "Workspace").resolve()
            fallback.mkdir(parents=True, exist_ok=True)
            self.manager.set_root(fallback)
            self.store.register_workspace(user_id, fallback)

        deleted = self.store.delete_workspace(user_id, path)
        return {
            "workspace_path": str(path),
            "was_current": was_current,
            "physical_directory_deleted": False,
            **deleted,
        }
