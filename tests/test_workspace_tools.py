from __future__ import annotations

import pytest

from academic_agent.agent.executor import tool_executor  # noqa: F401 - registers built-ins
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.tooling.handlers import project
from academic_agent.agent.tooling.registry import tool_registry
from academic_agent.agent.types import AgentRequest
from academic_agent.infrastructure.workspace_manager import WorkspaceManager


def test_write_project_file_previews_then_writes_inside_workspace(tmp_path, monkeypatch):
    manager = WorkspaceManager(tmp_path)
    monkeypatch.setattr(project, "workspace_manager", manager)

    preview = project.write_project_file(
        path="src/main.py",
        content="print('hello')\n",
    )

    assert preview["requires_confirmation"] is True
    assert not (tmp_path / "src" / "main.py").exists()

    result = manager.confirm(preview["operation_id"])

    assert result["success"] is True
    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_write_project_file_rejects_protected_files(tmp_path, monkeypatch):
    manager = WorkspaceManager(tmp_path)
    monkeypatch.setattr(project, "workspace_manager", manager)

    with pytest.raises(ValueError, match="保护"):
        project.write_project_file(path=".env", content="TOKEN=secret")


def test_code_request_routes_to_workspace_and_exposes_write_tool():
    request = AgentRequest(
        messages=[{"role": "user", "content": "请在工作区写代码"}],
        user_id="user",
        session_id="session",
        workspace_path="/tmp/project",
    )

    plan = TaskPlanner().create_plan(request)

    assert plan.route.value == "project"
    assert "write_project_file" in plan.available_tools
    assert tool_registry.get("write_project_file") is not None
    assert plan.requires_confirmation is True
