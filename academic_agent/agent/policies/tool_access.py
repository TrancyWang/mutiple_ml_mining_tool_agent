"""Agent 工具权限策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from academic_agent.agent.types import ExecutionScope
from academic_agent.agent.tooling.registry import ToolSpec
from academic_agent.agent.session import session_store
from academic_agent.infrastructure.workspace_manager import workspace_manager


class PolicyViolation(PermissionError):
    pass


@dataclass(slots=True)
class AgentPolicy:
    """权限在 Tool 层执行，不能只依赖系统提示词。"""

    def authorize(
        self,
        spec: ToolSpec,
        params: dict[str, Any],
        scope: ExecutionScope | None,
    ) -> None:
        if scope is None:
            return
        if scope.request.chat_only:
            raise PolicyViolation("Chat 模式不允许调用工具，请切换到 Work 模式。")
        if spec.category == "项目工作区" and not scope.request.workspace_path:
            raise PolicyViolation("当前任务没有绑定项目工作区。")
        if spec.name == "load_data" and params.get("file_path"):
            from pathlib import Path

            target = Path(str(params["file_path"])).expanduser()
            if not target.is_absolute():
                target = workspace_manager.root / target
            target = target.resolve()
            session = session_store.get(scope.request.session_id)
            uploaded = {Path(path).expanduser().resolve() for path in session.uploaded_files}
            in_workspace = target.is_relative_to(workspace_manager.root)
            if target not in uploaded and not in_workspace:
                raise PolicyViolation("只能加载当前工作区文件或用户主动上传的文件。")
        if spec.name == "execute_python_analysis" and scope.plan.route.value == "chat":
            raise PolicyViolation("普通聊天不能执行临时代码。")
