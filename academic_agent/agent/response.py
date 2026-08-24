"""Agent 响应后处理。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from academic_agent.infrastructure.workspace_manager import workspace_manager


def strip_confirmation_marker(content: str) -> str:
    return re.sub(r"\s*\[\[CONFIRM:[a-zA-Z0-9_-]+\]\]", "", str(content)).strip()


def attach_confirmation_link(final_response: Any, pending_before: set[str]) -> str | None:
    if not isinstance(final_response, list) or not final_response:
        return None
    created = [
        (operation_id, operation)
        for operation_id, operation in workspace_manager.pending_operations.items()
        if operation_id not in pending_before
    ]
    if not created:
        return None
    operation_id, operation = created[-1]
    file_name = Path(str(operation.get("path", "项目文件"))).name
    operation_name = operation.get("operation", "generate")
    final_response[-1]["content"] = (
        f"已生成文件操作预览：{file_name}（操作类型：{operation_name}，请点击下方确认链接）\n"
        f"[[CONFIRM:{operation_id}]]"
    )
    return operation_id


def assistant_content(response: Any) -> str:
    if not isinstance(response, list) or not response:
        return ""
    return strip_confirmation_marker(response[-1].get("content", ""))
