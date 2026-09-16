"""项目工作区和受控文件操作工具处理器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from academic_agent.infrastructure.workspace_manager import workspace_manager


def list_project_files(**kwargs: Any) -> dict[str, Any]:
    return {
        "success": True, "workspace": str(workspace_manager.root),
        "permission_mode": workspace_manager.permission_mode,
        "files": workspace_manager.list_files(kwargs.get("pattern", "*")),
    }


def search_project_files(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "workspace": str(workspace_manager.root), "results": workspace_manager.search(kwargs["query"], kwargs.get("pattern", "*"))}


def glob_project_files(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "workspace": str(workspace_manager.root), "files": workspace_manager.glob(kwargs.get("pattern", "*"))}


def grep_project_files(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "workspace": str(workspace_manager.root), "results": workspace_manager.grep(kwargs["query"], kwargs.get("pattern", "*"))}


def read_project_file(**kwargs: Any) -> dict[str, Any]:
    return {"success": True, "workspace": str(workspace_manager.root), **workspace_manager.read(kwargs["path"], kwargs.get("start_line", 1), kwargs.get("end_line"))}


def generate_project_file(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.infrastructure.document_generator import generate_document

    path = str(kwargs["path"])
    generated = generate_document(path, kwargs["content"])
    requested = Path(path).expanduser()
    if requested.is_absolute():
        relative = Path(requested.name)
    else:
        safe_parts = [part for part in requested.parts if part not in {"", ".", "..", "output"}]
        relative = Path(*safe_parts) if safe_parts else Path("generated.txt")
    result = workspace_manager.prepare_generate(
        str(Path("output") / relative), generated, bool(kwargs.get("overwrite", False))
    )
    if result.get("success") and result.get("file"):
        result["artifacts"] = [str((workspace_manager.root / str(result["file"])).resolve())]
    result["format"] = path.rsplit(".", 1)[-1].lower() if "." in path else "text"
    return result


def write_project_file(**kwargs: Any) -> dict[str, Any]:
    """Create or overwrite a source/text file directly in the current workspace.

    Unlike ``generate_project_file``, this tool is for project source files and
    does not redirect the path into ``output``. WorkspaceManager still enforces
    the workspace boundary, protected-file rules and confirmation workflow.
    """

    path = str(kwargs["path"])
    content = kwargs.get("content", "")
    if not isinstance(content, str):
        raise TypeError("代码文件内容必须是文本字符串")
    result = workspace_manager.prepare_generate(
        path,
        content,
        bool(kwargs.get("overwrite", False)),
    )
    result["format"] = Path(path).suffix.lower().lstrip(".") or "text"
    result["workspace"] = str(workspace_manager.root)
    return result


def execute_python_analysis(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.infrastructure.code_executor import execute_python_in_temp_workspace

    return execute_python_in_temp_workspace(
        script=kwargs["script"], input_files=kwargs.get("input_files", []),
        allowed_uploaded_files=kwargs.get("_allowed_uploaded_files", []),
        timeout=int(kwargs.get("timeout", 60)),
    )


def edit_project_file(**kwargs: Any) -> dict[str, Any]:
    return workspace_manager.edit(kwargs["path"], kwargs["old_text"], kwargs["new_text"], bool(kwargs.get("confirm", False)))


def delete_project_file(**kwargs: Any) -> dict[str, Any]:
    return workspace_manager.delete(kwargs["path"], bool(kwargs.get("confirm", False)))


def confirm_workspace_operation(**kwargs: Any) -> dict[str, Any]:
    return workspace_manager.confirm(kwargs["operation_id"])
