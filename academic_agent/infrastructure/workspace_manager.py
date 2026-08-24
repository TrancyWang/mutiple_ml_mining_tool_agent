"""当前项目工作区：文件列表、检索、读取和安全生成。"""

from __future__ import annotations

import fnmatch
import os
import difflib
import uuid
import json
from pathlib import Path
from typing import Any

from academic_agent.infrastructure.runtime_paths import default_workspace_root, output_root


class WorkspaceManager:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or default_workspace_root()).resolve()
        self.ignored_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "milvus_data", "chroma_db", ".academic_agent"}
        self.ignored_files = {".env", ".env.*", "*.key", "*.pem", "*.secret"}
        self.max_read_bytes = 2 * 1024 * 1024
        self.permission_mode = "read_only"
        self.pending_operations: dict[str, dict[str, Any]] = {}
        self._gitignore_patterns: list[str] = []
        self._load_gitignore()
        self._load_permissions()

    def set_root(self, root: str | Path) -> Path:
        candidate = Path(root).expanduser().resolve()
        if not candidate.is_dir():
            raise ValueError(f"工作区目录不存在：{candidate}")
        self.root = candidate
        self._load_gitignore()
        self._load_permissions()
        return self.root

    def _permission_path(self) -> Path:
        return self.root / ".academic_agent" / "permissions.json"

    def _load_permissions(self) -> None:
        path = self._permission_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            mode = payload.get("permission_mode", "read_only")
            self.permission_mode = mode if mode in {"read_only", "accept_edits", "accept_all"} else "read_only"
        except Exception:
            self.permission_mode = "read_only"

    def set_permission_mode(self, mode: str) -> str:
        if mode not in {"read_only", "accept_edits", "accept_all"}:
            raise ValueError(f"不支持的权限模式：{mode}")
        self.permission_mode = mode
        path = self._permission_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"permission_mode": mode}, ensure_ascii=False, indent=2), encoding="utf-8")
        return mode

    def _auto_allowed(self, operation: str) -> bool:
        return self.permission_mode == "accept_all" or (
            self.permission_mode == "accept_edits" and operation in {"generate", "edit"}
        )

    def _load_gitignore(self) -> None:
        path = self.root / ".gitignore"
        if path.is_file():
            self._gitignore_patterns = [
                line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            self._gitignore_patterns = []

    def _safe_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("文件路径必须位于当前项目工作区内")
        return candidate

    @staticmethod
    def _safe_output_path(path: str | Path) -> Path:
        """将生成型产物限制在应用同级 output 目录。"""
        root = output_root()
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            candidate = resolved if resolved.is_relative_to(root) else root / candidate.name
        else:
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("输出文件路径必须位于应用同级 output 目录内")
        return candidate

    def _ignored(self, path: Path) -> bool:
        if any(part in self.ignored_dirs for part in path.relative_to(self.root).parts):
            return True
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in self.ignored_files):
            return True
        relative = str(path.relative_to(self.root))
        return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern.rstrip("/"))
                   for pattern in self._gitignore_patterns if not pattern.startswith("!"))

    def list_files(self, pattern: str = "*") -> list[str]:
        result = []
        for path in self.root.rglob("*"):
            if path.is_file() and not self._ignored(path) and fnmatch.fnmatch(path.name, pattern):
                result.append(str(path.relative_to(self.root)))
        return sorted(result)

    def search(self, query: str, pattern: str = "*") -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        results = []
        for relative in self.list_files(pattern):
            path = self.root / relative
            if path.stat().st_size > self.max_read_bytes:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    results.append({"file": relative, "line": line_number, "text": line[:500]})
                    if len(results) >= 100:
                        return results
        return results

    def glob(self, pattern: str) -> list[str]:
        """按相对路径或文件名查找当前工作区文件。"""
        pattern = pattern.strip() or "*"
        return [item for item in self.list_files("*") if fnmatch.fnmatch(item, pattern) or fnmatch.fnmatch(Path(item).name, pattern)]

    def grep(self, query: str, pattern: str = "*") -> list[dict[str, Any]]:
        """内容检索别名，返回文件、行号和上下文。"""
        return self.search(query, pattern)

    def read(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
        target = self._safe_path(path)
        if self._ignored(target):
            raise ValueError("该文件属于受保护文件，不能读取")
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        if target.stat().st_size > self.max_read_bytes:
            raise ValueError("文件过大，请先使用文件检索或指定更小的文件")
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line)) if end_line else len(lines)
        return {"file": str(target.relative_to(self.root)), "start_line": start, "end_line": end, "content": "\n".join(lines[start - 1:end])}

    @staticmethod
    def _content_text(content: str | bytes) -> str:
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content)

    @staticmethod
    def _write_content(target: Path, content: str | bytes) -> None:
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(str(content), encoding="utf-8")

    def generate(self, path: str, content: str | bytes, overwrite: bool = False) -> dict[str, Any]:
        target = self._safe_path(path)
        if self._ignored(target):
            raise ValueError("不能生成或修改受保护文件")
        if target.exists() and not overwrite:
            raise FileExistsError("文件已存在；如需覆盖，请明确传入 overwrite=true")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write_content(target, content)
        return {"file": str(target.relative_to(self.root)), "bytes": target.stat().st_size, "success": True}

    def _preview_operation(
        self, operation: str, path: str, *, scope: str = "workspace", **payload: Any
    ) -> dict[str, Any]:
        target = self._safe_output_path(path) if scope == "output" else self._safe_path(path)
        if scope == "workspace" and self._ignored(target):
            raise ValueError("不能操作受保护文件")
        display_path = str(target) if scope == "output" else str(target.relative_to(self.root))
        operation_id = uuid.uuid4().hex[:12]
        old_content = target.read_text(encoding="utf-8", errors="ignore") if target.is_file() else ""
        new_content = payload.get("content", "")
        if isinstance(new_content, bytes):
            diff = (
                f"将生成二进制文件：{display_path}\n"
                f"文件大小：{len(new_content)} bytes\n"
                "文件内容将在文件生成后通过右侧文件面板查看。"
            )
        else:
            old_text = old_content
            new_text = self._content_text(new_content)
            diff = "".join(difflib.unified_diff(
                old_text.splitlines(True), new_text.splitlines(True),
                fromfile=display_path,
                tofile=display_path,
            ))
        self.pending_operations[operation_id] = {
            "operation": operation, "path": str(target), "content": new_content, "scope": scope,
        }
        return {
            "success": False,
            "requires_confirmation": True,
            "operation_id": operation_id,
            "operation": operation,
            "file": display_path,
            "preview": diff[:12000] or "将创建空文件或没有文本差异。",
            "message": f"已生成文件操作预览，请用户确认后调用 confirm_workspace_operation('{operation_id}')。",
        }

    def edit(self, path: str, old_text: str, new_text: str, confirm: bool = False) -> dict[str, Any]:
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        current = target.read_text(encoding="utf-8", errors="ignore")
        if old_text not in current:
            raise ValueError("未找到要替换的原文，文件未修改")
        new_content = current.replace(old_text, new_text, 1)
        if not confirm and not self._auto_allowed("edit"):
            return self._preview_operation("edit", path, content=new_content)
        target.write_text(new_content, encoding="utf-8")
        return {"success": True, "file": str(target.relative_to(self.root)), "operation": "edit", "auto_approved": not confirm}

    def prepare_generate(self, path: str, content: str | bytes, overwrite: bool = False) -> dict[str, Any]:
        target = self._safe_path(path)
        if target.exists() and not overwrite:
            raise FileExistsError("文件已存在；如需覆盖，请明确传入 overwrite=true")
        if self._auto_allowed("generate"):
            target.parent.mkdir(parents=True, exist_ok=True)
            self._write_content(target, content)
            return {"success": True, "file": str(target.relative_to(self.root)), "operation": "generate", "auto_approved": True}
        return self._preview_operation("generate", path, content=content)

    def prepare_output_generate(
        self, path: str, content: str | bytes, overwrite: bool = False
    ) -> dict[str, Any]:
        """准备写入统一 output 目录，沿用工作区权限与确认链接。"""
        target = self._safe_output_path(path)
        if target.exists() and not overwrite:
            raise FileExistsError("输出文件已存在；如需覆盖，请明确传入 overwrite=true")
        if self._auto_allowed("generate"):
            target.parent.mkdir(parents=True, exist_ok=True)
            self._write_content(target, content)
            return {
                "success": True,
                "file": str(target),
                "operation": "generate",
                "auto_approved": True,
            }
        return self._preview_operation("generate", path, scope="output", content=content)

    def confirm(self, operation_id: str) -> dict[str, Any]:
        operation = self.pending_operations.pop(operation_id, None)
        if not operation:
            raise ValueError("操作不存在、已确认或已过期")
        scope = operation.get("scope", "workspace")
        target = (
            self._safe_output_path(operation["path"])
            if scope == "output"
            else self._safe_path(operation["path"])
        )
        if operation["operation"] == "delete":
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._write_content(target, operation["content"])
        display_path = str(target) if scope == "output" else str(target.relative_to(self.root))
        return {"success": True, "file": display_path, "operation": operation["operation"]}

    def delete(self, path: str, confirm: bool = False) -> dict[str, Any]:
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        if not confirm and not self._auto_allowed("delete"):
            return self._preview_operation("delete", path, content="")
        target.unlink()
        return {"success": True, "file": str(target.relative_to(self.root)), "operation": "delete", "auto_approved": not confirm}


workspace_manager = WorkspaceManager()
