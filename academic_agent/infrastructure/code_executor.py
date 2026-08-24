"""受控的临时 Python 代码执行器。

代码只在当前工作区下的临时目录运行；输入文件需要显式声明。工作区输入的
产物写入工作区 output，用户上传输入的产物写入应用同级 output。
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from academic_agent.infrastructure.workspace_manager import workspace_manager
from academic_agent.infrastructure.analysis_artifacts import artifact_output_root


BLOCKED_MODULES = {
    "subprocess", "socket", "requests", "urllib", "http", "ftplib",
    "paramiko", "ctypes", "multiprocessing", "signal", "os", "sys",
}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}
MAX_SCRIPT_BYTES = 200_000
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


def _validate_script(script: str) -> None:
    if not script.strip():
        raise ValueError("临时代码不能为空")
    if len(script.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise ValueError(f"临时代码不能超过 {MAX_SCRIPT_BYTES // 1000} KB")
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        raise ValueError(f"Python 代码语法错误：{exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [(node.module or "").split(".")[0]]
        else:
            modules = []
        blocked = sorted(set(modules) & BLOCKED_MODULES)
        if blocked:
            raise ValueError(f"临时代码禁止导入高风险模块：{', '.join(blocked)}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                raise ValueError(f"临时代码禁止调用：{node.func.id}()")


def _copy_declared_inputs(
    temp_root: Path,
    input_files: list[str],
    allowed_uploaded_files: list[str] | None = None,
) -> list[str]:
    input_root = temp_root / "inputs"
    copied: list[str] = []
    allowed = {
        Path(path).expanduser().resolve() for path in (allowed_uploaded_files or [])
    }
    for declared in input_files:
        candidate = Path(declared).expanduser()
        if candidate.is_absolute():
            source = candidate.resolve()
            if source not in allowed:
                raise ValueError(f"该工作区外文件没有经过用户上传授权：{source.name}")
            destination_name = source.name
        else:
            source = workspace_manager._safe_path(declared)
            if workspace_manager._ignored(source):
                raise ValueError(f"不能使用受保护的输入文件：{declared}")
            destination_name = str(Path(declared))
        if not source.is_file():
            raise FileNotFoundError(f"输入文件不存在：{declared}")
        destination = input_root / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(Path("inputs") / destination_name))
    return copied


def _collect_artifacts(temp_root: Path, run_id: str, artifact_root: Path) -> list[str]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    for source in temp_root.rglob("*"):
        if not source.is_file() or source.name == "script.py" or "inputs" in source.parts:
            continue
        if source.stat().st_size > MAX_ARTIFACT_BYTES:
            continue
        target = artifact_root / f"code_run_{run_id}_{source.name}"
        shutil.copy2(source, target)
        artifacts.append(str(target.resolve()))
    return artifacts


def execute_python_in_temp_workspace(
    script: str,
    input_files: list[str] | None = None,
    allowed_uploaded_files: list[str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """在临时工作区运行 Python，并返回日志和产物路径。"""
    _validate_script(script)
    timeout = max(1, min(int(timeout), 180))
    run_id = uuid.uuid4().hex[:12]
    allowed_uploads = {
        Path(path).expanduser().resolve() for path in (allowed_uploaded_files or [])
    }
    declared_absolute = {
        Path(path).expanduser().resolve()
        for path in (input_files or [])
        if Path(path).expanduser().is_absolute()
    }
    source_scope = "upload" if declared_absolute & allowed_uploads else "workspace"
    artifact_root = artifact_output_root(source_scope=source_scope)
    temp_parent = workspace_manager.root / ".academic_agent" / "tmp_runs"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"run_{run_id}_", dir=temp_parent))
    script_path = temp_root / "script.py"
    script_path.write_text(script, encoding="utf-8")
    try:
        copied_inputs = _copy_declared_inputs(
            temp_root,
            input_files or [],
            allowed_uploaded_files=allowed_uploaded_files,
        )
        env = os.environ.copy()
        env.update({
            "PYTHONNOUSERSITE": "1",
            "MPLCONFIGDIR": str(temp_root / "matplotlib_cache"),
            "ACADEMIC_AGENT_TEMP_WORKSPACE": str(temp_root),
        })
        completed = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            cwd=str(temp_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        artifacts = _collect_artifacts(temp_root, run_id, artifact_root)
        return {
            "success": completed.returncode == 0,
            "run_id": run_id,
            "return_code": completed.returncode,
            "stdout": (completed.stdout or "")[-12000:],
            "stderr": (completed.stderr or "")[-12000:],
            "input_files": copied_inputs,
            "artifacts": artifacts,
            "output_directory": str(artifact_root),
            "message": (
                f"临时代码执行完成，产物已保存到 {artifact_root}。"
                if completed.returncode == 0 else "临时代码执行失败。"
            ),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "run_id": run_id,
            "error": f"临时代码执行超过 {timeout} 秒，已终止。",
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
