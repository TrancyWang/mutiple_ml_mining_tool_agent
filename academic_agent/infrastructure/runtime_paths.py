"""运行时路径，兼容源码运行和 PyInstaller 冻结运行。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """返回随应用发布的只读资源目录。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parents[2]


def executable_dir() -> Path:
    return Path(sys.executable).resolve().parent if is_frozen() else resource_root()


def bundled_resources_root() -> Path:
    """返回词典、技能等只读随包资源目录。"""
    return resource_root() / "resources"


def app_data_root() -> Path:
    """返回用户可写的数据目录，避免向 .app 或 _MEIPASS 写文件。"""
    configured = os.getenv("ACADEMIC_AGENT_DATA_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif is_frozen():
        root = Path.home() / "Documents" / "AcademicAgent"
    else:
        root = resource_root() / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def default_workspace_root() -> Path:
    return app_data_root() if is_frozen() else resource_root()


def output_root() -> Path:
    """返回统一产物目录；打包后优先位于 ``AcademicAgent.app`` 同级。"""
    configured = os.getenv("ACADEMIC_AGENT_OUTPUT_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif is_frozen():
        executable = Path(sys.executable).resolve()
        app_bundle = next(
            (parent for parent in executable.parents if parent.suffix.lower() == ".app"),
            None,
        )
        root = (app_bundle.parent if app_bundle else executable.parent) / "output"
    else:
        root = resource_root() / "output"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = app_data_root() / "output"
        root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def configure_qwen_agent_workspace() -> Path:
    """把 Qwen-Agent 内部文件记忆目录切换到可写位置。

    qwen-agent 默认使用相对路径 ``workspace``，而 macOS 冻结应用的当前
    目录可能是只读的 .app 资源目录。必须在导入 qwen_agent 之前设置该环境变量。
    """
    configured = os.getenv("QWEN_AGENT_DEFAULT_WORKSPACE", "").strip()
    candidate = Path(configured).expanduser() if configured else Path()
    if not candidate.is_absolute():
        candidate = app_data_root() / ".academic_agent" / "qwen_workspace"

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 用户配置的目录可能是只读路径，自动回退到应用数据目录。
        candidate = app_data_root() / ".academic_agent" / "qwen_workspace"
        candidate.mkdir(parents=True, exist_ok=True)

    resolved = candidate.resolve()
    os.environ["QWEN_AGENT_DEFAULT_WORKSPACE"] = str(resolved)
    return resolved


def env_candidates() -> list[Path]:
    """返回管理员配置搜索位置，优先使用用户目录中的可更新配置。"""
    candidates = [app_data_root() / ".env", resource_root() / ".env"]
    if is_frozen():
        binary_dir = executable_dir()
        candidates.extend([
            binary_dir / ".env",
            binary_dir.parent / ".env",
            binary_dir.parent.parent / ".env",  # .app 根目录
            binary_dir.parent.parent.parent / ".env",  # .app 所在目录
        ])
    return list(dict.fromkeys(path.resolve() for path in candidates))
