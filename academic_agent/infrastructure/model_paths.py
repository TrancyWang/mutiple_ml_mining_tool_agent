"""本地模型目录配置。

模型不必随 Qt 客户端一起发布。用户可以把模型放在任意目录，目录结构
建议为 ``<模型根目录>/bge-cn`` 和可选的情感模型目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from academic_agent.infrastructure.runtime_paths import resource_root

MODEL_ROOT_ENV = "PRETRAINED_MODELS_DIR"


def normalize_model_root(value: str | Path | None) -> Path | None:
    """把用户输入规范化为包含 ``bge-cn`` 的模型根目录。"""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if path.name == "bge-cn":
        return path.parent
    return path


def default_model_root() -> Path:
    """返回源码运行或冻结运行时的默认模型目录。"""
    configured = normalize_model_root(os.getenv(MODEL_ROOT_ENV))
    if configured is not None:
        return configured
    if getattr(sys, "frozen", False):
        return (resource_root() / "video_text_mutiplemodal_agent" / "pretrain_models").resolve()
    return (
        resource_root().parent
        / "video_text_mutiplemodal_agent"
        / "pretrain_models"
    ).resolve()


def model_components(root: str | Path | None = None) -> dict[str, Path | bool]:
    """返回模型目录状态，用于界面提示和启动诊断。"""
    model_root = normalize_model_root(root) or default_model_root()
    bge = model_root / "bge-cn"
    sentiment = model_root / "xuyuan-trial-sentiment-bert-chinese"
    return {
        "root": model_root,
        "bge": bge,
        "sentiment": sentiment,
        "has_bge": bge.is_dir(),
        "has_sentiment": sentiment.is_dir(),
    }


def apply_model_root(value: str | Path | None) -> Path | None:
    """设置当前进程使用的模型根目录；传入空值时清除覆盖。"""
    root = normalize_model_root(value)
    if root is None:
        os.environ.pop(MODEL_ROOT_ENV, None)
    else:
        os.environ[MODEL_ROOT_ENV] = str(root)
    return root
