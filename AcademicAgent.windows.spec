# -*- mode: python ; coding: utf-8 -*-
"""Academic Agent Windows onedir 构建配置。

Windows 发布包不携带真实 .env 和本地大模型。用户可以在应用目录旁边配置
.env，或者在界面中填写自己的模型地址/API Key。

video_text_mutiplemodal_agent 是本地开发时的外部源项目，不在当前 Git 仓库中；
因此 Windows 包使用当前仓库内的算法和工具模块完成启动及通用功能。若发布时
需要源项目的专用模型算法，可在构建机上额外提供 VIDEO_AGENT_SOURCE_ROOT，或
将源项目作为构建依赖加入 CI，不能依赖当前项目的父目录结构。
"""

from pathlib import Path
import os

from PyInstaller.building.api import COLLECT
from PyInstaller.building.build_main import Analysis, EXE, PYZ
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


PROJECT_ROOT = Path(SPECPATH).resolve()

datas = [
    (str(PROJECT_ROOT / "resources" / "academic_agent_icon.png"), "resources"),
    (str(PROJECT_ROOT / "resources" / "academic_agent_icon.svg"), "resources"),
    (str(PROJECT_ROOT / "resources" / "data"), "resources/data"),
    (str(PROJECT_ROOT / "resources" / "dicts"), "resources/dicts"),
    (str(PROJECT_ROOT / ".env.example"), "."),
]

binaries = []
hiddenimports = [
    "academic_agent",
    "academic_agent.algorithms",
    "academic_agent.application",
    "academic_agent.controllers",
    "academic_agent.infrastructure",
    "academic_agent.repositories",
    "academic_agent.tools",
    "academic_agent.views",
    "PIL._tkinter_finder",
]


def _runtime_module(name: str) -> bool:
    excluded_parts = {
        ".tests",
        ".test",
        ".testing",
        ".examples",
        ".gui",
        ".command_line",
    }
    return not any(part in name for part in excluded_parts)


# qwen-agent、Chroma 和可视化库包含动态导入，显式收集运行时模块。
for package_name in ("qwen_agent", "chromadb"):
    try:
        hiddenimports += collect_submodules(
            package_name,
            filter=_runtime_module,
            on_error="ignore",
        )
        datas += collect_data_files(package_name)
        binaries += collect_dynamic_libs(package_name)
        datas += copy_metadata(package_name)
    except Exception:
        # 可选依赖缺失时仍允许生成基础客户端包。
        pass

for package_name in ("torch", "onnxruntime", "tokenizers"):
    try:
        datas += collect_data_files(package_name)
        binaries += collect_dynamic_libs(package_name)
        datas += copy_metadata(package_name)
    except Exception:
        pass


hiddenimports = sorted(set(hiddenimports))
icon_path = PROJECT_ROOT / "resources" / "academic_agent_icon.ico"

a = Analysis(
    [str(PROJECT_ROOT / "qt_client.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AcademicAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(icon_path) if icon_path.is_file() else None,
)

COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="AcademicAgent",
)
