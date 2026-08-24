# -*- mode: python ; coding: utf-8 -*-
"""Academic Agent macOS onedir/.app 构建配置。

默认不携带本地大模型；构建时设置 INCLUDE_LOCAL_MODELS=1 才会把
video_text_mutiplemodal_agent/pretrain_models 一起复制进应用包。
"""

from pathlib import Path
import os

from PyInstaller.building.build_main import Analysis, EXE, PYZ
from PyInstaller.building.api import COLLECT
from PyInstaller.building.osx import BUNDLE
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


PROJECT_ROOT = Path(SPECPATH).resolve()
SOURCE_ROOT = Path(
    os.getenv(
        "VIDEO_AGENT_SOURCE_ROOT",
        str(PROJECT_ROOT.parent / "video_text_mutiplemodal_agent"),
    )
).expanduser().resolve()
STAGE3_ROOT = SOURCE_ROOT / "src_codes" / "main_process_agent" / "text_processor_subagent_stage_3"

if not STAGE3_ROOT.is_dir():
    raise SystemExit(
        "找不到源项目的 text_processor_subagent_stage_3："
        f" {STAGE3_ROOT}\n"
        "请设置 VIDEO_AGENT_SOURCE_ROOT 指向 video_text_mutiplemodal_agent。"
    )


datas = [
    (str(PROJECT_ROOT / "resources" / "academic_agent_icon.png"), "resources"),
    (str(PROJECT_ROOT / "resources" / "academic_agent_icon.icns"), "resources"),
    (str(PROJECT_ROOT / "resources" / "academic_agent_icon.svg"), "resources"),
    (str(PROJECT_ROOT / "resources" / "data"), "resources/data"),
    (str(PROJECT_ROOT / "resources" / "dicts"), "resources/dicts"),
    (
        str(PROJECT_ROOT / "academic_agent" / "algorithms"),
        "academic_agent/algorithms",
    ),
    (
        str(STAGE3_ROOT),
        "video_text_mutiplemodal_agent/src_codes/main_process_agent/text_processor_subagent_stage_3",
    ),
    (
        str(SOURCE_ROOT / "dicts"),
        "video_text_mutiplemodal_agent/dicts",
    ),
    (str(PROJECT_ROOT / ".env.example"), "."),
]

# 管理员 trancy 登录必须能够在单独移动 .app 后继续使用构建时配置。
# 真实 .env 若存在则封装进应用资源；发布该应用等同于发布其中的 API Key。
root_env = PROJECT_ROOT / ".env"
if root_env.is_file():
    datas.append((str(root_env), "."))

include_models = os.getenv("INCLUDE_LOCAL_MODELS", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
model_root = SOURCE_ROOT / "pretrain_models"
if include_models:
    if not model_root.is_dir():
        raise SystemExit(f"INCLUDE_LOCAL_MODELS=1 但模型目录不存在：{model_root}")
    datas.append((str(model_root), "video_text_mutiplemodal_agent/pretrain_models"))


binaries = []
hiddenimports = [
    # 源项目通过 importlib 按功能加载，必须显式告诉 PyInstaller。
    "text_processor_subagent_stage_3",
    "text_processor_subagent_stage_3.common",
    "text_processor_subagent_stage_3.common.data_preprocess",
    "text_processor_subagent_stage_3.common.textEmbedding",
    "text_processor_subagent_stage_3.common.remark_rule_enginee",
    "text_processor_subagent_stage_3.clustering",
    "text_processor_subagent_stage_3.clustering.clustering",
    "text_processor_subagent_stage_3.clustering.textKeyBert_cluster_sentence_fuse_agent",
    "text_processor_subagent_stage_3.sentiment",
    "text_processor_subagent_stage_3.sentiment.chinese_sentiment_bert_analysis",
    "text_processor_subagent_stage_3.sentiment.sentiment_general_analysis",
    "text_processor_subagent_stage_3.sentiment.textkeyBert_sentiment_sentence",
    "text_processor_subagent_stage_3.utils",
    "umap",
    "chromadb.telemetry.product.posthog",
    "milvus_lite",
    "milvus_lite.server_manager",
    "milvus_lite.adapter.grpc.server",
]

# 这些库含有动态库、数据文件或运行时元数据。不要使用 collect_all：
# 它会把 pandas/sklearn/torch 的 tests、examples 和开发工具全部装入包。
for package_name in (
    "torch",
    "onnxruntime",
    "tokenizers",
    "qwen_agent",
):
    try:
        datas += collect_data_files(package_name)
        binaries += collect_dynamic_libs(package_name)
    except Exception:
        continue
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass


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


try:
    hiddenimports += collect_submodules(
        "qwen_agent",
        filter=_runtime_module,
        on_error="ignore",
    )
except Exception:
    pass

for package_name in ("chromadb", "milvus_lite"):
    try:
        hiddenimports += collect_submodules(
            package_name,
            filter=_runtime_module,
            on_error="ignore",
        )
    except Exception:
        pass

try:
    # milvus_lite._version 通过 importlib.metadata 读取发行版名称。
    datas += copy_metadata("milvus-lite")
except Exception:
    pass

hiddenimports = sorted(set(hiddenimports))


a = Analysis(
    [str(PROJECT_ROOT / "qt_client.py")],
    pathex=[str(PROJECT_ROOT), str(STAGE3_ROOT.parent)],
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
    argv_emulation=False,
)
app = BUNDLE(
    COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name="AcademicAgent",
    ),
    name="AcademicAgent.app",
    icon=str(PROJECT_ROOT / "resources" / "academic_agent_icon.icns"),
    bundle_identifier="com.academic.agent",
)
