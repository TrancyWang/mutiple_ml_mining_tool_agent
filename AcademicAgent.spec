# -*- mode: python ; coding: utf-8 -*-
"""Academic Agent macOS onedir/.app 构建配置。

默认不携带本地大模型；构建时设置 INCLUDE_LOCAL_MODELS=1 才会把
text_mining_tools_agent_client_cpu/pretrain_models 一起复制进应用包。
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
        "TEXT_MINING_SOURCE_ROOT",
        str(PROJECT_ROOT.parent / "text_mining_tools_agent_client_cpu"),
    )
).expanduser().resolve()
SRC_CODE_ROOT = SOURCE_ROOT / "src_code"

if not SRC_CODE_ROOT.is_dir():
    raise SystemExit(
        "找不到新版文本算法源项目的 src_code："
        f" {SRC_CODE_ROOT}\n"
        "请设置 TEXT_MINING_SOURCE_ROOT 指向 text_mining_tools_agent_client_cpu。"
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
        str(SRC_CODE_ROOT),
        "text_mining_tools_agent_client_cpu/src_code",
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
    datas.append((str(model_root), "text_mining_tools_agent_client_cpu/pretrain_models"))


binaries = []
hiddenimports = [
    # 源项目通过 importlib 按功能加载，必须显式告诉 PyInstaller。
    "src_code",
    "src_code.preprocessing",
    "src_code.preprocessing.data_preprocess",
    "src_code.text_mining",
    "src_code.text_mining.embeddings_models",
    "src_code.text_mining.embeddings_models.textEmbedding",
    "src_code.text_mining.sentiment",
    "src_code.text_mining.sentiment.chinese_sentiment_bert_analysis",
    "src_code.text_mining.sentiment.sentiment_general_analysis",
    "src_code.text_mining.keyword_extractor",
    "src_code.text_mining.keyword_extractor.textKeyBert_cluster_sentence_fuse_pro",
    "umap",
    "chromadb.telemetry.product.posthog",
    "milvus_lite",
    "milvus_lite.server_manager",
    "milvus_lite.adapter.grpc.server",
    "mysql.connector",
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
    pathex=[str(PROJECT_ROOT), str(SOURCE_ROOT)],
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
    info_plist={
        "NSMicrophoneUsageDescription": "Academic Agent 需要使用麦克风，将你的语音转换为对话文字。",
    },
)
