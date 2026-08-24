"""运行时依赖检查与按功能安装。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Dependency:
    feature: str
    package: str
    import_name: str
    description: str


DEPENDENCIES = [
    Dependency("基础数据处理", "pandas", "pandas", "CSV、Excel、数据预览"),
    Dependency("可视化与 PDF 生成", "matplotlib", "matplotlib", "折线图、分布图、图片输出和 PDF 文档"),
    Dependency("可视化", "seaborn", "seaborn", "统计图表美化"),
    Dependency("机器学习", "scikit-learn", "sklearn", "聚类、分类、回归"),
    Dependency("文本向量", "sentence-transformers", "sentence_transformers", "BGE 中文向量模型"),
    Dependency("PDF 解析", "pypdf", "pypdf", "读取 PDF 文档"),
    Dependency("Word 读写", "python-docx", "docx", "读取和生成 DOCX 文档"),
    Dependency("PPT 读写", "python-pptx", "pptx", "读取和生成 PPTX 文档"),
    Dependency("Milvus Lite 记忆", "pymilvus[milvus_lite]", "pymilvus", "本地语义记忆，不需要独立 Milvus 服务"),
    Dependency("LlamaIndex 记忆编排", "llama-index-core", "llama_index.core", "统一管理 Node、Embedding 和检索"),
    Dependency("LlamaIndex Milvus 适配", "llama-index-vector-stores-milvus", "llama_index.vector_stores.milvus", "LlamaIndex 对接 Milvus"),
    Dependency("Elasticsearch 记忆", "elasticsearch", "elasticsearch", "可选的全文会话记忆"),
]


def missing_dependencies() -> list[Dependency]:
    return [item for item in DEPENDENCIES if importlib.util.find_spec(item.import_name) is None]


def install_dependencies(items: list[Dependency]) -> tuple[bool, str]:
    if not items:
        return True, "没有需要安装的依赖。"
    packages = list(dict.fromkeys(item.package for item in items))
    command = [sys.executable, "-m", "pip", "install", *packages]
    result = subprocess.run(command, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output[-4000:]
