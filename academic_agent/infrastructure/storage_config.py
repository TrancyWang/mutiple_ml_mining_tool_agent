"""统一管理 Academic Agent 的记忆存储后端配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6.QtCore import QSettings


SETTINGS_ORG = "AcademicAgent"
SETTINGS_APP = "Storage"


@dataclass(frozen=True)
class StorageConfig:
    elasticsearch: bool = False
    milvus: bool = False

    @property
    def label(self) -> str:
        if self.elasticsearch and self.milvus:
            return "Elasticsearch + Milvus Lite"
        if self.elasticsearch:
            return "Elasticsearch"
        if self.milvus:
            return "Milvus Lite"
        return "SQLite（默认）"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def apply_storage_environment(config: StorageConfig) -> StorageConfig:
    """把配置同步到记忆服务读取的环境变量。"""
    os.environ["ES_ENABLED"] = "true" if config.elasticsearch else "false"
    os.environ["MILVUS_ENABLED"] = "true" if config.milvus else "false"
    return config


def load_persisted_storage_config() -> StorageConfig:
    """读取设置页保存的配置；首次运行明确使用 SQLite。"""
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    configured = _as_bool(settings.value("configured", False))
    if not configured:
        return StorageConfig()
    return StorageConfig(
        elasticsearch=_as_bool(settings.value("elasticsearch", False)),
        milvus=_as_bool(settings.value("milvus", False)),
    )


def apply_persisted_storage_config() -> StorageConfig:
    return apply_storage_environment(load_persisted_storage_config())


def save_storage_config(config: StorageConfig) -> StorageConfig:
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.setValue("configured", True)
    settings.setValue("elasticsearch", config.elasticsearch)
    settings.setValue("milvus", config.milvus)
    settings.sync()
    return apply_storage_environment(config)


def current_storage_config() -> StorageConfig:
    return StorageConfig(
        elasticsearch=_as_bool(os.getenv("ES_ENABLED", "false")),
        milvus=_as_bool(os.getenv("MILVUS_ENABLED", "false")),
    )
