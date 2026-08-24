"""兼容入口：源项目适配器已迁移到 integrations。"""

from academic_agent.integrations.video_text_adapter import (
    SourceVideoTextAdapter,
    source_video_text_adapter,
)

__all__ = ["SourceVideoTextAdapter", "source_video_text_adapter"]
