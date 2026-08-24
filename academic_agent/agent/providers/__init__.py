"""大模型供应商配置。"""

from academic_agent.agent.providers.config import (
    get_lightweight_text_config,
    get_llm_config_priority,
    get_qwen_fallback_config,
)
from academic_agent.agent.providers.multimodal import MultimodalProviderManager

__all__ = [
    "get_lightweight_text_config",
    "get_llm_config_priority",
    "get_qwen_fallback_config",
    "MultimodalProviderManager",
]
