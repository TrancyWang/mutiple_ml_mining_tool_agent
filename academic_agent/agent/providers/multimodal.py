"""Provider abstraction for future Gemini/Qwen/local multimodal calls.

The adapter is intentionally lazy: importing the project does not make network
requests. A local OpenAI-compatible endpoint can be configured with
MULTIMODAL_BASE_URL, while concrete Gemini/Qwen adapters can be added without
changing analysis tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


class MultimodalProviderManager:
    def __init__(self) -> None:
        self.provider = os.getenv("MULTIMODAL_PROVIDER", "local")
        self.config = ProviderConfig(
            name=self.provider,
            model=os.getenv("MULTIMODAL_MODEL", ""),
            base_url=os.getenv("MULTIMODAL_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("MULTIMODAL_API_KEY", ""),
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.config.name,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "configured": bool(self.config.model or self.config.api_key),
            "supported_inputs": ["text", "image", "audio", "video"],
        }

    def analyze(self, prompt: str, **media: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "多模态 Provider 接口已预留；请在 provider adapter 中实现 Gemini、Qwen 或本地 OpenAI-compatible 调用。"
        )


multimodal_provider_manager = MultimodalProviderManager()
