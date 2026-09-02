"""Embedding and model provider adapters."""

from .embedding import HashEmbedder, OpenAICompatibleClient

__all__ = ["HashEmbedder", "OpenAICompatibleClient"]
