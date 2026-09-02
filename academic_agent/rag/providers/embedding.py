from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

from ..common.config import settings


class HashEmbedder:
    """Dependency-free fallback embedding for local development and tests."""

    def __init__(self, dimension: int = 256):
        self.dimension = max(32, dimension)

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text.lower())
        vector = [0.0] * self.dimension
        for token in tokens:
            index = hash(token) % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAICompatibleClient:
    def __init__(self):
        self.chat_enabled = bool(settings.openai_base_url and settings.chat_model)
        self.embedding_enabled = bool(settings.embedding_base_url and settings.embedding_model)
        self.fallback = HashEmbedder(settings.embedding_dim)
        self.local_bge = None

    def _embed_with_target_bge(self, texts: list[str]) -> list[list[float]] | None:
        """Reuse Academic Agent's configured BGE model when it is available."""
        if self.local_bge is None:
            try:
                from academic_agent.infrastructure.persistence.chroma_store import BGEEmbeddingFunction
                candidate = BGEEmbeddingFunction()
                if not candidate.model_path.is_dir():
                    return None
                self.local_bge = candidate
            except (ImportError, OSError):
                return None
        try:
            return self.local_bge(texts)
        except (ImportError, OSError, RuntimeError, ValueError):
            return None

    def _post(self, url: str, payload: dict[str, Any], key: str, timeout: int) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.embedding_enabled:
            local = self._embed_with_target_bge(texts)
            return local or [self.fallback.embed(text) for text in texts]
        try:
            payload = {"model": settings.embedding_model, "input": texts}
            result = self._post(settings.embedding_base_url + "/embeddings", payload, settings.embedding_api_key, settings.embedding_timeout)
            return [item["embedding"] for item in sorted(result.get("data", []), key=lambda x: x.get("index", 0))]
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            return [self.fallback.embed(text) for text in texts]

    def answer(self, system: str, user: str) -> str | None:
        if not self.chat_enabled:
            return None
        payload = {"model": settings.chat_model, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            result = self._post(settings.openai_base_url + "/chat/completions", payload, settings.openai_api_key, settings.chat_timeout)
            return result["choices"][0]["message"]["content"].strip()
        except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
            return None


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / ((sum(a * a for a in left) ** 0.5 or 1.0) * (sum(b * b for b in right) ** 0.5 or 1.0))
