"""OpenAI-compatible LLM 实体识别和关系抽取。

只依赖 Python 标准库，通过 OpenAI-compatible ``/chat/completions`` 接口兼容
Gemini、Qwen、Ollama 及其他本地/云端模型。模型输出经过严格的 JSON 解析和
字段归一化后才返回给上层，避免把模型生成的自由文本直接当成结构化结果。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from academic_agent.agent.providers.config import get_llm_config_priority


@dataclass(frozen=True)
class ExtractedBatch:
    items: list[dict[str, Any]]
    model: str
    provider: str


def _completion_url(base_url: str) -> str:
    base_url = str(base_url).rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


def _content_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM 响应缺少 choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    if not str(content).strip():
        raise ValueError("LLM 响应内容为空")
    return str(content)


def _parse_json(content: str) -> Any:
    """解析纯 JSON、Markdown fenced JSON 或夹带说明文字的 JSON。"""
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("LLM 没有返回合法 JSON")


def _text_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("name") or value.get("value") or "").strip()
    return str(value or "").strip()


class LLMInformationExtractor:
    """调用 OpenAI-compatible 对话模型完成实体识别和关系抽取。"""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        explicit_base_url = base_url or os.getenv("LLM_EXTRACTION_BASE_URL")
        explicit_api_key = api_key or os.getenv("LLM_EXTRACTION_API_KEY")
        requested_model = model or os.getenv("LLM_EXTRACTION_MODEL") or "gemini-3.6-flash"
        config = get_llm_config_priority(
            requested_model,
            provider=provider or os.getenv("LLM_EXTRACTION_PROVIDER") or os.getenv("LLM_PROVIDER", "auto"),
        )
        if explicit_base_url:
            config["model_server"] = explicit_base_url
        if explicit_api_key:
            config["api_key"] = explicit_api_key
        if model:
            config["model"] = model
        self.config = config
        self.timeout = float(timeout or os.getenv("LLM_EXTRACTION_TIMEOUT", "90"))

    @property
    def model(self) -> str:
        return str(self.config.get("model") or "configured-llm")

    @property
    def provider(self) -> str:
        return str(self.config.get("provider") or "openai-compatible")

    def _chat_completion(self, messages: list[dict[str, str]], max_tokens: int = 4096) -> str:
        base_url = self.config.get("model_server")
        if not base_url:
            raise ValueError("未配置 LLM_EXTRACTION_BASE_URL 或模型服务地址")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": int(max_tokens),
        }
        request = Request(
            _completion_url(str(base_url)),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.get('api_key') or 'EMPTY'}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM 接口返回 HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"无法连接 LLM 服务: {exc.reason}") from exc
        return _content_from_response(payload)

    @staticmethod
    def _prompt(items: list[dict[str, Any]], extraction_type: str) -> list[dict[str, str]]:
        type_description = {
            "entities": "只识别实体",
            "relations": "只抽取关系",
            "both": "同时识别实体和关系",
        }[extraction_type]
        user_payload = json.dumps(items, ensure_ascii=False)
        return [
            {
                "role": "system",
                "content": (
                    "你是严谨的中文学术文本信息抽取器。" + type_description + "。"
                    "只根据文本原文，不要补造文本中没有出现的实体或关系。"
                    "必须只返回 JSON，不要 Markdown、解释、前后缀。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请处理以下文本列表。每个元素的 id 必须原样保留。"
                    "JSON 结构必须是 {\"items\":[{\"id\":0,\"entities\":[],\"relations\":[]}] }。"
                    "实体字段为 text、type、start、end、confidence；start/end 是原文字符位置，无法确定时用 null。"
                    "关系字段为 subject、subject_type、predicate、object、object_type、confidence。"
                    "没有结果就返回空数组。\n文本列表：\n" + user_payload
                ),
            },
        ]

    @staticmethod
    def _normalize_entity(entity: Any, source_text: str) -> dict[str, Any] | None:
        if not isinstance(entity, dict):
            return None
        text = _text_value(entity.get("text") or entity.get("name"))
        entity_type = _text_value(entity.get("type") or entity.get("label") or "实体")
        if not text:
            return None
        start = entity.get("start")
        end = entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            start = source_text.find(text)
            end = start + len(text) if start >= 0 else None
        return {
            "text": text,
            "type": entity_type or "实体",
            "start": start,
            "end": end,
            "confidence": LLMInformationExtractor._confidence(entity.get("confidence")),
        }

    @staticmethod
    def _normalize_relation(relation: Any) -> dict[str, Any] | None:
        if not isinstance(relation, dict):
            return None
        subject = _text_value(relation.get("subject") or relation.get("head") or relation.get("主语"))
        obj = _text_value(relation.get("object") or relation.get("tail") or relation.get("宾语"))
        predicate = _text_value(relation.get("predicate") or relation.get("relation") or relation.get("关系"))
        if not subject or not obj or not predicate:
            return None
        return {
            "subject": subject,
            "subject_type": _text_value(relation.get("subject_type") or relation.get("head_type") or "实体"),
            "predicate": predicate,
            "object": obj,
            "object_type": _text_value(relation.get("object_type") or relation.get("tail_type") or "实体"),
            "confidence": LLMInformationExtractor._confidence(relation.get("confidence")),
        }

    @staticmethod
    def _confidence(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return round(min(max(float(value), 0.0), 1.0), 6)
        except (TypeError, ValueError):
            return None

    def extract_batch(
        self,
        texts: Iterable[str],
        extraction_type: str = "both",
        batch_size: int = 8,
    ) -> ExtractedBatch:
        if extraction_type not in {"entities", "relations", "both"}:
            raise ValueError("extraction_type 只能是 entities、relations 或 both")
        clean_texts = [str(text).strip() for text in texts]
        all_items: list[dict[str, Any]] = []
        for start in range(0, len(clean_texts), max(1, int(batch_size))):
            batch = clean_texts[start:start + max(1, int(batch_size))]
            request_items = [{"id": index, "text": text} for index, text in enumerate(batch)]
            content = self._chat_completion(self._prompt(request_items, extraction_type))
            payload = _parse_json(content)
            raw_items = payload.get("items", []) if isinstance(payload, dict) else payload
            if not isinstance(raw_items, list):
                raise ValueError("LLM JSON 中 items 必须是数组")
            by_id = {int(item["id"]): item for item in raw_items if isinstance(item, dict) and str(item.get("id", "")).isdigit()}
            for local_id, source_text in enumerate(batch):
                raw = by_id.get(local_id, {})
                raw_entities = raw.get("entities", []) if isinstance(raw, dict) else []
                raw_relations = raw.get("relations", []) if isinstance(raw, dict) else []
                entities = []
                seen_entities = set()
                for entity in raw_entities if isinstance(raw_entities, list) else []:
                    normalized = self._normalize_entity(entity, source_text)
                    if normalized:
                        key = tuple(normalized.get(key) for key in ("text", "type", "start", "end"))
                        if key not in seen_entities:
                            seen_entities.add(key)
                            entities.append(normalized)
                relations = []
                seen_relations = set()
                for relation in raw_relations if isinstance(raw_relations, list) else []:
                    normalized = self._normalize_relation(relation)
                    if normalized:
                        key = tuple(normalized.get(key) for key in ("subject", "predicate", "object"))
                        if key not in seen_relations:
                            seen_relations.add(key)
                            relations.append(normalized)
                all_items.append({"id": start + local_id, "text": source_text, "entities": entities, "relations": relations})
        return ExtractedBatch(all_items, self.model, self.provider)

