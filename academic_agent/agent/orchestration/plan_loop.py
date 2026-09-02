"""Plan Loop 的信息框架和回填规则。"""

from __future__ import annotations

import json
from typing import Any


MAX_QUESTIONS_PER_ROUND = 3


def normalize_information_frame(
    review: dict[str, Any],
    round_index: int,
    previously_asked: set[str] | None = None,
) -> list[dict[str, Any]]:
    """由程序编号、限量并初始化模型给出的信息缺口。"""
    review = normalize_plan_review(review)
    raw_items = review.get("information_frame")
    if not isinstance(raw_items, list):
        raise ValueError("information_frame 必须是列表")
    if len(raw_items) > MAX_QUESTIONS_PER_ROUND:
        raw_items = raw_items[:MAX_QUESTIONS_PER_ROUND]

    asked = previously_asked or set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        why_needed = str(item.get("why_needed") or "").strip()
        question = str(item.get("question") or "").strip()
        if not topic or not why_needed or not question or question in asked:
            continue
        internal_text = f"{topic} {question}".lower()
        if (
            "plan review" in internal_text
            or "information_frame" in internal_text
            or ("输出结构" in internal_text and ("json" in internal_text or "plan" in internal_text))
        ):
            raise ValueError("information_frame 不能把 Plan Review 的内部格式当成用户问题")
        options = _normalize_options(item)
        normalized.append({
            "info_id": f"R{round_index}-F{index}",
            "topic": topic,
            "why_needed": why_needed,
            "question": question,
            "options": options,
            "expected_format": str(
                item.get("expected_format") or item.get("answer_hint") or ""
            ).strip(),
            "example": str(item.get("example") or "").strip(),
            "default_assumption": str(
                item.get("default_assumption") or item.get("assumption") or ""
            ).strip(),
            "answer": "",
            "status": "missing",
            "source": "",
        })
    plan_ready = review.get("plan_ready") is True
    if plan_ready and normalized:
        raise ValueError("plan_ready=true 时 information_frame 必须为空")
    if not plan_ready and not normalized:
        raise ValueError("Plan 未就绪时必须提供新的信息缺口")
    return normalized


def normalize_plan_review(review: dict[str, Any]) -> dict[str, Any]:
    """兼容模型常见的安全等价写法，同时保留 Plan Review 的边界。"""
    if not isinstance(review, dict):
        raise ValueError("Plan Review 必须返回对象")

    # 有些模型会把结果再包一层 ``plan_review``，只在值确实是对象时展开。
    nested = review.get("plan_review")
    normalized = dict(nested) if isinstance(nested, dict) else dict(review)

    raw_frame = normalized.get("information_frame")
    if raw_frame is None:
        for alias in ("information_gaps", "missing_information", "questions"):
            if alias in normalized:
                raw_frame = normalized.get(alias)
                break
    if raw_frame is None:
        raw_frame = []
    elif isinstance(raw_frame, dict):
        raw_frame = [raw_frame]
    elif isinstance(raw_frame, tuple):
        raw_frame = list(raw_frame)
    elif isinstance(raw_frame, str):
        text = raw_frame.strip()
        if not text:
            raw_frame = []
        else:
            try:
                raw_frame = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("information_frame 必须是列表") from exc
            if isinstance(raw_frame, dict):
                raw_frame = [raw_frame]
    if not isinstance(raw_frame, list):
        raise ValueError("information_frame 必须是列表")

    items: list[dict[str, Any]] = []
    for raw_item in raw_frame:
        if not isinstance(raw_item, dict):
            items.append(raw_item)
            continue
        item = dict(raw_item)
        aliases = {
            "topic": ("topic", "subject", "title", "name"),
            "why_needed": ("why_needed", "reason", "why"),
            "question": ("question", "prompt", "ask"),
            "options": ("options", "choices", "候选项"),
        }
        for target, names in aliases.items():
            if item.get(target) is not None:
                continue
            for name in names:
                if item.get(name) is not None:
                    item[target] = item[name]
                    break
        items.append(item)
    normalized["information_frame"] = items

    raw_ready = normalized.get("plan_ready")
    if isinstance(raw_ready, bool):
        pass
    elif isinstance(raw_ready, int) and raw_ready in {0, 1}:
        normalized["plan_ready"] = bool(raw_ready)
    elif isinstance(raw_ready, str):
        value = raw_ready.strip().lower()
        if value in {"true", "1", "yes", "y", "是", "已就绪", "就绪"}:
            normalized["plan_ready"] = True
        elif value in {"false", "0", "no", "n", "否", "未就绪", "不足"}:
            normalized["plan_ready"] = False
        else:
            raise ValueError("plan_ready 必须是布尔值")
    elif raw_ready is None:
        # 模型漏掉该字段时，只根据是否存在信息缺口做保守推断。
        normalized["plan_ready"] = not bool(items)
    else:
        raise ValueError("plan_ready 必须是布尔值")
    return normalized


def _normalize_options(item: dict[str, Any]) -> list[dict[str, str]]:
    """保留模型给出的单选项，并为旧格式提供一个安全默认项。"""
    normalized: list[dict[str, str]] = []
    raw_options = item.get("options") or item.get("choices") or []
    if isinstance(raw_options, list):
        for raw in raw_options[:6]:
            if isinstance(raw, dict):
                label = str(raw.get("label") or raw.get("value") or "").strip()
                value = str(raw.get("value") or label).strip()
                description = str(raw.get("description") or "").strip()
            else:
                label = str(raw).strip()
                value = label
                description = ""
            if label and value:
                normalized.append({
                    "label": label,
                    "value": value,
                    "description": description,
                })
    if normalized:
        return normalized
    default = str(item.get("default_assumption") or item.get("assumption") or "").strip()
    return [{
        "label": "按建议的默认方案继续",
        "value": default or "未特别指定，请采用低风险、可逆的默认方案。",
        "description": default or "不额外指定特殊要求。",
    }]


def apply_user_answer(
    information_frame: list[dict[str, Any]],
    info_id: str,
    answer: str,
) -> dict[str, Any]:
    """只回填当前问题，保留其余缺口。"""
    value = str(answer or "").strip()
    if not value:
        raise ValueError("用户回答不能为空")
    for item in information_frame:
        if item.get("info_id") == info_id:
            item["answer"] = value
            item["status"] = "filled"
            item["source"] = "user"
            return item
    raise ValueError(f"不存在的信息项：{info_id}")


def next_missing_info(information_frame: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in information_frame:
        if item.get("status") == "missing":
            return item
    return None
