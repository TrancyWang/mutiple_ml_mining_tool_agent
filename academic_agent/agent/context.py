"""会话文件、计划和历史记忆的上下文组装。"""

from __future__ import annotations

import json
from typing import Any

from academic_agent.agent.types import AgentPlan, AgentRequest
from academic_agent.agent.session import session_store


class AgentContextBuilder:
    @staticmethod
    def _normalize_message_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure OpenAI-compatible providers receive user as the first non-system turn."""
        normalized: list[dict[str, Any]] = []
        user_turn_seen = False
        for raw_message in messages:
            message = dict(raw_message)
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content", "")
            if role == "system":
                normalized.append(message)
                continue
            if role not in {"user", "assistant", "tool"}:
                continue
            if not str(content or "").strip():
                continue
            # A UI-only status message (for example, after file upload) can be
            # an assistant turn before the first user turn. It is not model
            # context; session schema already carries the upload information.
            if not user_turn_seen and role != "user":
                continue
            if role == "user":
                user_turn_seen = True
            normalized.append({"role": role, "content": content})
        return normalized

    def build(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        messages = self._normalize_message_order(request.messages)
        system_parts = [f"【运行时执行计划】\n{plan.as_prompt()}"]

        session = session_store.get(request.session_id)
        if session.uploaded_files:
            system_parts.append(
                "【用户上传文件】\n"
                f"{json.dumps(session.uploaded_files, ensure_ascii=False)}\n"
                "这些文件由用户主动上传，是当前会话的授权输入，即使它们位于工作区之外也可以用于分析。"
                "不要把它们当作可任意修改的项目文件；由上传文件产生的结果写入应用同级 output。"
            )
        if session.schema:
            system_parts.append(
                "【当前文件上下文】\n"
                f"{json.dumps(session.schema, ensure_ascii=False, default=str)}\n"
                "请基于当前文件字段、数据规模和文本列回答；缺少参数时主动询问。"
            )
        if memory_context:
            system_parts.append(
                "【历史记忆上下文】\n"
                f"{memory_context}\n"
                "仅使用与当前请求相关的历史信息，无关内容必须忽略。"
            )

        runtime_prompt = "\n\n".join(system_parts)
        system_message = next((item for item in messages if item.get("role") == "system"), None)
        if system_message is None:
            messages.insert(0, {"role": "system", "content": runtime_prompt})
        else:
            system_message["content"] = f"{system_message.get('content', '')}\n\n{runtime_prompt}"
        return messages
