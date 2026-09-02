"""会话文件、计划和历史记忆的上下文组装。"""

from __future__ import annotations

import json
from typing import Any

from academic_agent.agent.types import AgentPlan, AgentRequest
from academic_agent.agent.session import session_store


class AgentContextBuilder:
    @staticmethod
    def _normalize_message_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages to one leading system message plus conversation turns."""
        normalized: list[dict[str, Any]] = []
        system_parts: list[str] = []
        user_turn_seen = False
        for raw_message in messages:
            message = dict(raw_message)
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content", "")
            if role == "system":
                if str(content or "").strip():
                    system_parts.append(str(content))
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
        if system_parts:
            normalized.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return normalized

    def build(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        memory_context: str = "",
        plan_configuration: dict[str, Any] | None = None,
        plan_document: str = "",
        confirmed_information: list[dict[str, Any]] | None = None,
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

        if confirmed_information:
            system_parts.append(
                "【用户已确认信息】\n"
                f"{json.dumps(confirmed_information, ensure_ascii=False, default=str)}\n"
                "这些信息优先于模型猜测；不要重复询问已经确认的内容。"
            )

        if plan_document:
            system_parts.append(
                "【已确认的自然语言 Plan】\n"
                f"{plan_document}\n"
                "后续执行必须围绕这份 Plan；Plan 与最终产物不是同一份内容。"
            )

        if plan_configuration:
            system_parts.append(
                "【用户已确认的算法配置】\n"
                f"{json.dumps(plan_configuration, ensure_ascii=False, default=str)}\n"
                "调用工具时必须把 tool_arguments 中的字段原样传入（例如聚类的 algorithm、"
                "情感分析的 mode、机器学习的 model_type/method），并使用 parameters 中的值；"
                "不要用其他算法替换，也不要覆盖用户已填写的参数。"
            )

        runtime_prompt = "\n\n".join(system_parts)
        system_message = next((item for item in messages if item.get("role") == "system"), None)
        if system_message is None:
            messages.insert(0, {"role": "system", "content": runtime_prompt})
        else:
            system_message["content"] = f"{system_message.get('content', '')}\n\n{runtime_prompt}"
        return messages
