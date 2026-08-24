"""模型 Agent 生命周期控制；不依赖 Qt。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from academic_agent.models import ApplicationState


@dataclass(slots=True)
class AgentController:
    state: ApplicationState
    service: Any | None = field(default=None, repr=False)

    def _get_service(self):
        """首次需要模型时才加载重量级算法和向量依赖。"""
        if self.service is None:
            from academic_agent.agent.adapters.qwen import agent_service

            self.service = agent_service
        return self.service

    def initialize(self) -> tuple[bool, str]:
        if self.state.auth_session is None:
            return False, "访客模式：使用 Agent 功能时需要先登录"
        service = self._get_service()
        if service.agent is not None:
            active = service.llm_config.get("model", "当前模型")
            return True, f"Agent 已就绪：{active}"
        if service.init_agent() is None:
            detail = service.last_init_error or "未获得底层错误信息"
            return False, f"Agent 初始化失败：{detail}"
        active = service.llm_config.get("model", "当前模型")
        return True, f"Agent 已就绪：{active}"

    def reset(self) -> None:
        if self.service is None:
            return
        self.service.agent = None
        if hasattr(self.service, "chat_agent"):
            self.service.chat_agent = None

    def switch_provider(
        self,
        provider: str,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        from academic_agent.agent.providers.config import get_llm_config_priority

        service = self._get_service()
        previous_agent = service.agent
        previous_config = dict(service.llm_config)
        try:
            config = get_llm_config_priority(
                model_name=model_name or "gemini-3.6-flash",
                provider=provider,
            )
            if service.init_agent(llm_config=config) is None:
                detail = service.last_init_error or "未获得底层错误信息"
                raise RuntimeError(f"Agent 初始化失败：{detail}")
            self.state.model_provider = provider
            return config
        except Exception:
            service.agent = previous_agent
            service.llm_config = previous_config
            raise
