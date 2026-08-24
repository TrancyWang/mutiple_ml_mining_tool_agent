"""Agent Runtime 的请求、计划与执行作用域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AgentRoute(str, Enum):
    CHAT = "chat"
    PROJECT = "project"
    DATA = "data"
    MACHINE_LEARNING = "machine_learning"
    VISUALIZATION = "visualization"


@dataclass(frozen=True, slots=True)
class AgentRequest:
    messages: list[dict[str, Any]]
    user_id: str
    session_id: str
    workspace_path: str | None = None
    chat_only: bool = False

    @property
    def query(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""


@dataclass(frozen=True, slots=True)
class PlanStep:
    name: str
    description: str
    suggested_tools: tuple[str, ...] = ()
    requires_observation: bool = False


@dataclass(frozen=True, slots=True)
class AgentPlan:
    objective: str
    route: AgentRoute
    steps: tuple[PlanStep, ...]
    available_tools: tuple[str, ...] = ()
    requires_confirmation: bool = False

    def as_prompt(self) -> str:
        lines = [f"目标：{self.objective}", f"路由：{self.route.value}", "建议步骤："]
        for index, step in enumerate(self.steps, 1):
            tool_hint = f"；候选工具：{', '.join(step.suggested_tools)}" if step.suggested_tools else ""
            lines.append(f"{index}. {step.description}{tool_hint}")
        lines.append("该计划是运行时建议；应依据工具观察结果动态调整。界面会展示简化步骤，最终回答不要重复内部计划。")
        return "\n".join(lines)


@dataclass(slots=True)
class ExecutionScope:
    request: AgentRequest
    plan: AgentPlan
    events: list[dict[str, Any]] = field(default_factory=list)
    event_callback: Callable[[dict[str, Any]], None] | None = None

    def record(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        self.events.append(event)
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception:
                pass
