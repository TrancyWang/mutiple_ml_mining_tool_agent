"""Plan → Context → Model/Tools → Memory 的 Agent 执行闭环。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from academic_agent.agent.context import AgentContextBuilder
from academic_agent.agent.executor import ToolExecutor, tool_executor
from academic_agent.agent.memory.coordinator import MemoryCoordinator
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.response import (
    assistant_content,
    attach_confirmation_link,
)
from academic_agent.agent.types import AgentPlan, AgentRequest, ExecutionScope
from academic_agent.infrastructure.workspace_manager import workspace_manager


class AgentRuntime:
    def __init__(
        self,
        planner: TaskPlanner | None = None,
        memory: MemoryCoordinator | None = None,
        context_builder: AgentContextBuilder | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        self.planner = planner or TaskPlanner()
        self.memory = memory or MemoryCoordinator()
        self.context_builder = context_builder or AgentContextBuilder()
        self.executor = executor or tool_executor
        self.last_plan: AgentPlan | None = None
        self.last_events: list[dict[str, Any]] = []

    def prepare(self, request: AgentRequest) -> tuple[AgentPlan, list[dict[str, Any]]]:
        plan = self.planner.create_plan(request)
        memory_context = self.memory.recall(
            request.user_id,
            request.query,
            request.session_id,
            request.workspace_path,
        )
        messages = self.context_builder.build(request, plan, memory_context)
        self.last_plan = plan
        return plan, messages

    def run(
        self,
        request: AgentRequest,
        model_runner: Callable[[list[dict[str, Any]]], Any],
    ) -> Any:
        plan, messages = self.prepare(request)
        scope = ExecutionScope(request=request, plan=plan)
        pending_before = set(workspace_manager.pending_operations)
        with self.executor.bind(scope):
            generated = model_runner(messages)
            if hasattr(generated, "__iter__") and not isinstance(generated, (list, dict, str)):
                final_response = None
                for response in generated:
                    final_response = response
            else:
                final_response = generated
        attach_confirmation_link(final_response, pending_before)
        self._finish(request, final_response, scope)
        return final_response if final_response is not None else {"error": "无响应"}

    def stream(
        self,
        request: AgentRequest,
        model_runner: Callable[[list[dict[str, Any]]], Iterable[Any]],
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[Any]:
        plan, messages = self.prepare(request)
        if event_callback is not None:
            event_callback({
                "type": "plan_created",
                "route": plan.route.value,
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "suggested_tools": list(step.suggested_tools),
                    }
                    for step in plan.steps
                ],
            })
        scope = ExecutionScope(
            request=request,
            plan=plan,
            event_callback=event_callback,
        )
        pending_before = set(workspace_manager.pending_operations)
        final_response = None
        with self.executor.bind(scope):
            for response in model_runner(messages):
                final_response = response
                yield response
        confirmation_id = attach_confirmation_link(final_response, pending_before)
        if confirmation_id:
            yield final_response
        self._finish(request, final_response, scope)

    def _finish(
        self,
        request: AgentRequest,
        final_response: Any,
        scope: ExecutionScope,
    ) -> None:
        self.last_events = list(scope.events)
        content = assistant_content(final_response)
        self.memory.commit(
            user_id=request.user_id,
            user_text=request.query,
            assistant_text=content,
            session_id=request.session_id,
            workspace_path=request.workspace_path,
        )
