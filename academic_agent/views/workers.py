"""Qt 后台线程。"""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from academic_agent.agent.adapters.qwen import agent_service


class StreamWorker(QThread):
    """在后台线程中消费 Agent 流式响应，避免阻塞 Qt 主线程。"""

    chunk = Signal(str)
    progress = Signal(dict)
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str | None = None,
        workspace_path: str | None = None,
        chat_only: bool = False,
    ):
        super().__init__()
        self.messages = messages
        self.user_id = user_id
        self.session_id = session_id
        self.workspace_path = workspace_path
        self.chat_only = chat_only
        self._plan_decision_event = threading.Event()
        self._plan_configuration: dict[str, Any] | None = None
        self._plan_review_event = threading.Event()
        self._plan_decision: dict[str, Any] | None = None
        self._clarification_event = threading.Event()
        self._clarification_answer: str | None = None

    def set_plan_configuration(self, configuration: dict[str, Any] | None) -> None:
        """由 Qt 主线程提交算法确认窗口的结果，并唤醒后台 Agent。"""
        self._plan_configuration = dict(configuration or {"cancelled": True})
        self._plan_decision_event.set()

    def _wait_for_plan_configuration(self, plan) -> dict[str, Any]:
        self._plan_configuration = None
        self._plan_decision_event.clear()
        self._plan_decision_event.wait()
        return self._plan_configuration or {"cancelled": True}

    def set_plan_decision(self, decision: dict[str, Any] | None) -> None:
        self._plan_decision = dict(decision or {"cancelled": True})
        self._plan_review_event.set()

    def _wait_for_plan_review(self, plan, board) -> dict[str, Any]:
        self._plan_decision = None
        self._plan_review_event.clear()
        self.progress.emit({
            "type": "plan_review_required",
            "plan": {
                "plan_id": plan.plan_id,
                "objective": plan.objective,
                "route": plan.route.value,
                "expected_outputs": list(plan.expected_outputs),
                "plan_document": plan.plan_document,
                "is_replan": bool(getattr(plan, "is_replan", False)),
                "replan_reason": str(getattr(plan, "replan_reason", "")),
            },
            "board": board.to_dict(),
        })
        self._plan_review_event.wait()
        return self._plan_decision or {"cancelled": True}

    def set_clarification_answer(self, answer: str | None) -> None:
        self._clarification_answer = answer
        self._clarification_event.set()

    def _wait_for_clarification(self, item: dict[str, Any]) -> str | None:
        self._clarification_answer = None
        self._clarification_event.clear()
        self.progress.emit({"type": "clarification_required", "information": item})
        self._clarification_event.wait()
        return self._clarification_answer

    def run(self) -> None:
        try:
            self.progress.emit({"type": "phase_started", "key": "prepare", "label": "准备任务上下文"})
            if agent_service.agent is None:
                self.progress.emit({"type": "phase_started", "key": "model", "label": "初始化模型服务"})
                agent_service.init_agent()
            if agent_service.agent is None:
                detail = agent_service.last_init_error or "未获得底层错误信息"
                self.failed.emit(f"Agent 初始化失败：{detail}")
                return
            response_started = False
            for response in agent_service.chat_stream(
                self.messages,
                user_id=self.user_id,
                session_id=self.session_id,
                workspace_path=self.workspace_path,
                chat_only=self.chat_only,
                progress_callback=self.progress.emit,
                plan_confirmation_callback=self._wait_for_plan_configuration,
                plan_review_callback=self._wait_for_plan_review,
                clarification_callback=self._wait_for_clarification,
            ):
                if isinstance(response, dict) and response.get("error"):
                    self.failed.emit(response["error"])
                    return
                if isinstance(response, list) and response:
                    content = response[-1].get("content", "") or ""
                    if content:
                        if not response_started:
                            self.progress.emit({"type": "response_started"})
                            response_started = True
                        self.chunk.emit(content)
            self.progress.emit({"type": "completed"})
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
