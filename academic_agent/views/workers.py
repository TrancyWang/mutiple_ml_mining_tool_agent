"""Qt 后台线程。"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from academic_agent.agent.adapters.qwen import agent_service


class StreamWorker(QThread):
    """在后台线程中消费 Agent 流式响应，避免阻塞 Qt 主线程。"""

    chunk = pyqtSignal(str)
    progress = pyqtSignal(dict)
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

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
