"""Plan Loop → Task Board → Convergence Loop 的 Agent Runtime。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from academic_agent.agent.context import AgentContextBuilder
from academic_agent.agent.executor import ToolExecutor, tool_executor
from academic_agent.agent.memory.coordinator import MemoryCoordinator
from academic_agent.agent.orchestration.plan_loop import (
    apply_user_answer,
    next_missing_info,
    normalize_information_frame,
    normalize_plan_review,
)
from academic_agent.agent.orchestration.verifier import TaskVerifier
from academic_agent.agent.planning.algorithm_scheduler import (
    build_algorithm_catalog,
    fallback_algorithm_schedule,
    normalize_algorithm_schedule,
)
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.response import assistant_content, attach_confirmation_link
from academic_agent.agent.session import session_store
from academic_agent.agent.specialists import AgentCoordinator, agent_catalog
from academic_agent.agent.types import (
    AgentPlan,
    AgentRequest,
    AgentRoute,
    ExecutionScope,
    TaskBoard,
    TaskItem,
    TaskStatus,
)
from academic_agent.harness import RunConfig, RunEnvelope, RunEventRecorder
from academic_agent.harness.contracts import HarnessBudgetExceeded
from academic_agent.infrastructure.runtime_paths import output_root
from academic_agent.infrastructure.task_workspace import WorkspaceContext
from academic_agent.infrastructure.workspace_manager import WorkspaceManager, workspace_manager
from academic_agent.repositories.sqlite_store import SQLiteConversationStore, sqlite_memory


ModelRunner = Callable[[list[dict[str, Any]]], Any]
SemanticRunner = Callable[[list[dict[str, Any]]], Any]
ClarificationCallback = Callable[[dict[str, Any]], str | None]
PlanReviewCallback = Callable[[AgentPlan, TaskBoard], dict[str, Any] | None]


class AgentRuntime:
    """由程序控制状态、顺序、重试和退出，由模型处理语义判断。"""

    def __init__(
        self,
        planner: TaskPlanner | None = None,
        memory: MemoryCoordinator | None = None,
        context_builder: AgentContextBuilder | None = None,
        executor: ToolExecutor | None = None,
        verifier: TaskVerifier | None = None,
        coordinator: AgentCoordinator | None = None,
        run_store: SQLiteConversationStore | None = None,
    ) -> None:
        self.planner = planner or TaskPlanner()
        self.memory = memory or MemoryCoordinator()
        self.context_builder = context_builder or AgentContextBuilder()
        self.executor = executor or tool_executor
        self.verifier = verifier or TaskVerifier()
        self.coordinator = coordinator or AgentCoordinator()
        self.run_store = run_store or sqlite_memory
        self.last_plan: AgentPlan | None = None
        self.last_task_board: TaskBoard | None = None
        self.last_events: list[dict[str, Any]] = []

    def get_persisted_run(self, run_id: str) -> dict[str, Any] | None:
        return self.run_store.get_agent_run(run_id)

    @staticmethod
    def _workspace_id(request: AgentRequest, plan: AgentPlan) -> str | None:
        """Return a stable task-workspace identity without creating one for chat."""
        if request.chat_only or plan.route is AgentRoute.CHAT or not request.workspace_path:
            return None
        if request.workspace_id:
            return str(request.workspace_id)
        identity = f"{request.user_id}\n{request.session_id}\n{Path(request.workspace_path).expanduser().resolve()}"
        return f"ws-{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex[:20]}"

    def _ensure_workspace(self, request: AgentRequest, plan: AgentPlan) -> AgentRequest:
        workspace_id = self._workspace_id(request, plan)
        if not workspace_id:
            return request
        bound = replace(request, workspace_id=workspace_id)
        root = Path(str(bound.workspace_path)).expanduser().resolve()
        permission_mode = WorkspaceManager(root).permission_mode
        self.run_store.ensure_workspace(
            workspace_id=workspace_id,
            user_id=bound.user_id,
            session_id=bound.session_id,
            workspace_path=root,
            output_path=root / "output",
            permission_mode=permission_mode,
            metadata={"route": plan.route.value, "plan_id": plan.plan_id},
        )
        return bound

    @staticmethod
    def _workspace_context(request: AgentRequest, plan: AgentPlan) -> WorkspaceContext | None:
        if not request.workspace_id or not request.workspace_path or plan.route is AgentRoute.CHAT:
            return None
        root = Path(request.workspace_path).expanduser().resolve()
        permission_mode = WorkspaceManager(root).permission_mode
        return WorkspaceContext(
            workspace_id=str(request.workspace_id),
            user_id=request.user_id,
            session_id=request.session_id,
            root_path=str(root),
            output_path=str(root / "output"),
            permission_mode=permission_mode,
            allowed_tools=plan.available_tools,
            metadata={"route": plan.route.value, "plan_id": plan.plan_id},
        )

    def _sync_workspace_artifacts(
        self,
        workspace: WorkspaceContext | None,
        run_id: str,
        board: TaskBoard,
    ) -> None:
        """Persist artifact identity while leaving file contents in the backend."""
        if workspace is None:
            return
        allowed_roots = (workspace.root, workspace.output_root, output_root())
        for task in board.tasks.values():
            for raw in task.artifacts:
                candidate = Path(str(raw)).expanduser()
                if not candidate.is_absolute():
                    candidate = workspace.root / candidate
                try:
                    candidate = candidate.resolve()
                    if not candidate.is_file() or not any(
                        candidate.is_relative_to(root.resolve()) for root in allowed_roots
                    ):
                        continue
                    digest = hashlib.sha256()
                    with candidate.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    self.run_store.register_workspace_artifact(
                        workspace.workspace_id,
                        str(candidate),
                        run_id=run_id,
                        task_id=task.id,
                        artifact_type=candidate.suffix.lstrip(".") or "file",
                        sha256=digest.hexdigest(),
                        size_bytes=candidate.stat().st_size,
                        metadata={"title": task.title, "route": workspace.metadata.get("route", "")},
                    )
                except (OSError, RuntimeError, ValueError):
                    continue

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
        model_runner: ModelRunner,
        semantic_runner: SemanticRunner | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        clarification_callback: ClarificationCallback | None = None,
        plan_review_callback: PlanReviewCallback | None = None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None = None,
        resume_run_id: str | None = None,
        run_config: RunConfig | None = None,
    ) -> Any:
        result = self._run_internal(
            request,
            model_runner,
            semantic_runner=semantic_runner,
            event_callback=event_callback,
            clarification_callback=clarification_callback,
            plan_review_callback=plan_review_callback,
            plan_confirmation_callback=plan_confirmation_callback,
            resume_run_id=resume_run_id,
            run_config=run_config,
        )
        return result

    def stream(
        self,
        request: AgentRequest,
        model_runner: ModelRunner,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None = None,
        semantic_runner: SemanticRunner | None = None,
        clarification_callback: ClarificationCallback | None = None,
        plan_review_callback: PlanReviewCallback | None = None,
        resume_run_id: str | None = None,
        run_config: RunConfig | None = None,
    ) -> Iterator[Any]:
        result = self._run_internal(
            request,
            model_runner,
            semantic_runner=semantic_runner,
            event_callback=event_callback,
            clarification_callback=clarification_callback,
            plan_review_callback=plan_review_callback,
            plan_confirmation_callback=plan_confirmation_callback,
            resume_run_id=resume_run_id,
            run_config=run_config,
        )
        if isinstance(result, dict) and result.get("error"):
            yield result
            return
        if event_callback is not None:
            event_callback({"type": "response_started"})
        yield result

    def _run_internal(
        self,
        request: AgentRequest,
        model_runner: ModelRunner,
        semantic_runner: SemanticRunner | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        clarification_callback: ClarificationCallback | None,
        plan_review_callback: PlanReviewCallback | None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None,
        resume_run_id: str | None = None,
        run_config: RunConfig | None = None,
    ) -> Any:
        config = run_config or RunConfig()
        envelope = RunEnvelope.create(request, config, run_id=resume_run_id)
        if resume_run_id:
            existing = self.run_store.get_agent_run(resume_run_id)
            existing_runtime = existing.get("runtime") if isinstance(existing, dict) else {}
            existing_harness = (
                existing_runtime.get("harness")
                if isinstance(existing_runtime, dict) else None
            )
            if isinstance(existing_harness, dict) and existing_harness:
                envelope = RunEnvelope.from_dict(existing_harness)
                if run_config is not None:
                    envelope.config = run_config
            get_cursor = getattr(self.run_store, "get_agent_event_cursor", None)
            if callable(get_cursor):
                envelope.event_count = max(envelope.event_count, int(get_cursor(resume_run_id)))
        recorder = RunEventRecorder(envelope, store=self.run_store, forward=event_callback)
        event_callback = recorder.emit
        recorder.emit({
            "type": "run_started",
            "request_id": envelope.request_id,
            "resume": bool(resume_run_id),
        })
        tracked_model_runner = self._tracked_runner(model_runner, recorder, "model")
        tracked_semantic_runner = (
            self._tracked_runner(semantic_runner, recorder, "semantic")
            if semantic_runner is not None else None
        )
        try:
            if resume_run_id:
                plan, board, messages, auto_mode, run_context = self._restore_checkpoint(
                    request,
                    resume_run_id,
                    event_callback,
                )
                request = run_context.pop("request", request)
                stored_config = run_context.get("run_config")
                if isinstance(stored_config, dict):
                    config = RunConfig.from_dict(stored_config)
                    recorder.configure(config)
                envelope.user_id = request.user_id
                envelope.session_id = request.session_id
                envelope.workspace_path = str(request.workspace_path or "")
                envelope.mode = "chat" if request.chat_only else "work"
            else:
                plan, board, messages, auto_mode = self._prepare_workflow(
                    request,
                    semantic_runner=tracked_semantic_runner,
                    event_callback=event_callback,
                    clarification_callback=clarification_callback,
                    plan_review_callback=plan_review_callback,
                    plan_confirmation_callback=plan_confirmation_callback,
                )
                run_context = {
                    "run_id": envelope.run_id,
                    "replan_count": 0,
                    "recovery_mode": False,
                    "run_config": config.to_dict(),
                }
        except HarnessBudgetExceeded as exc:
            event_callback({"type": "run_interrupted", "run_id": envelope.run_id, "reason": "budget_exceeded", "error": str(exc)})
            return {"error": f"任务执行中断：{exc}"}
        except Exception as exc:
            if event_callback is not None:
                event_callback({"type": "plan_failed", "error": str(exc)})
                event_callback({"type": "run_failed", "reason": "plan_failed", "error": str(exc)})
            if "本次任务未执行" in str(exc):
                message = str(exc).rstrip("。") + "。"
                return {"error": message}
            return {"error": f"Plan 阶段未完成：{exc}"}

        request = self._ensure_workspace(request, plan)
        run_id = str(run_context.get("run_id") or "")
        recovery_mode = bool(run_context.get("recovery_mode", False))
        replan_count = max(0, int(run_context.get("replan_count", 0) or 0))
        envelope.plan_id = plan.plan_id
        envelope.workspace_id = str(request.workspace_id or "")
        envelope.workspace_path = str(request.workspace_path or "")
        direct_chat = (
            request.chat_only
            or semantic_runner is None
            or plan.route is AgentRoute.CHAT
        )
        if run_id and not direct_chat:
            self._checkpoint(
                run_id,
                request,
                plan,
                board,
                messages,
                auto_mode=auto_mode,
                replan_count=replan_count,
                status="running",
                harness_envelope=recorder.snapshot(),
            )
        workspace_context = self._workspace_context(request, plan)
        scope = ExecutionScope(
            request=request,
            plan=plan,
            workspace=workspace_context,
            event_callback=event_callback,
            recovery_mode=recovery_mode,
            budget_guard=recorder.consume,
            harness_snapshot=recorder.snapshot,
        )
        scope.checkpoint_callback = lambda: self._checkpoint_scope(
            run_id, scope, board, messages, auto_mode, replan_count
        )
        pending_before = set(workspace_manager.pending_operations)
        try:
            with workspace_manager.bind(workspace_context.root_path if workspace_context else None):
                with self.executor.bind(scope):
                    if direct_chat:
                        final_response = self._consume(tracked_model_runner, messages)
                    else:
                        final_response = self._execute_task_board(
                            plan,
                            board,
                            messages,
                            tracked_model_runner,
                            tracked_semantic_runner,
                            scope,
                            run_id=run_id,
                            replan_count=replan_count,
                            auto_mode=auto_mode,
                            clarification_callback=clarification_callback,
                            plan_review_callback=plan_review_callback,
                            plan_confirmation_callback=plan_confirmation_callback,
                        )
        except HarnessBudgetExceeded as exc:
            event_callback({"type": "run_interrupted", "run_id": run_id, "reason": "budget_exceeded", "error": str(exc)})
            if run_id:
                self._checkpoint_scope(
                    run_id,
                    scope,
                    board,
                    messages,
                    auto_mode,
                    replan_count,
                    status="interrupted",
                    last_error=str(exc),
                )
            return {"error": f"任务执行中断：{exc}"}
        except Exception as exc:
            if event_callback is not None:
                event_callback({"type": "run_interrupted", "run_id": run_id, "error": str(exc)})
            if run_id:
                self._checkpoint_scope(
                    run_id,
                    scope,
                    board,
                    messages,
                    auto_mode,
                    replan_count,
                    status="interrupted",
                    last_error=str(exc),
                )
            return {"error": f"任务执行中断：{exc}"}
        attach_confirmation_link(final_response, pending_before)
        self._finish(request, final_response, scope)
        event_callback({"type": "run_completed", "run_id": run_id, "plan_id": plan.plan_id, "reason": "workflow_completed"})
        if run_id:
            self.run_store.finish_agent_run(run_id, "completed")
        return final_response if final_response is not None else {"error": "无响应"}

    def _checkpoint(
        self,
        run_id: str,
        request: AgentRequest,
        plan: AgentPlan,
        board: TaskBoard,
        messages: list[dict[str, Any]],
        *,
        auto_mode: bool,
        replan_count: int,
        status: str = "running",
        last_error: str = "",
        harness_envelope: dict[str, Any] | None = None,
    ) -> None:
        if not run_id:
            return
        self.run_store.save_agent_run(
            run_id=run_id,
            user_id=request.user_id,
            session_id=request.session_id,
            workspace_path=request.workspace_path,
            request=request.to_dict(),
            plan=plan.to_dict(),
            board=board.to_dict(),
            messages=messages,
            runtime={
                "auto_mode": bool(auto_mode),
                "replan_count": max(0, int(replan_count)),
                "harness": harness_envelope or {},
            },
            status=status,
            last_error=last_error,
        )

    def _checkpoint_scope(
        self,
        run_id: str,
        scope: ExecutionScope,
        board: TaskBoard,
        messages: list[dict[str, Any]],
        auto_mode: bool,
        replan_count: int,
        *,
        status: str = "running",
        last_error: str = "",
    ) -> None:
        if not run_id:
            return
        for task_id, observations in scope.task_observations.items():
            if task_id in board.tasks:
                board.tasks[task_id].observations = list(observations)
        self._sync_workspace_artifacts(scope.workspace, run_id, board)
        session_store.get(scope.request.session_id).task_board = board.to_dict()
        self._checkpoint(
            run_id,
            scope.request,
            scope.plan,
            board,
            messages,
            auto_mode=auto_mode,
            replan_count=replan_count,
            status=status,
            last_error=last_error,
            harness_envelope=(scope.harness_snapshot() if scope.harness_snapshot else None),
        )

    def _restore_checkpoint(
        self,
        request: AgentRequest,
        run_id: str,
        event_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[AgentPlan, TaskBoard, list[dict[str, Any]], bool, dict[str, Any]]:
        record = self.run_store.get_agent_run(run_id)
        if record is None:
            raise RuntimeError(f"找不到可恢复的运行：{run_id}")
        if str(record.get("user_id")) != str(request.user_id):
            raise RuntimeError("当前用户无权恢复该运行")
        stored_request = AgentRequest.from_dict(record.get("request") or {})
        if request.session_id != stored_request.session_id:
            request = stored_request
        plan = AgentPlan.from_dict(record.get("plan") or {})
        board = TaskBoard.from_dict(record.get("board") or {})
        for task in board.tasks.values():
            if task.status is TaskStatus.RUNNING:
                task.status = TaskStatus.READY
                task.feedback.append("上次运行中断，已恢复为可重试任务")
            elif task.status is TaskStatus.VERIFYING:
                task.status = TaskStatus.READY
                task.resume_verification = bool(task.result)
                task.feedback.append("上次运行在验收阶段中断，已恢复验收")
            elif task.status is TaskStatus.WAITING_USER:
                task.status = TaskStatus.READY
                task.feedback.append("上次运行等待期间中断，已恢复为可执行任务")
        board.current_task_id = None
        runtime = record.get("runtime") if isinstance(record.get("runtime"), dict) else {}
        context = {
            "run_id": str(record["run_id"]),
            "replan_count": max(0, int(runtime.get("replan_count", 0) or 0)),
            "recovery_mode": True,
            "request": stored_request,
            "run_config": (
                runtime.get("harness", {}).get("config")
                if isinstance(runtime.get("harness"), dict)
                else runtime.get("run_config")
            ) or runtime.get("run_config") or {},
        }
        if event_callback is not None:
            event_callback({"type": "run_resumed", "run_id": run_id, "board": board.to_dict()})
            if board.tasks:
                event_callback({"type": "task_board_created", "board": board.to_dict(), "is_resume": True})
        messages = record.get("messages") if isinstance(record.get("messages"), list) else []
        return plan, board, [dict(item) for item in messages if isinstance(item, dict)], bool(runtime.get("auto_mode", False)), context

    def _prepare_workflow(
        self,
        request: AgentRequest,
        semantic_runner: SemanticRunner | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        clarification_callback: ClarificationCallback | None,
        plan_review_callback: PlanReviewCallback | None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None,
    ) -> tuple[AgentPlan, TaskBoard, list[dict[str, Any]], bool]:
        base_plan = self.planner.create_plan(request)
        session = session_store.get(request.session_id)
        confirmed = list(session.confirmed_information)

        # 轻量问候、身份和能力问题不需要 Routing / Plan Review / Task Board。
        # 这一步必须发生在第一次 semantic_runner 调用之前，否则本地已经
        # 能识别的简单对话仍会承担完整 Work 流程的等待时间。
        intent = self.planner.recognize_intent(request)
        if event_callback is not None and intent != "task":
            event_callback({
                "type": "intent_recognized",
                "intent": intent,
                "route": base_plan.route.value,
            })
        if intent in {"chat", "lightweight_chat"}:
            return self._conversation_workflow(request, confirmed)

        # 单文件只读请求不需要完整的语义 Plan 和人工审批。保留文档工具的
        # 任务板与验收，但直接使用本地安全步骤，避免每次弹出确认卡片。
        if intent == "simple_document_read":
            plan = replace(base_plan, confirmed_information=tuple(confirmed))
            messages = self._build_context(request, plan, "", None)
            board = TaskBoard.from_task_plan(
                plan,
                {
                    "board_goal": plan.objective,
                    "final_deliverable": "文档读取结果与引用",
                    "tasks": [
                        {
                            "task_title": "读取指定文档",
                            "task_goal": "读取用户指定文件并提取与请求相关的内容",
                            "deliverable": "文档内容摘要与引用信息",
                            "done_when": "返回读取结果，或明确说明文件无法读取",
                            "suggested_tools": ["document_rag"],
                        },
                    ],
                },
            )
            self.coordinator.assign_board(plan, board)
            session.confirmed_information = confirmed
            session.plan_document = plan.plan_document
            session.task_board = board.to_dict()
            self.last_plan = plan
            self.last_task_board = board
            if event_callback is not None:
                event_callback({
                    "type": "routing_decided",
                    "route": plan.route.value,
                    "secondary_routes": [],
                    "available_tools": list(plan.available_tools),
                    "focus": [],
                    "reason": "简单文件读取采用本地只读路径",
                })
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
                event_callback({
                    "type": "plan_review_skipped",
                    "reason": "simple_document_read",
                })
                event_callback({"type": "task_board_created", "board": board.to_dict()})
            return plan, board, messages, False

        if semantic_runner is not None and not request.chat_only:
            # 在任何澄清卡片、计划审批或算法配置之前，先理解整句话和上下文。
            # 关键词路由只是候选类别，不足以判断“讲解聚类”和“执行聚类”。
            routing = self._semantic_json(
                semantic_runner,
                self._planning_messages(
                    "Routing",
                    {
                        "user_requirement": request.query,
                        "recent_messages": [
                            {"role": item.get("role"), "content": str(item.get("content") or "")[-4000:]}
                            for item in request.messages[-8:]
                            if item.get("role") in {"user", "assistant"}
                        ],
                        "uploaded_files": list(session.uploaded_files),
                        "current_file_schema": session.schema,
                        "confirmed_information": confirmed,
                        "capability_routes": [route.value for route in AgentRoute],
                        "capability_route": base_plan.route.value,
                        "available_tools": list(base_plan.available_tools),
                    },
                    [
                        "首先判断用户是在聊天/解释概念/咨询思路（chat），要求执行具体任务（task），还是尚不清楚需求（clarify）",
                        "结合最近对话、上传文件和字段理解代词与省略；提到聚类、情感分析、机器学习等名词不等于要求运行算法",
                        "例如“聚类是什么意思”“我不想做情感分析，只想了解方法”属于 chat；“对上传的评论列做聚类”属于 task",
                        "“帮我分析一下”应先结合上下文推断；已有文件和明确目标就继续，仍缺少对象或目的才用 clarify",
                        "chat 不需要算法配置或研究方向选择；clarify 只在 question 中给出一个贴合原话的简短问题，在对话中询问",
                        "task 的 route 和 secondary_routes 只能从 capability_routes 选择；不得返回工具权限或凭空添加能力",
                        "明确要求安装 Skill 或执行文件操作时属于 task，不能因出现礼貌问句误判成 chat",
                        "focus 只填写本次最值得优先解决的 1 到 4 个问题",
                        "不要生成 Plan，不要调用工具",
                        "严格只返回 JSON：intent（chat/task/clarify）、route（能力类别）、secondary_routes（类别列表）、focus（字符串列表）、reason（字符串）、question（clarify 时的追问，其他为空）",
                    ],
                ),
            ) or {}
            semantic_intent = routing.get("intent")
            routes = None
            if semantic_intent == "task":
                try:
                    primary_route = AgentRoute(routing.get("route"))
                    secondary_routes = routing.get("secondary_routes", [])
                    if primary_route is AgentRoute.CHAT or not isinstance(secondary_routes, list):
                        raise ValueError("执行任务需要有效的能力路由")
                    routes = (primary_route, *(AgentRoute(item) for item in secondary_routes))
                except (TypeError, ValueError):
                    semantic_intent = "clarify"
            elif semantic_intent not in ("chat", "clarify"):
                # 结构失配不等于用户要求执行；退回无工具的对话，不能用关键词
                # 兜底触发算法参数框，也不能把内部 JSON 错误交给用户解决。
                semantic_intent = "clarify"
            if event_callback is not None:
                event_callback({
                    "type": "intent_recognized",
                    "intent": semantic_intent,
                    "route": routes[0].value if routes else AgentRoute.CHAT.value,
                    "reason": str(routing.get("reason") or ""),
                })
            if semantic_intent in {"chat", "clarify"}:
                guidance = (
                    "本轮是自然语言问答：直接回应用户问题，不要求选择研究方向、算法或本地模型。"
                    if semantic_intent == "chat" else
                    "先结合最近对话和已有文件理解原话；只对仍无法确定的任务对象或目标追问一句。"
                    "若原话已是明确的知识问题，直接回答。不要展示固定研究方向菜单或内部路由错误。"
                )
                question = routing.get("question")
                if semantic_intent == "clarify" and isinstance(question, str) and question.strip():
                    guidance += "\n当前需要澄清的问题：" + question.strip()
                return self._conversation_workflow(request, confirmed, guidance)
            base_plan = self.planner.create_plan(request, capability_routes=routes)
            focus = routing.get("focus")
            base_plan = replace(
                base_plan,
                routing_focus=tuple(
                    item.strip() for item in (focus if isinstance(focus, list) else [])
                    if isinstance(item, str) and item.strip()
                )[:4],
                routing_reason=str(routing.get("reason") or ""),
            )
        plan_trace: list[dict[str, Any]] = []

        if event_callback is not None:
            event_callback({
                "type": "routing_decided",
                "route": base_plan.route.value,
                "secondary_routes": [item.value for item in (base_plan.route_decision.secondary_routes if base_plan.route_decision else ())],
                "available_tools": list(base_plan.available_tools),
                "focus": list(base_plan.routing_focus),
                "reason": base_plan.routing_reason,
            })
            event_callback({
                "type": "plan_created",
                "route": base_plan.route.value,
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "suggested_tools": list(step.suggested_tools),
                    }
                    for step in base_plan.steps
                ],
            })

        auto_mode = False
        if request.chat_only or semantic_runner is None:
            plan = replace(base_plan, confirmed_information=tuple(confirmed))
        else:
            review = self._plan_loop(
                request,
                base_plan,
                confirmed,
                semantic_runner,
                clarification_callback,
                plan_trace,
                event_callback,
                planning_context={
                    "recent_messages": request.messages[-8:],
                    "uploaded_files": list(session.uploaded_files),
                    "current_file_schema": session.schema,
                },
            )
            plan_document = self._create_plan_document(
                request, base_plan, confirmed, review, semantic_runner
            )
            plan = replace(
                base_plan,
                plan_document=plan_document,
                confirmed_information=tuple(confirmed),
                information_frame=tuple(review.get("information_frame") or ()),
                routing_focus=tuple(str(item) for item in review.get("focus", ()) if str(item).strip()),
                routing_reason=str(review.get("readiness_summary") or ""),
            )
            if event_callback is not None:
                event_callback({
                    "type": "plan_document_created",
                    "plan_id": plan.plan_id,
                    "objective": plan.objective,
                    "route": plan.route.value,
                    "plan_document": plan.plan_document,
                    "confirmed_information": list(plan.confirmed_information),
                })

            preliminary_board = TaskBoard.from_plan(plan)
            if plan_review_callback is not None and self.planner.requires_plan_review(request, plan):
                decision = plan_review_callback(plan, preliminary_board)
                if not decision or decision.get("cancelled") or decision.get("approved") is False:
                    raise RuntimeError("用户取消了 Plan，本次任务未执行")
                auto_mode = bool(decision.get("auto_mode"))
            elif plan_review_callback is not None and event_callback is not None:
                event_callback({"type": "plan_review_skipped", "reason": "low_risk_read"})

        algorithm_scheduling_supported = self._supports_algorithm_scheduling(plan)
        skip_automatic_algorithm_configuration = bool(
            auto_mode and algorithm_scheduling_supported
        )
        if plan.configuration and plan_confirmation_callback is not None and not skip_automatic_algorithm_configuration:
            if event_callback is not None:
                event_callback({"type": "configuration_required", "configuration": plan.configuration})
            configuration = plan_confirmation_callback(plan)
            if not configuration or configuration.get("cancelled"):
                raise RuntimeError("用户取消了算法配置，本次任务未执行")
        else:
            configuration = None
            if skip_automatic_algorithm_configuration and event_callback is not None:
                event_callback({
                    "type": "algorithm_configuration_skipped",
                    "reason": "全自动模式交由 LLM 算法调度器选择方法",
                })

        # 机器学习任务增加一个独立的算法调度阶段：LLM 只提出工具顺序、
        # 算法和参数，normalize_algorithm_schedule 会把它限制在本地能力边界
        # 内；真正的工具调用在 _execute_task_board 中由程序完成。
        if semantic_runner is not None and algorithm_scheduling_supported:
            plan = self._schedule_algorithms(
                request=request,
                plan=plan,
                confirmed=confirmed,
                configuration=configuration,
                semantic_runner=semantic_runner,
                event_callback=event_callback,
            )

        messages = self._build_context(
            request,
            plan,
            self.memory.recall(
                request.user_id,
                request.query,
                request.session_id,
                request.workspace_path,
            ),
            configuration,
        )

        board = self._create_task_board(plan, messages, semantic_runner) if semantic_runner is not None else TaskBoard(
            plan_document=plan.plan_document,
        )
        self.coordinator.assign_board(plan, board)
        session.confirmed_information = confirmed
        session.plan_document = plan.plan_document
        session.plan_trace.extend(plan_trace)
        session.task_board = board.to_dict()
        self.last_plan = plan
        self.last_task_board = board
        if event_callback is not None and board.tasks:
            event_callback({"type": "task_board_created", "board": board.to_dict()})
        return plan, board, messages, auto_mode

    def _conversation_workflow(
        self,
        request: AgentRequest,
        confirmed: list[dict[str, Any]],
        guidance: str = "",
    ) -> tuple[AgentPlan, TaskBoard, list[dict[str, Any]], bool]:
        """问答和初步澄清在对话里完成，不进入确认回调或任务执行。"""
        plan = self.planner.create_plan(replace(request, chat_only=True))
        plan = replace(plan, confirmed_information=tuple(confirmed), routing_reason=guidance)
        messages = self._build_context(request, plan, "", None)
        if guidance:
            messages[0]["content"] += "\n【本轮意图】\n" + guidance
        board = TaskBoard()
        self.last_plan, self.last_task_board = plan, board
        return plan, board, messages, False

    def _schedule_algorithms(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        confirmed: list[dict[str, Any]],
        configuration: dict[str, Any] | None,
        semantic_runner: SemanticRunner,
        event_callback: Callable[[dict[str, Any]], None] | None,
        planning_context: dict[str, Any] | None = None,
    ) -> AgentPlan:
        """让规划模型选择当前平台的算法工具，并生成可执行调度任务。"""

        catalog = build_algorithm_catalog(plan.available_tools)
        raw = self._semantic_json(
            semantic_runner,
            self._planning_messages(
                "Algorithm Tool Scheduling",
                {
                    "user_requirement": request.query,
                    "confirmed_information": confirmed,
                    "plan_goal": plan.objective,
                    "capability_route": plan.route.value,
                    "available_tools": list(plan.available_tools),
                    "algorithm_catalog": catalog,
                    "specialist_agents": agent_catalog.for_plan(plan.available_tools),
                    "user_algorithm_configuration": configuration or {},
                    "planning_context": planning_context or {},
                },
                [
                    "根据用户问题、当前文件字段和算法目录，规划本平台算法工具的执行顺序",
                    "只选择 algorithm_catalog 中出现的工具；不得发明新工具、代码或外部 API",
                    "一个 tasks 项只调度一个工具，按依赖顺序排列；预处理应放在建模之前",
                    "回归和分类使用 model_type，因果推断使用 method；用户已确认的算法和参数优先",
                    "scheduled_arguments 必须给出工具需要的字段名；不确定的列名不要猜，留空并让执行模型补齐",
                    "每项必须包含 task_title、task_goal、deliverable、done_when、scheduled_tool、scheduled_arguments",
                    "algorithm_decision 说明为何选择该任务类型和算法，但不要生成计算结果",
                    "严格只返回 JSON：algorithm_decision 和 tasks",
                ],
            ),
        )
        normalized = normalize_algorithm_schedule(
            raw,
            plan.available_tools,
            configuration=configuration,
        )
        if not normalized["tasks"]:
            normalized = fallback_algorithm_schedule(plan.route.value, configuration)

        scheduled = replace(
            plan,
            algorithm_schedule=tuple(normalized["tasks"]),
            algorithm_decision=dict(normalized.get("algorithm_decision") or {}),
        )
        if event_callback is not None:
            event_callback({
                "type": "algorithm_schedule_created",
                "schedule": list(scheduled.algorithm_schedule),
                "decision": scheduled.algorithm_decision,
                "warnings": list(normalized.get("errors") or []),
                "fallback_used": bool(normalized.get("fallback_used")),
            })
        return scheduled

    @staticmethod
    def _supports_algorithm_scheduling(plan: AgentPlan) -> bool:
        # 文档取证和项目文件交付不在算法目录中。混合任务保留完整任务拆分，
        # 由各专业 Agent 调用工具，避免只生成算法步骤而遗漏检索或报告写入。
        has_document_or_project = any(
            agent.id in {"document", "project"} and set(agent.primary_tools).intersection(plan.available_tools)
            for agent in agent_catalog.all()
        )
        return bool(build_algorithm_catalog(plan.available_tools)) and not has_document_or_project

    def _build_context(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        memory_context: str,
        configuration: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build extended context while keeping older custom builders usable."""
        try:
            return self.context_builder.build(
                request,
                plan,
                memory_context,
                plan_configuration=configuration,
                plan_document=plan.plan_document,
                confirmed_information=list(plan.confirmed_information),
            )
        except TypeError as exc:
            # External integrations may still implement the pre-Plan context
            # builder contract. Only fall back for an unknown keyword; never
            # hide a TypeError raised by the builder's actual implementation.
            if "unexpected keyword argument" not in str(exc):
                raise
            return self.context_builder.build(
                request,
                plan,
                memory_context,
                plan_configuration=configuration,
            )

    def _plan_loop(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        confirmed: list[dict[str, Any]],
        semantic_runner: SemanticRunner,
        clarification_callback: ClarificationCallback | None,
        trace: list[dict[str, Any]],
        event_callback: Callable[[dict[str, Any]], None] | None,
        planning_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asked: set[str] = {
            str(item.get("question"))
            for item in confirmed
            if item.get("question")
        }
        review: dict[str, Any] = {}
        for round_index in range(1, 7):
            payload = {
                "user_requirement": request.query,
                "confirmed_information": confirmed,
                "planning_context": planning_context or {},
                "routing": {
                    "capability_route": plan.route.value,
                    "available_tools": list(plan.available_tools),
                    "routing_focus": list(plan.routing_focus),
                    "specialist_agents": agent_catalog.for_plan(plan.available_tools),
                    "routing_reason": plan.routing_reason,
                },
            }
            review = None
            frame = None
            validation_error = ""
            for repair_attempt in range(2):
                stage = "Plan Review" if repair_attempt == 0 else "Repair Plan Review"
                repair_payload = {
                    **payload,
                    "previous_review": review or {},
                    "validation_error": validation_error,
                }
                try:
                    raw_review = self._consume(
                        semantic_runner,
                        self._planning_messages(
                            stage,
                            repair_payload,
                            [
                                "判断已知信息是否足以形成可执行的行动规划",
                                "信息足够时设置 plan_ready=true，并将 information_frame 返回为空列表",
                                "信息不足时设置 plan_ready=false，每轮只保留当前最影响规划的缺口，最多 3 个",
                                "不要重复询问已确认的问题，只追问会改变方案方向且无法安全假设的信息",
                                "低风险、可逆的细节留到 Plan 中作为默认假设，不要为了穷尽细节而继续提问",
                                "每个缺口必须提供 2 到 5 个可点击 options，每项包含 label、value、description；不要要求用户自由输入",
                                "适合时补充 expected_format、example 或 default_assumption，帮助用户理解选项",
                                "information_frame 只描述用户业务或任务本身的缺口，不得询问 Plan Review、JSON、information_frame 或任何系统内部输出格式",
                                "此阶段不调用工具、不生成最终产物",
                                "严格返回 JSON：plan_goal、final_deliverable、readiness_summary、information_frame、plan_ready；缺口项必须带 options",
                            ],
                        ),
                    )
                    parsed_review = self._parse_json(raw_review)
                    if parsed_review is None:
                        raise ValueError("Plan Review 必须返回 JSON 对象")
                    review = normalize_plan_review(parsed_review)
                    frame = normalize_information_frame(review, round_index, asked)
                    break
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    validation_error = str(exc)
                    frame = None
            if frame is None:
                # Plan Review 是模型辅助判断，不应因为格式偶发失配而让整个任务无法开始。
                # 这里不伪造用户答案，只回退到本地已有 Plan，并把原因写入进度日志，
                # 后续仍然会经过用户的“请求批准/帮我批准”和任务验收。
                review = self._fallback_plan_review(plan, validation_error)
                frame = []
                if event_callback is not None:
                    event_callback({
                        "type": "plan_review_fallback",
                        "reason": validation_error or "规划模型未返回可识别的结构",
                    })
            trace.append({"round": round_index, "review": review, "frame": frame})
            if event_callback is not None:
                event_callback({
                    "type": "plan_reviewed",
                    "round": round_index,
                    "plan_ready": review.get("plan_ready") is True,
                    "information_frame": frame,
                    "readiness_summary": review.get("readiness_summary", ""),
                })
            if review.get("plan_ready") is True:
                review["information_frame"] = []
                return review
            if clarification_callback is None:
                raise RuntimeError("规划仍缺少关键信息，请补充后再执行")
            current = next_missing_info(frame)
            while current is not None:
                answer = clarification_callback(current)
                if answer is None or not str(answer).strip():
                    raise RuntimeError("用户未完成必要的信息补充")
                answered = apply_user_answer(frame, current["info_id"], str(answer))
                confirmed.append(answered)
                asked.add(str(answered["question"]))
                if event_callback is not None:
                    event_callback({"type": "information_filled", "information": answered})
                current = next_missing_info(frame)
        raise RuntimeError("Plan Loop 超过最大轮数 6")

    @staticmethod
    def _fallback_plan_review(plan: AgentPlan, reason: str) -> dict[str, Any]:
        """模型结构连续失配时，返回不新增用户约束的本地安全 Plan Review。"""
        deliverable = ", ".join(plan.expected_outputs) or "按用户需求形成可交付结果"
        return {
            "plan_goal": plan.objective,
            "final_deliverable": deliverable,
            "readiness_summary": (
                "规划模型未按约定返回结构，已采用本地已有计划和低风险默认假设；"
                "未新增或猜测用户约束。"
            ),
            "information_frame": [],
            "plan_ready": True,
            "fallback_used": True,
            "fallback_reason": reason,
        }

    def _create_plan_document(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        confirmed: list[dict[str, Any]],
        review: dict[str, Any],
        semantic_runner: SemanticRunner,
        planning_context: dict[str, Any] | None = None,
    ) -> str:
        result = self._semantic_text(
            semantic_runner,
            self._planning_messages(
                "Create Plan Document",
                {
                    "user_requirement": request.query,
                    "confirmed_information": confirmed,
                    "plan_review": review,
                    "planning_context": planning_context or {},
                    "capability_route": plan.route.value,
                    "available_tools": list(plan.available_tools),
                    "specialist_agents": agent_catalog.for_plan(plan.available_tools),
                    "routing_focus": list(plan.routing_focus),
                    "routing_reason": plan.routing_reason,
                },
                [
                    "生成一份供用户审阅的 Markdown Plan 计划书",
                    "说明执行目标、最终产物、已确认边界、实施思路、执行顺序和完成标准",
                    "根据 specialist_agents 说明各专业 Agent 的分工，以及前序结果如何交给后续 Agent",
                    "明确区分这份 Plan 与后续真正要生成的最终产物",
                    "未确定内容必须显式标记，不要编造用户约束",
                    "只描述后续如何行动，不直接生成最终分析结果或最终文档正文",
                ],
            ),
        )
        return result.strip() if result.strip() else plan.as_prompt()

    def _create_task_board(
        self,
        plan: AgentPlan,
        messages: list[dict[str, Any]],
        semantic_runner: SemanticRunner | None,
    ) -> TaskBoard:
        if semantic_runner is None or plan.route.value == "chat":
            return TaskBoard.from_plan(plan)
        if plan.algorithm_schedule:
            # 调度模型已经输出了带验收标准的任务，因此不再让第二个语义
            # 拆分阶段重新猜测工具；TaskBoard 只负责分配 ID 和维护状态。
            return TaskBoard.from_task_plan(
                plan,
                {
                    "board_goal": plan.objective,
                    "final_deliverable": ", ".join(plan.expected_outputs) or "算法分析结果",
                    "tasks": list(plan.algorithm_schedule),
                },
            )
        task_plan = self._semantic_json(
            semantic_runner,
            self._planning_messages(
                "Plan to Task Board",
                {
                    "plan_document": plan.plan_document,
                    "plan_goal": plan.objective,
                    "final_deliverable": ", ".join(plan.expected_outputs),
                    "available_tools": list(plan.available_tools),
                    "specialist_agents": agent_catalog.for_plan(plan.available_tools),
                },
                [
                    "根据 Plan 生成初始动态任务清单",
                    "保持 Plan 的交付目标，不把未来现实活动的结果当成当前交付物",
                    "每个任务只负责一个可由模型生成或当前工具执行的结果槽位",
                    "每个任务交给一个专业 Agent，跨专业工作拆成顺序任务；共享数据加载和统计工具可作为辅助",
                    "输出 tasks，每项必须包含 task_title、task_goal、deliverable、done_when",
                    "done_when 必须能根据当前任务结果检查",
                    "如果任务需要工具，补充 suggested_tools，并且只能从 available_tools 中选择；没有必要时返回空列表",
                    "只拆分计划，不执行任务，不填写实际结果",
                    "严格只返回 JSON 对象，必须包含 board_goal、final_deliverable、tasks",
                ],
            ),
        )
        if task_plan is None:
            return TaskBoard.from_plan(plan)
        raw_tasks = task_plan.get("tasks")
        for raw_task in raw_tasks if isinstance(raw_tasks, list) else ():
            if isinstance(raw_task, dict):
                # 只有算法规范化器可以绑定直接执行；普通任务拆分不能绕过它。
                raw_task.pop("scheduled_tool", None)
                raw_task.pop("algorithm_tool", None)
                raw_task.pop("scheduled_arguments", None)
                raw_task.pop("tool_arguments", None)
        try:
            return TaskBoard.from_task_plan(plan, task_plan)
        except ValueError:
            return TaskBoard.from_plan(plan)

    def _execute_task_board(
        self,
        plan: AgentPlan,
        board: TaskBoard,
        base_messages: list[dict[str, Any]],
        model_runner: ModelRunner,
        semantic_runner: SemanticRunner | None,
        scope: ExecutionScope,
        run_id: str = "",
        replan_count: int = 0,
        auto_mode: bool = False,
        clarification_callback: ClarificationCallback | None = None,
        plan_review_callback: PlanReviewCallback | None = None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        replan_count = max(0, int(replan_count))
        while True:
            ready = board.ready_tasks()
            if not ready:
                break
            task = ready[0]
            task_id = task.id
            board.start(task_id)
            scope.current_task_id = task_id
            agent = self.coordinator.assign(plan, task)
            scope.current_agent_id = agent.id
            scope.current_agent_name = agent.name
            scope.allowed_tools = self.coordinator.allowed_tools(plan, task, agent)
            scope.task_observations[task_id] = list(task.observations)
            scope.record("agent_assigned", task_id=task_id, allowed_tools=sorted(scope.allowed_tools))
            scope.record("task_started", task=task.to_dict())
            self._checkpoint_scope(
                run_id,
                scope,
                board,
                base_messages,
                auto_mode,
                replan_count,
            )

            repair_tool = self._repair_tool_for_task(plan, task)
            resume_verification = bool(task.resume_verification and task.result)
            task.resume_verification = False
            if resume_verification:
                generated = [{"role": "assistant", "content": str(task.result)}]
                scope.record("task_verification_resumed", task_id=task_id)
            elif repair_tool and self.coordinator.can_repair(plan, agent, repair_tool):
                # 修复任务不是普通任务重试：直接走确定性的结果修复工具，
                # 不再让主 Agent 重新调用文本向量或 BERT 模型。
                scope.allowed_tools.add(repair_tool)
                repair_kind = "聚类结果" if repair_tool == "repair_text_clustering" else "情感分析结果"
                scope.record(
                    "task_repair_started",
                    task_id=task_id,
                    tool=repair_tool,
                    repair_kind=repair_kind,
                )
                cached = self._cached_observation(task, repair_tool, {}) if scope.recovery_mode else None
                repair_result = cached if cached is not None else self.executor.execute(repair_tool)
                if cached is not None:
                    scope.record("tool_result_reused", task_id=task_id, tool=repair_tool)
                scope.record(
                    "task_repair_finished",
                    task_id=task_id,
                    tool=repair_tool,
                    success=bool(repair_result.get("success")),
                )
                generated = [{
                    "role": "assistant",
                    "content": str(
                        repair_result.get("summary_for_agent")
                        or repair_result.get("message")
                        or repair_result.get("error")
                        or "规则修复未返回说明"
                    ),
                }]
            elif task.scheduled_tool:
                # 调度任务由程序直接触发确定性工具；LLM 后续只负责把
                # 结构化结果翻译成用户可读文本，不能再次重复调用算法。
                cached = self._cached_observation(
                    task,
                    task.scheduled_tool,
                    task.scheduled_arguments,
                ) if scope.recovery_mode else None
                scheduled_result = cached if cached is not None else self.executor.execute(
                    task.scheduled_tool,
                    **task.scheduled_arguments,
                )
                if cached is not None:
                    scope.record("tool_result_reused", task_id=task_id, tool=task.scheduled_tool)
                scope.record(
                    "algorithm_tool_scheduled",
                    task_id=task_id,
                    tool=task.scheduled_tool,
                    arguments=task.scheduled_arguments,
                    success=bool(scheduled_result.get("success")),
                )
                scope.allowed_tools = set()
                generated = self._consume(
                    model_runner,
                    self._task_messages(
                        plan,
                        board,
                        task,
                        base_messages,
                        current_observations=scope.task_observations.get(task_id, []),
                    ),
                )
            else:
                generated = self._consume(model_runner, self._task_messages(plan, board, task, base_messages))
            task.result = self._response_text(generated)
            scope.record("task_draft_created", task_id=task_id)
            task.status = TaskStatus.VERIFYING
            self._checkpoint_scope(
                run_id,
                scope,
                board,
                base_messages,
                auto_mode,
                replan_count,
            )

            observations = scope.task_observations.get(task_id, [])
            task.result = self._append_structured_observation_summary(task.result, observations)
            review = self._semantic_json(
                semantic_runner,
                self._check_messages(plan, task, observations),
            ) if semantic_runner is not None else None
            if review is None:
                review = {
                    "passed": bool(task.result.strip() or observations),
                    "feedback": "使用程序侧非空结果兜底检查",
                }
            verification = self.verifier.verify(task, review, generated, observations)
            scope.record(
                "task_checked",
                task_id=task_id,
                passed=verification.passed,
                feedback=verification.feedback,
            )
            blocked = False
            if verification.passed:
                board.complete(task_id, verification)
                scope.record("task_completed", task=task.to_dict())
            elif task.attempts < task.max_attempts:
                board.retry(task_id, verification.feedback)
                scope.record("task_retry", task=task.to_dict(), feedback=verification.feedback)
            else:
                board.block(task_id, verification.feedback)
                scope.record("task_blocked", task=task.to_dict(), feedback=verification.feedback)
                blocked = True

            self._checkpoint_scope(
                run_id,
                scope,
                board,
                base_messages,
                auto_mode,
                replan_count,
            )

            reflection: dict[str, Any] | None = None
            if semantic_runner is not None:
                reflection = self._semantic_json(
                    semantic_runner,
                    self._reflection_messages(plan, board, task, verification),
                )
                added = board.apply_reflection(reflection or {})
                for added_task in added:
                    self.coordinator.assign(plan, added_task)
                if reflection and reflection.get("action") == "replan_plan":
                    scope.record("replan_requested", reason=reflection.get("reason", ""))
                if added:
                    scope.record(
                        "task_board_updated",
                        reason=reflection.get("reason", "Reflection 增加后续任务"),
                        added_tasks=[item.to_dict() for item in added],
                        board=board.to_dict(),
                    )

            should_replan = blocked or bool(
                reflection and reflection.get("action") == "replan_plan"
            )
            if should_replan and semantic_runner is not None and replan_count < 1:
                if scope.budget_guard is not None:
                    scope.budget_guard("replan")
                try:
                    replanned = self._replan_after_failure(
                        plan=plan,
                        board=board,
                        failed_task=task,
                        verification=verification,
                        semantic_runner=semantic_runner,
                        scope=scope,
                        auto_mode=auto_mode,
                        clarification_callback=clarification_callback,
                        plan_review_callback=plan_review_callback,
                        plan_confirmation_callback=plan_confirmation_callback,
                    )
                except Exception as exc:
                    scope.record("replan_failed", reason=str(exc))
                    replanned = None
                if replanned is not None:
                    plan, board, base_messages, auto_mode = replanned
                    scope.plan = plan
                    replan_count += 1
                    scope.checkpoint_callback = lambda: self._checkpoint_scope(
                        run_id, scope, board, base_messages, auto_mode, replan_count
                    )
                    self._checkpoint_scope(
                        run_id,
                        scope,
                        board,
                        base_messages,
                        auto_mode,
                        replan_count,
                    )
                    continue

            if blocked:
                break
            if scope.event_callback is not None:
                scope.record("task_board_state", board=board.to_dict())

        if any(task.status is TaskStatus.BLOCKED for task in board.tasks.values()):
            scope.record("convergence_blocked", board=board.to_dict())
        else:
            scope.record("convergence_completed", board=board.to_dict())
        self.last_task_board = board
        self._checkpoint_scope(
            run_id,
            scope,
            board,
            base_messages,
            auto_mode,
            replan_count,
        )
        scope.current_task_id = None
        scope.current_agent_id = ""
        scope.current_agent_name = ""
        scope.allowed_tools = set()
        return [{"role": "assistant", "content": self._assemble_results(board)}]

    @staticmethod
    def _cached_observation(
        task: TaskItem,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """恢复时复用同一任务已落盘的成功工具结果，避免重复副作用。"""
        for observation in reversed(task.observations):
            if str(observation.get("tool") or "") != str(tool):
                continue
            observed_arguments = observation.get("params")
            try:
                same_arguments = json.dumps(
                    observed_arguments or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ) == json.dumps(
                    arguments or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            except (TypeError, ValueError):
                same_arguments = observed_arguments == arguments
            if not same_arguments:
                continue
            result = observation.get("result")
            if not isinstance(result, dict):
                continue
            if result.get("success") and not result.get("requires_confirmation"):
                return result
        return None

    @staticmethod
    def _repair_tool_for_task(plan: AgentPlan, task: TaskItem) -> str | None:
        """判断任务是否应进入结果修复路径，而不是再次调用原始模型。"""
        text = " ".join(
            str(value or "")
            for value in (task.title, task.description, task.task_goal, task.deliverable, task.done_when)
        ).lower()
        repair_markers = (
            "修复", "补全", "重建", "截断", "不完整", "repair", "retry",
        )
        is_retry_or_repair = task.attempts > 1 or bool(getattr(plan, "is_replan", False)) or any(
            marker in text for marker in repair_markers
        )
        if not is_retry_or_repair:
            return None

        cluster_markers = (
            "聚类", "主题簇", "cluster", "kmeans", "minibatch", "agglomerative",
            "dbscan", "gmm", "spectral", "birch", "affinity", "meanshift",
        )
        sentiment_markers = (
            "情感分析", "情绪分析", "情感分布", "sentiment",
        )
        if any(marker in text for marker in cluster_markers):
            return "repair_text_clustering"
        if any(marker in text for marker in sentiment_markers):
            return "repair_sentiment_analysis"
        return None

    def _replan_after_failure(
        self,
        plan: AgentPlan,
        board: TaskBoard,
        failed_task: TaskItem,
        verification: Any,
        semantic_runner: SemanticRunner,
        scope: ExecutionScope,
        auto_mode: bool,
        clarification_callback: ClarificationCallback | None,
        plan_review_callback: PlanReviewCallback | None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None,
    ) -> tuple[AgentPlan, TaskBoard, list[dict[str, Any]], bool] | None:
        """重新进入一次 Plan Loop，给失败任务一个新的执行路径。"""
        request = scope.request
        reason = str(verification.feedback or "当前任务未通过验收")
        planning_context = {
            "reason": reason,
            "failed_task": failed_task.to_dict(),
            "verification": {
                "passed": bool(verification.passed),
                "feedback": reason,
            },
            "completed_results": [
                {
                    "task_id": item.id,
                    "task_title": item.title,
                    "result": item.result,
                }
                for item in board.tasks.values()
                if item.status is TaskStatus.COMPLETED
            ],
        }
        scope.record(
            "replan_started",
            reason=reason,
            failed_task_id=failed_task.id,
            attempt=1,
        )
        session = session_store.get(request.session_id)
        confirmed = list(session.confirmed_information)
        trace: list[dict[str, Any]] = []
        review = self._plan_loop(
            request,
            plan,
            confirmed,
            semantic_runner,
            clarification_callback,
            trace,
            scope.event_callback,
            planning_context=planning_context,
        )
        plan_document = self._create_plan_document(
            request,
            plan,
            confirmed,
            review,
            semantic_runner,
            planning_context=planning_context,
        )
        replanned = replace(
            plan,
            objective=str(review.get("plan_goal") or plan.objective),
            plan_id=f"{plan.plan_id}-r1",
            plan_document=plan_document,
            confirmed_information=tuple(confirmed),
            information_frame=tuple(review.get("information_frame") or ()),
            routing_focus=tuple(str(item) for item in review.get("focus", ()) if str(item).strip()),
            routing_reason=str(review.get("readiness_summary") or ""),
            is_replan=True,
            replan_reason=reason,
        )
        if scope.event_callback is not None:
            scope.record(
                "plan_document_created",
                plan_id=replanned.plan_id,
                objective=replanned.objective,
                route=replanned.route.value,
                plan_document=replanned.plan_document,
                confirmed_information=list(replanned.confirmed_information),
                is_replan=True,
                replan_reason=reason,
            )

        preliminary_board = TaskBoard.from_plan(replanned)
        next_auto_mode = auto_mode
        if (
            not auto_mode
            and plan_review_callback is not None
            and self.planner.requires_plan_review(request, replanned)
        ):
            decision = plan_review_callback(replanned, preliminary_board)
            if not decision or decision.get("cancelled") or decision.get("approved") is False:
                raise RuntimeError("用户取消了重新规划，本次任务未执行")
            next_auto_mode = bool(decision.get("auto_mode"))
        elif not auto_mode and plan_review_callback is not None and scope.event_callback is not None:
            scope.record("plan_review_skipped", reason="low_risk_read")

        algorithm_scheduling_supported = self._supports_algorithm_scheduling(replanned)
        skip_automatic_algorithm_configuration = bool(
            next_auto_mode and algorithm_scheduling_supported
        )
        if replanned.configuration and plan_confirmation_callback is not None and not skip_automatic_algorithm_configuration:
            if scope.event_callback is not None:
                scope.record("configuration_required", configuration=replanned.configuration)
            configuration = plan_confirmation_callback(replanned)
            if not configuration or configuration.get("cancelled"):
                raise RuntimeError("用户取消了算法配置，本次任务未执行")
        else:
            configuration = None
            if skip_automatic_algorithm_configuration:
                scope.record(
                    "algorithm_configuration_skipped",
                    reason="全自动模式交由 LLM 算法调度器选择方法",
                )

        if semantic_runner is not None and algorithm_scheduling_supported:
            replanned = self._schedule_algorithms(
                request=request,
                plan=replanned,
                confirmed=confirmed,
                configuration=configuration,
                semantic_runner=semantic_runner,
                event_callback=scope.event_callback,
                planning_context=planning_context,
            )

        messages = self._build_context(
            request,
            replanned,
            self.memory.recall(
                request.user_id,
                request.query,
                request.session_id,
                request.workspace_path,
            ),
            configuration,
        )
        new_board = self._create_task_board(replanned, messages, semantic_runner)
        self.coordinator.assign_board(replanned, new_board)
        session.confirmed_information = confirmed
        session.plan_document = replanned.plan_document
        session.plan_trace.extend(trace)
        session.task_board = new_board.to_dict()
        self.last_plan = replanned
        self.last_task_board = new_board
        scope.record(
            "replan_completed",
            plan_id=replanned.plan_id,
            board=new_board.to_dict(),
        )
        if scope.event_callback is not None:
            scope.record("task_board_created", board=new_board.to_dict(), is_replan=True)
        return replanned, new_board, messages, next_auto_mode

    @staticmethod
    def _task_messages(
        plan: AgentPlan,
        board: TaskBoard,
        task: TaskItem,
        base_messages: list[dict[str, Any]],
        current_observations: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        completed = [
            {
                "task_id": item.id,
                "task_title": item.title,
                "agent_id": item.agent_id,
                "deliverable": item.deliverable,
                "result": item.result,
                "artifacts": list(item.artifacts),
                "evidence": list(item.evidence),
            }
            for item in board.tasks.values()
            if item.status is TaskStatus.COMPLETED
        ]
        payload = {
            "current_task": {
                "task_id": task.id,
                "agent_id": task.agent_id,
                "agent_name": task.agent_name,
                "task_title": task.title,
                "task_goal": task.task_goal,
                "deliverable": task.deliverable,
                "done_when": task.done_when,
                "scheduled_tool": task.scheduled_tool,
                "scheduled_arguments": task.scheduled_arguments,
            },
            "completed_results": completed,
            "previous_feedback": task.feedback[-1] if task.feedback else "",
        }
        if current_observations:
            payload["scheduled_tool_observations"] = current_observations
        instruction = (
            "【当前单任务执行】\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "只完成 current_task，不扩展其他任务；只使用 Plan、已确认信息和 completed_results 中已有的信息。"
            "如果存在 previous_feedback，优先补齐该缺口。只输出当前任务的结果，不要声称未来的现实动作已经执行。"
            "工具返回的 summary_for_agent、cluster_distribution、cluster_profiles 等结构化字段优先于长篇原始表格；"
            "聚类任务必须覆盖返回的每一个主题簇及其分布，并根据 representative_texts 概括特征，"
            "不要在表格中途停止，也不要为了展示全部原始行而输出超长内容。"
            f"\n【Plan 文书】\n{plan.plan_document}"
        )
        agent = agent_catalog.get(task.agent_id)
        if agent is not None:
            instruction = f"【专业 Agent 职责】\n{agent.as_prompt()}\n\n{instruction}"
        if task.scheduled_tool:
            instruction += (
                "\n本任务的算法工具已经由 Runtime 按 scheduled_tool 执行；"
                "本轮禁止再次调用任何工具，只根据 scheduled_tool_observations 解释成功结果或错误。"
            )
        elif task.scheduled_arguments:
            instruction += (
                "\n调度器已经确定了部分工具参数；调用 suggested_tools 时必须保留"
                "scheduled_arguments 中的值，只补齐缺失的字段。"
            )
        return AgentRuntime._append_system_instruction(base_messages, instruction)

    @staticmethod
    def _append_system_instruction(
        messages: list[dict[str, Any]],
        instruction: str,
    ) -> list[dict[str, Any]]:
        """合并系统提示，满足 OpenAI-compatible 消息协议。"""
        system_parts: list[str] = []
        conversation: list[dict[str, Any]] = []
        for raw in messages:
            message = dict(raw)
            if str(message.get("role", "")).strip().lower() == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    system_parts.append(content)
            else:
                conversation.append(message)
        system_parts.append(str(instruction).strip())
        return [
            {"role": "system", "content": "\n\n".join(system_parts)},
            *conversation,
        ]

    @staticmethod
    def _check_messages(
        plan: AgentPlan,
        task: TaskItem,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return AgentRuntime._planning_messages(
            "Check Task",
            {
                "task": task.to_dict(),
                "draft": task.result,
                "tool_observations": observations,
                "plan_document": plan.plan_document,
            },
            [
                "检查当前任务结果是否满足 done_when",
                "只返回 JSON 对象，必须包含 passed 布尔值和非空 feedback",
                "passed=true 时说明通过依据，passed=false 时只指出最重要的一个缺口",
                "如果工具观察已经提供完整的结构化结果（例如每个主题簇的分布和代表文本），即使主模型草稿的长表格被截断，也应判定为通过",
                "不要修改任务状态，不要检查其他未完成任务",
            ],
        )

    @staticmethod
    def _reflection_messages(
        plan: AgentPlan,
        board: TaskBoard,
        task: TaskItem,
        verification: Any,
    ) -> list[dict[str, Any]]:
        return AgentRuntime._planning_messages(
            "Reflect Task Board",
            {
                "plan_document": plan.plan_document,
                "current_task": task.to_dict(),
                "verification": {
                    "passed": verification.passed,
                    "feedback": verification.feedback,
                },
                "task_board": board.to_dict(),
            },
            [
                "判断后续任务是否需要调整",
                "通常返回 action=continue；只有新结果明确暴露新问题时才使用 action=add_tasks",
                "如果需要新增任务，new_tasks 每项包含 task_title、task_goal、deliverable、done_when",
                "如果整体目标或范围已经变化，返回 action=replan_plan，但不要自行改写 Plan",
                "不要把未来外部动作标记为已经完成",
            ],
        )

    @staticmethod
    def _planning_messages(title: str, payload: dict[str, Any], instructions: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是 Academic Agent 的语义规划与检查模块。"
                    "当前阶段不调用工具，严格遵守输出契约。\n"
                    f"【阶段】{title}\n"
                    "【约束】\n- " + "\n- ".join(instructions)
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]

    @staticmethod
    def _consume(runner: ModelRunner, messages: list[dict[str, Any]]) -> Any:
        generated = runner(messages)
        if hasattr(generated, "__iter__") and not isinstance(generated, (list, dict, str, bytes)):
            final = None
            for item in generated:
                final = item
            return final
        if isinstance(generated, list):
            # A few older integrations return ``[[message, ...]]`` instead
            # of the canonical ``[message, ...]`` shape.
            while generated and isinstance(generated[0], list):
                generated = generated[-1]
        return generated

    @staticmethod
    def _tracked_runner(
        runner: Callable[[list[dict[str, Any]]], Any],
        recorder: RunEventRecorder,
        channel: str,
    ) -> Callable[[list[dict[str, Any]]], Any]:
        """给旧模型适配器套上统一的调用计数和运行事件。"""
        def wrapped(messages: list[dict[str, Any]]) -> Any:
            recorder.before_model()
            started = time.monotonic()
            recorder.emit({"type": "model_turn_started", "channel": channel})
            try:
                result = runner(messages)
            except Exception as exc:
                recorder.emit({
                    "type": "model_turn_failed",
                    "channel": channel,
                    "error": str(exc),
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                })
                raise
            recorder.emit({
                "type": "model_turn_finished",
                "channel": channel,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            })
            return result

        return wrapped

    def _semantic_json(self, runner: SemanticRunner, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        try:
            return self._parse_json(self._consume(runner, messages))
        except HarnessBudgetExceeded:
            raise
        except Exception:
            return None

    def _semantic_text(self, runner: SemanticRunner, messages: list[dict[str, Any]]) -> str:
        try:
            return self._response_text(self._consume(runner, messages))
        except HarnessBudgetExceeded:
            raise
        except Exception:
            return ""

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, list):
            # Some legacy adapters wrap one assistant message in an extra
            # list (for example ``[[{"role": "assistant", ...}]]``).
            # Normalize that shape before using the shared response helper.
            while response and isinstance(response[0], list):
                response = response[-1]
            return assistant_content(response)
        if isinstance(response, dict):
            return str(response.get("content") or response.get("text") or "")
        return str(response or "")

    @classmethod
    def _parse_json(cls, response: Any) -> dict[str, Any] | None:
        text = cls._response_text(response).strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return None
            value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None

    @staticmethod
    def _assemble_results(board: TaskBoard) -> str:
        completed = [task for task in board.tasks.values() if task.status is TaskStatus.COMPLETED]
        sections = [
            f"## {task.title}\n{str(task.result or '').strip()}"
            for task in completed if str(task.result or '').strip()
        ]
        if any(task.status is TaskStatus.BLOCKED for task in board.tasks.values()):
            blocked = next(task for task in board.tasks.values() if task.status is TaskStatus.BLOCKED)
            header = f"任务未全部完成：{blocked.title}\n原因：{blocked.feedback[-1] if blocked.feedback else '达到最大尝试次数'}"
            return header + ("\n\n" + "\n\n".join(sections) if sections else "")
        return "\n\n".join(sections) or "没有形成可交付结果。"

    @staticmethod
    def _append_structured_observation_summary(
        draft: str,
        observations: list[dict[str, Any]],
    ) -> str:
        """把关键结构化工具证据补到草稿末尾，避免长文本截断丢失结论。"""
        clustering = [
            item.get("result") or {}
            for item in observations
            if item.get("tool") == "text_clustering"
            and isinstance(item.get("result"), dict)
            and bool((item.get("result") or {}).get("success"))
        ]
        if not clustering:
            return draft
        result = clustering[-1]
        profiles = result.get("cluster_profiles")
        if not isinstance(profiles, list) or not profiles:
            return draft

        lines = ["【聚类结构化摘要】"]
        summary = str(result.get("summary_for_agent") or "").strip()
        if summary:
            lines.append(summary)
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            cluster_id = profile.get("cluster_id", "?")
            count = profile.get("segment_count", "?")
            percentage = profile.get("percentage")
            share = f"（{float(percentage):.2%}）" if isinstance(percentage, (int, float)) else ""
            samples = "；".join(
                str(text).replace("\n", " ").strip()
                for text in (profile.get("representative_texts") or [])
                if str(text).strip()
            )
            lines.append(f"簇{cluster_id}：{count} 条{share}；代表文本：{samples or '无'}")
        structured = "\n".join(lines)
        if structured in draft:
            return draft
        return f"{draft.rstrip()}\n\n{structured}".strip()

    def _finish(self, request: AgentRequest, final_response: Any, scope: ExecutionScope) -> None:
        self.last_events = list(scope.events)
        content = assistant_content(final_response)
        self.memory.commit(
            user_id=request.user_id,
            user_text=request.query,
            assistant_text=content,
            session_id=request.session_id,
            workspace_path=request.workspace_path,
        )
