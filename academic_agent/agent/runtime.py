"""Plan Loop → Task Board → Convergence Loop 的 Agent Runtime。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
import json
import re
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
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.response import assistant_content, attach_confirmation_link
from academic_agent.agent.session import session_store
from academic_agent.agent.types import (
    AgentPlan,
    AgentRequest,
    ExecutionScope,
    TaskBoard,
    TaskItem,
    TaskStatus,
)
from academic_agent.infrastructure.workspace_manager import workspace_manager


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
    ) -> None:
        self.planner = planner or TaskPlanner()
        self.memory = memory or MemoryCoordinator()
        self.context_builder = context_builder or AgentContextBuilder()
        self.executor = executor or tool_executor
        self.verifier = verifier or TaskVerifier()
        self.last_plan: AgentPlan | None = None
        self.last_task_board: TaskBoard | None = None
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
        model_runner: ModelRunner,
        semantic_runner: SemanticRunner | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        clarification_callback: ClarificationCallback | None = None,
        plan_review_callback: PlanReviewCallback | None = None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None = None,
    ) -> Any:
        result = self._run_internal(
            request,
            model_runner,
            semantic_runner=semantic_runner,
            event_callback=event_callback,
            clarification_callback=clarification_callback,
            plan_review_callback=plan_review_callback,
            plan_confirmation_callback=plan_confirmation_callback,
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
    ) -> Iterator[Any]:
        result = self._run_internal(
            request,
            model_runner,
            semantic_runner=semantic_runner,
            event_callback=event_callback,
            clarification_callback=clarification_callback,
            plan_review_callback=plan_review_callback,
            plan_confirmation_callback=plan_confirmation_callback,
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
    ) -> Any:
        try:
            plan, board, messages, auto_mode = self._prepare_workflow(
                request,
                semantic_runner=semantic_runner,
                event_callback=event_callback,
                clarification_callback=clarification_callback,
                plan_review_callback=plan_review_callback,
                plan_confirmation_callback=plan_confirmation_callback,
            )
        except Exception as exc:
            if event_callback is not None:
                event_callback({"type": "plan_failed", "error": str(exc)})
            if "本次任务未执行" in str(exc):
                message = str(exc).rstrip("。") + "。"
                return {"error": message}
            return {"error": f"Plan 阶段未完成：{exc}"}

        scope = ExecutionScope(request=request, plan=plan, event_callback=event_callback)
        pending_before = set(workspace_manager.pending_operations)
        with self.executor.bind(scope):
            if request.chat_only or semantic_runner is None:
                final_response = self._consume(model_runner, messages)
            else:
                final_response = self._execute_task_board(
                    plan,
                    board,
                    messages,
                    model_runner,
                    semantic_runner,
                    scope,
                    auto_mode=auto_mode,
                    clarification_callback=clarification_callback,
                    plan_review_callback=plan_review_callback,
                    plan_confirmation_callback=plan_confirmation_callback,
                )
        attach_confirmation_link(final_response, pending_before)
        self._finish(request, final_response, scope)
        return final_response if final_response is not None else {"error": "无响应"}

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

        # Work 模式下如果本地路由无法确认学术方向，先用固定的学术选项
        # 收敛意图，再调用语义规划模型。这样“帮我分析一下”不会直接
        # 变成一个无边界的通用 Chat/任务板请求。
        if (
            semantic_runner is not None
            and not request.chat_only
            and self.planner.requires_academic_intent_clarification(request)
        ):
            if clarification_callback is None:
                raise RuntimeError("当前需求不够明确，请先选择文本挖掘、机器学习、数据分析或学术文献方向")
            intent_info = self.planner.academic_intent_question()
            if event_callback is not None:
                event_callback({"type": "clarification_required", "information": intent_info})
            answer = clarification_callback(intent_info)
            if answer is None or not str(answer).strip():
                raise RuntimeError("用户未选择学术研究方向，本次任务未执行")
            answered = apply_user_answer([intent_info], "R0-F1", str(answer))
            answered["source"] = "user_selection"
            confirmed.append(answered)
            if event_callback is not None:
                event_callback({"type": "information_filled", "information": answered})
            # 选项值只作为确定性路由提示，不能扩大工具权限；可用工具仍由
            # TaskPlanner 根据本地技能目录重新计算。
            base_plan = self.planner.create_plan(request, routing_hint=str(answer))

        if semantic_runner is not None and not request.chat_only:
            routing = self._semantic_json(
                semantic_runner,
                self._planning_messages(
                    "Routing",
                    {
                        "user_requirement": request.query,
                        "capability_route": base_plan.route.value,
                        "available_tools": list(base_plan.available_tools),
                    },
                    [
                        "判断用户当前最关注的讨论重点，而不是重新选择工具权限",
                        "focus 只填写本次最值得优先解决的 1 到 4 个问题",
                        "不要生成 Plan，不要调用工具",
                        "严格只返回 JSON：focus（字符串列表）和 reason（字符串）",
                    ],
                ),
            ) or {}
            base_plan = replace(
                base_plan,
                routing_focus=tuple(str(item) for item in routing.get("focus", ()) if str(item).strip()),
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
            if plan_review_callback is not None:
                decision = plan_review_callback(plan, preliminary_board)
                if not decision or decision.get("cancelled") or decision.get("approved") is False:
                    raise RuntimeError("用户取消了 Plan，本次任务未执行")
                auto_mode = bool(decision.get("auto_mode"))

        if plan.configuration and plan_confirmation_callback is not None:
            if event_callback is not None:
                event_callback({"type": "configuration_required", "configuration": plan.configuration})
            configuration = plan_confirmation_callback(plan)
            if not configuration or configuration.get("cancelled"):
                raise RuntimeError("用户取消了算法配置，本次任务未执行")
        else:
            configuration = None

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
        session.confirmed_information = confirmed
        session.plan_document = plan.plan_document
        session.plan_trace.extend(plan_trace)
        session.task_board = board.to_dict()
        self.last_plan = plan
        self.last_task_board = board
        if event_callback is not None and board.tasks:
            event_callback({"type": "task_board_created", "board": board.to_dict()})
        return plan, board, messages, auto_mode

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
                    "routing_focus": list(plan.routing_focus),
                    "routing_reason": plan.routing_reason,
                },
                [
                    "生成一份供用户审阅的 Markdown Plan 计划书",
                    "说明执行目标、最终产物、已确认边界、实施思路、执行顺序和完成标准",
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
        task_plan = self._semantic_json(
            semantic_runner,
            self._planning_messages(
                "Plan to Task Board",
                {
                    "plan_document": plan.plan_document,
                    "plan_goal": plan.objective,
                    "final_deliverable": ", ".join(plan.expected_outputs),
                    "available_tools": list(plan.available_tools),
                },
                [
                    "根据 Plan 生成初始动态任务清单",
                    "保持 Plan 的交付目标，不把未来现实活动的结果当成当前交付物",
                    "每个任务只负责一个可由模型生成或当前工具执行的结果槽位",
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
        auto_mode: bool = False,
        clarification_callback: ClarificationCallback | None = None,
        plan_review_callback: PlanReviewCallback | None = None,
        plan_confirmation_callback: Callable[[AgentPlan], dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        replan_count = 0
        while True:
            ready = board.ready_tasks()
            if not ready:
                break
            task = ready[0]
            task_id = task.id
            board.start(task_id)
            scope.current_task_id = task_id
            scope.allowed_tools = set(task.suggested_tools or plan.available_tools)
            scope.task_observations[task_id] = []
            scope.record("task_started", task=task.to_dict())

            generated = self._consume(model_runner, self._task_messages(plan, board, task, base_messages))
            task.result = self._response_text(generated)
            scope.record("task_draft_created", task_id=task_id)
            task.status = TaskStatus.VERIFYING

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

            reflection: dict[str, Any] | None = None
            if semantic_runner is not None:
                reflection = self._semantic_json(
                    semantic_runner,
                    self._reflection_messages(plan, board, task, verification),
                )
                added = board.apply_reflection(reflection or {})
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
        session_store.get(scope.request.session_id).task_board = board.to_dict()
        return [{"role": "assistant", "content": self._assemble_results(board)}]

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
        if not auto_mode and plan_review_callback is not None:
            decision = plan_review_callback(replanned, preliminary_board)
            if not decision or decision.get("cancelled") or decision.get("approved") is False:
                raise RuntimeError("用户取消了重新规划，本次任务未执行")
            next_auto_mode = bool(decision.get("auto_mode"))

        if replanned.configuration and plan_confirmation_callback is not None:
            if scope.event_callback is not None:
                scope.record("configuration_required", configuration=replanned.configuration)
            configuration = plan_confirmation_callback(replanned)
            if not configuration or configuration.get("cancelled"):
                raise RuntimeError("用户取消了算法配置，本次任务未执行")
        else:
            configuration = None

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
    ) -> list[dict[str, Any]]:
        completed = [
            {
                "task_id": item.id,
                "task_title": item.title,
                "deliverable": item.deliverable,
                "result": item.result,
            }
            for item in board.tasks.values()
            if item.status is TaskStatus.COMPLETED
        ]
        payload = {
            "current_task": {
                "task_id": task.id,
                "task_title": task.title,
                "task_goal": task.task_goal,
                "deliverable": task.deliverable,
                "done_when": task.done_when,
            },
            "completed_results": completed,
            "previous_feedback": task.feedback[-1] if task.feedback else "",
        }
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

    def _semantic_json(self, runner: SemanticRunner, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        try:
            return self._parse_json(self._consume(runner, messages))
        except Exception:
            return None

    def _semantic_text(self, runner: SemanticRunner, messages: list[dict[str, Any]]) -> str:
        try:
            return self._response_text(self._consume(runner, messages))
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
