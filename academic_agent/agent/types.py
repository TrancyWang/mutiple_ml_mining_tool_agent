"""Agent Runtime 的请求、计划与执行作用域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable
import uuid

if TYPE_CHECKING:
    from academic_agent.infrastructure.task_workspace import WorkspaceContext


class AgentRoute(str, Enum):
    CHAT = "chat"
    SKILL_MANAGEMENT = "skill_management"
    DOCUMENT = "document"
    PROJECT = "project"
    DATA = "data"
    TEXT2SQL = "text2sql"
    MACHINE_LEARNING = "machine_learning"
    VISUALIZATION = "visualization"


class TaskStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WAITING_USER = "waiting_user"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """路由阶段的结构化产物，而不是只返回一个字符串路由。"""

    primary_route: AgentRoute
    intent: str
    confidence: float = 1.0
    secondary_routes: tuple[AgentRoute, ...] = ()
    missing_information: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentRequest:
    messages: list[dict[str, Any]]
    user_id: str
    session_id: str
    workspace_path: str | None = None
    chat_only: bool = False
    workspace_id: str | None = None

    @property
    def query(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [dict(message) for message in self.messages],
            "user_id": self.user_id,
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "workspace_id": self.workspace_id,
            "chat_only": self.chat_only,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentRequest":
        messages = value.get("messages") if isinstance(value, dict) else []
        return cls(
            messages=[dict(item) for item in messages if isinstance(item, dict)],
            user_id=str(value.get("user_id") or "default_user"),
            session_id=str(value.get("session_id") or "default"),
            workspace_path=value.get("workspace_path"),
            workspace_id=value.get("workspace_id"),
            chat_only=bool(value.get("chat_only", False)),
        )


@dataclass(frozen=True, slots=True)
class PlanStep:
    name: str
    description: str
    suggested_tools: tuple[str, ...] = ()
    requires_observation: bool = False
    done_when: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentPlan:
    objective: str
    route: AgentRoute
    steps: tuple[PlanStep, ...]
    available_tools: tuple[str, ...] = ()
    requires_confirmation: bool = False
    configuration: dict[str, Any] | None = None
    # 由语义规划模型提出、由程序校验后的算法调度任务。它不是工具权限；
    # 真正可调用的工具仍然只能来自 available_tools。
    algorithm_schedule: tuple[dict[str, Any], ...] = ()
    algorithm_decision: dict[str, Any] = field(default_factory=dict)
    route_decision: RoutingDecision | None = None
    expected_outputs: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    task_templates: tuple[dict[str, Any], ...] = ()
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    plan_document: str = ""
    confirmed_information: tuple[dict[str, Any], ...] = ()
    information_frame: tuple[dict[str, Any], ...] = ()
    routing_focus: tuple[str, ...] = ()
    routing_reason: str = ""
    is_replan: bool = False
    replan_reason: str = ""

    def as_prompt(self) -> str:
        lines = [f"目标：{self.objective}", f"路由：{self.route.value}", "建议步骤："]
        for index, step in enumerate(self.steps, 1):
            tool_hint = f"；候选工具：{', '.join(step.suggested_tools)}" if step.suggested_tools else ""
            lines.append(f"{index}. {step.description}{tool_hint}")
        lines.append("该计划是运行时建议；应依据工具观察结果动态调整。界面会展示简化步骤，最终回答不要重复内部计划。")
        if self.configuration:
            lines.append("如果用户确认了算法配置，执行时必须遵守用户选择的算法和参数。")
        if self.algorithm_schedule:
            lines.append("算法调度顺序：")
            for index, item in enumerate(self.algorithm_schedule, 1):
                suggested = item.get("suggested_tools") or ()
                tool = str(item.get("scheduled_tool") or (suggested[0] if suggested else ""))
                title = str(item.get("task_title") or item.get("title") or tool)
                lines.append(f"{index}. {title}（{tool}）")
            lines.append("上述调度由程序校验后执行；如果算法工具返回错误，只能根据观察结果重试或重规划。")
        if self.expected_outputs:
            lines.append("预期产物：" + "、".join(self.expected_outputs))
        if self.success_criteria:
            lines.append("完成条件：" + "；".join(self.success_criteria))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        route_decision = None
        if self.route_decision is not None:
            route_decision = {
                "primary_route": self.route_decision.primary_route.value,
                "intent": self.route_decision.intent,
                "confidence": self.route_decision.confidence,
                "secondary_routes": [item.value for item in self.route_decision.secondary_routes],
                "missing_information": list(self.route_decision.missing_information),
                "available_tools": list(self.route_decision.available_tools),
            }
        return {
            "objective": self.objective,
            "route": self.route.value,
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "suggested_tools": list(step.suggested_tools),
                    "requires_observation": step.requires_observation,
                    "done_when": list(step.done_when),
                }
                for step in self.steps
            ],
            "available_tools": list(self.available_tools),
            "requires_confirmation": self.requires_confirmation,
            "configuration": self.configuration,
            "algorithm_schedule": list(self.algorithm_schedule),
            "algorithm_decision": self.algorithm_decision,
            "route_decision": route_decision,
            "expected_outputs": list(self.expected_outputs),
            "success_criteria": list(self.success_criteria),
            "questions": list(self.questions),
            "task_templates": list(self.task_templates),
            "plan_id": self.plan_id,
            "plan_document": self.plan_document,
            "confirmed_information": list(self.confirmed_information),
            "information_frame": list(self.information_frame),
            "routing_focus": list(self.routing_focus),
            "routing_reason": self.routing_reason,
            "is_replan": self.is_replan,
            "replan_reason": self.replan_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentPlan":
        raw_route = str(value.get("route") or AgentRoute.CHAT.value)
        try:
            route = AgentRoute(raw_route)
        except ValueError:
            route = AgentRoute.CHAT
        steps = tuple(
            PlanStep(
                name=str(item.get("name") or "step"),
                description=str(item.get("description") or ""),
                suggested_tools=tuple(str(tool) for tool in item.get("suggested_tools", ()) or ()),
                requires_observation=bool(item.get("requires_observation", False)),
                done_when=tuple(str(condition) for condition in item.get("done_when", ()) or ()),
            )
            for item in value.get("steps", ()) or ()
            if isinstance(item, dict)
        )
        raw_decision = value.get("route_decision")
        route_decision = None
        if isinstance(raw_decision, dict):
            try:
                primary_route = AgentRoute(str(raw_decision.get("primary_route") or route.value))
            except ValueError:
                primary_route = route
            secondary_routes = []
            for item in raw_decision.get("secondary_routes", ()) or ():
                try:
                    secondary_routes.append(AgentRoute(str(item)))
                except ValueError:
                    continue
            route_decision = RoutingDecision(
                primary_route=primary_route,
                intent=str(raw_decision.get("intent") or ""),
                confidence=float(raw_decision.get("confidence", 1.0)),
                secondary_routes=tuple(secondary_routes),
                missing_information=tuple(str(item) for item in raw_decision.get("missing_information", ()) or ()),
                available_tools=tuple(str(item) for item in raw_decision.get("available_tools", ()) or ()),
            )
        return cls(
            objective=str(value.get("objective") or ""),
            route=route,
            steps=steps,
            available_tools=tuple(str(item) for item in value.get("available_tools", ()) or ()),
            requires_confirmation=bool(value.get("requires_confirmation", False)),
            configuration=value.get("configuration") if isinstance(value.get("configuration"), dict) else None,
            algorithm_schedule=tuple(item for item in value.get("algorithm_schedule", ()) or () if isinstance(item, dict)),
            algorithm_decision=value.get("algorithm_decision") if isinstance(value.get("algorithm_decision"), dict) else {},
            route_decision=route_decision,
            expected_outputs=tuple(str(item) for item in value.get("expected_outputs", ()) or ()),
            success_criteria=tuple(str(item) for item in value.get("success_criteria", ()) or ()),
            questions=tuple(str(item) for item in value.get("questions", ()) or ()),
            task_templates=tuple(item for item in value.get("task_templates", ()) or () if isinstance(item, dict)),
            plan_id=str(value.get("plan_id") or uuid.uuid4().hex[:12]),
            plan_document=str(value.get("plan_document") or ""),
            confirmed_information=tuple(item for item in value.get("confirmed_information", ()) or () if isinstance(item, dict)),
            information_frame=tuple(item for item in value.get("information_frame", ()) or () if isinstance(item, dict)),
            routing_focus=tuple(str(item) for item in value.get("routing_focus", ()) or ()),
            routing_reason=str(value.get("routing_reason") or ""),
            is_replan=bool(value.get("is_replan", False)),
            replan_reason=str(value.get("replan_reason") or ""),
        )


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    feedback: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskItem:
    """动态任务清单中的一项，和具体工具调用明确区分。"""

    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.READY
    depends_on: tuple[str, ...] = ()
    suggested_tools: tuple[str, ...] = ()
    # scheduled_tool 非空时由 Runtime 直接执行该工具，再让模型解释结果；
    # 为空时保留旧的模型自主补参调用路径。
    scheduled_tool: str = ""
    scheduled_arguments: dict[str, Any] = field(default_factory=dict)
    task_goal: str = ""
    deliverable: str = ""
    done_when: str = ""
    max_attempts: int = 2
    attempts: int = 0
    feedback: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    result: Any = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    resume_verification: bool = False
    agent_id: str = ""
    agent_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "depends_on": list(self.depends_on),
            "suggested_tools": list(self.suggested_tools),
            "scheduled_tool": self.scheduled_tool,
            "scheduled_arguments": dict(self.scheduled_arguments),
            "task_goal": self.task_goal,
            "deliverable": self.deliverable,
            "done_when": self.done_when,
            "max_attempts": self.max_attempts,
            "attempts": self.attempts,
            "feedback": list(self.feedback),
            "evidence": list(self.evidence),
            "artifacts": list(self.artifacts),
            "result": self.result,
            "observations": list(self.observations),
            "resume_verification": self.resume_verification,
        }


@dataclass(slots=True)
class TaskBoard:
    """任务状态板；Scheduler 只从这里选择可执行任务。"""

    tasks: dict[str, TaskItem] = field(default_factory=dict)
    version: int = 1
    current_task_id: str | None = None
    board_goal: str = ""
    final_deliverable: str = ""
    plan_document: str = ""

    @classmethod
    def from_plan(cls, plan: AgentPlan) -> "TaskBoard":
        templates = plan.task_templates
        if not templates:
            templates = tuple(
                {
                    "id": f"plan_{index}_{step.name}",
                    "title": step.description,
                    "description": step.description,
                    "suggested_tools": step.suggested_tools,
                    "task_goal": step.description,
                    "deliverable": step.description,
                    "done_when": "模型返回当前任务的可用结果",
                    "depends_on": (
                        (f"plan_{index - 1}_{plan.steps[index - 2].name}",)
                        if index > 1 else ()
                    ),
                }
                for index, step in enumerate(plan.steps, 1)
            )
        tasks: dict[str, TaskItem] = {}
        for index, raw in enumerate(templates, 1):
            task_id = str(raw.get("id") or f"task_{index}")
            depends_on = tuple(str(item) for item in raw.get("depends_on", ()))
            tasks[task_id] = TaskItem(
                id=task_id,
                title=str(raw.get("title") or raw.get("description") or task_id),
                description=str(raw.get("description") or raw.get("title") or task_id),
                depends_on=depends_on,
                suggested_tools=tuple(str(item) for item in (
                    raw.get("tool_hints") or raw.get("suggested_tools") or ()
                )),
                scheduled_tool=str(raw.get("scheduled_tool") or ""),
                scheduled_arguments=(
                    dict(raw.get("scheduled_arguments"))
                    if isinstance(raw.get("scheduled_arguments"), dict) else {}
                ),
                task_goal=str(raw.get("task_goal") or raw.get("description") or raw.get("title") or task_id),
                deliverable=str(raw.get("deliverable") or raw.get("description") or raw.get("title") or task_id),
                done_when=str(raw.get("done_when") or "模型返回当前任务的可用结果"),
                max_attempts=max(1, int(raw.get("max_attempts", 2))),
            )
        return cls(tasks=tasks, plan_document=plan.plan_document)

    @classmethod
    def from_task_plan(
        cls,
        plan: AgentPlan,
        task_plan: dict[str, Any],
        max_attempts: int = 2,
    ) -> "TaskBoard":
        """把模型拆出的任务语义转换为程序管理的状态板。"""
        raw_tasks = task_plan.get("tasks") if isinstance(task_plan, dict) else None
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("任务拆分结果必须包含非空 tasks 列表")
        if len(raw_tasks) > 50:
            raise ValueError("任务数量超过安全上限 50")
        tasks: dict[str, TaskItem] = {}
        previous_id: str | None = None
        for index, raw in enumerate(raw_tasks, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"第 {index} 个任务不是对象")
            task_id = f"T{index}"
            title = str(raw.get("task_title") or raw.get("title") or "").strip()
            goal = str(raw.get("task_goal") or "").strip()
            deliverable = str(raw.get("deliverable") or "").strip()
            done_when = str(raw.get("done_when") or "").strip()
            if not title or not goal or not deliverable or not done_when:
                raise ValueError(f"第 {index} 个任务缺少 task_title/task_goal/deliverable/done_when")
            raw_tools = raw.get("suggested_tools") or raw.get("tool_hints") or ()
            if isinstance(raw_tools, str):
                raw_tools = [raw_tools]
            suggested_tools = tuple(dict.fromkeys(
                str(item) for item in raw_tools
                if str(item) in plan.available_tools
            ))
            scheduled_tool = str(raw.get("scheduled_tool") or raw.get("algorithm_tool") or "").strip()
            if scheduled_tool not in suggested_tools:
                scheduled_tool = ""
            raw_arguments = raw.get("scheduled_arguments") or raw.get("tool_arguments") or {}
            scheduled_arguments = dict(raw_arguments) if isinstance(raw_arguments, dict) else {}
            depends_on = (previous_id,) if previous_id else ()
            tasks[task_id] = TaskItem(
                id=task_id,
                title=title,
                description=goal,
                task_goal=goal,
                deliverable=deliverable,
                done_when=done_when,
                suggested_tools=suggested_tools,
                scheduled_tool=scheduled_tool,
                scheduled_arguments=scheduled_arguments,
                status=TaskStatus.READY,
                depends_on=depends_on,
                max_attempts=max(1, int(max_attempts)),
            )
            previous_id = task_id
        return cls(
            tasks=tasks,
            board_goal=str(task_plan.get("board_goal") or plan.objective),
            final_deliverable=str(task_plan.get("final_deliverable") or "最终任务结果"),
            plan_document=plan.plan_document,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskBoard":
        raw_tasks = value.get("tasks", ()) if isinstance(value, dict) else ()
        if isinstance(raw_tasks, dict):
            raw_tasks = list(raw_tasks.values())
        tasks: dict[str, TaskItem] = {}
        for raw in raw_tasks or ():
            if not isinstance(raw, dict):
                continue
            raw_status = str(raw.get("status") or TaskStatus.READY.value)
            try:
                status = TaskStatus(raw_status)
            except ValueError:
                status = TaskStatus.READY
            try:
                max_attempts = max(1, int(raw.get("max_attempts", 2)))
            except (TypeError, ValueError):
                max_attempts = 2
            try:
                attempts = max(0, int(raw.get("attempts", 0)))
            except (TypeError, ValueError):
                attempts = 0
            task_id = str(raw.get("id") or f"T{len(tasks) + 1}")
            tasks[task_id] = TaskItem(
                id=task_id,
                agent_id=str(raw.get("agent_id") or ""),
                agent_name=str(raw.get("agent_name") or ""),
                title=str(raw.get("title") or task_id),
                description=str(raw.get("description") or raw.get("title") or task_id),
                status=status,
                depends_on=tuple(str(item) for item in raw.get("depends_on", ()) or ()),
                suggested_tools=tuple(str(item) for item in raw.get("suggested_tools", ()) or ()),
                scheduled_tool=str(raw.get("scheduled_tool") or ""),
                scheduled_arguments=raw.get("scheduled_arguments") if isinstance(raw.get("scheduled_arguments"), dict) else {},
                task_goal=str(raw.get("task_goal") or raw.get("description") or ""),
                deliverable=str(raw.get("deliverable") or ""),
                done_when=str(raw.get("done_when") or ""),
                max_attempts=max_attempts,
                attempts=attempts,
                feedback=[str(item) for item in raw.get("feedback", ()) or ()],
                evidence=[item for item in raw.get("evidence", ()) or () if isinstance(item, dict)],
                artifacts=[str(item) for item in raw.get("artifacts", ()) or ()],
                result=raw.get("result"),
                observations=[item for item in raw.get("observations", ()) or () if isinstance(item, dict)],
                resume_verification=bool(raw.get("resume_verification", False)),
            )
        return cls(
            tasks=tasks,
            version=int(value.get("version", 1) or 1) if isinstance(value, dict) else 1,
            current_task_id=(str(value.get("current_task_id")) if value.get("current_task_id") else None),
            board_goal=str(value.get("board_goal") or "") if isinstance(value, dict) else "",
            final_deliverable=str(value.get("final_deliverable") or "") if isinstance(value, dict) else "",
            plan_document=str(value.get("plan_document") or "") if isinstance(value, dict) else "",
        )

    def add_task(
        self,
        title: str,
        task_goal: str,
        deliverable: str,
        done_when: str,
        depends_on: tuple[str, ...] = (),
        max_attempts: int = 2,
    ) -> TaskItem:
        """由 Reflection 增加任务；编号和初始状态始终由程序创建。"""
        task_id = f"T{len(self.tasks) + 1}"
        while task_id in self.tasks:
            task_id = f"T{len(self.tasks) + 2}"
        unknown = [item for item in depends_on if item not in self.tasks]
        if unknown:
            raise ValueError(f"新任务依赖不存在的任务：{unknown}")
        task = TaskItem(
            id=task_id,
            title=str(title),
            description=str(task_goal),
            task_goal=str(task_goal),
            deliverable=str(deliverable),
            done_when=str(done_when),
            depends_on=tuple(depends_on),
            max_attempts=max(1, int(max_attempts)),
        )
        self.tasks[task_id] = task
        self.version += 1
        return task

    def apply_reflection(self, reflection: dict[str, Any]) -> list[TaskItem]:
        """应用受限的 Reflection 变更，返回新增任务。"""
        added: list[TaskItem] = []
        if not isinstance(reflection, dict):
            return added
        if str(reflection.get("action", "continue")) != "add_tasks":
            return added
        raw_tasks = reflection.get("new_tasks") or []
        if not isinstance(raw_tasks, list):
            return added
        dependency = self.current_task_id
        for raw in raw_tasks[:5]:
            if not isinstance(raw, dict):
                continue
            dependencies = tuple(str(item) for item in raw.get("depends_on", ()) if str(item) in self.tasks)
            if not dependencies and dependency:
                dependencies = (dependency,)
            try:
                added.append(self.add_task(
                    title=str(raw.get("task_title") or raw.get("title") or "新增任务"),
                    task_goal=str(raw.get("task_goal") or raw.get("objective") or "补充当前任务"),
                    deliverable=str(raw.get("deliverable") or "补充结果"),
                    done_when=str(raw.get("done_when") or "结果满足新增任务目标"),
                    depends_on=dependencies,
                ))
            except ValueError:
                continue
        return added

    def ready_tasks(self) -> list[TaskItem]:
        ready: list[TaskItem] = []
        for task in self.tasks.values():
            if task.status is not TaskStatus.READY:
                continue
            if all(
                self.tasks.get(dependency) is not None
                and self.tasks[dependency].status is TaskStatus.COMPLETED
                for dependency in task.depends_on
            ):
                ready.append(task)
        return ready

    def start(self, task_id: str) -> TaskItem:
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        self.current_task_id = task_id
        return task

    def complete(self, task_id: str, verification: VerificationResult) -> TaskItem:
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.evidence.extend(verification.evidence)
        task.artifacts.extend(item for item in verification.artifacts if item not in task.artifacts)
        self.current_task_id = None
        return task

    def retry(self, task_id: str, feedback: str) -> TaskItem:
        task = self.tasks[task_id]
        task.feedback.append(str(feedback))
        task.status = TaskStatus.READY
        self.current_task_id = None
        return task

    def block(self, task_id: str, feedback: str) -> TaskItem:
        task = self.tasks[task_id]
        task.feedback.append(str(feedback))
        task.status = TaskStatus.BLOCKED
        self.current_task_id = None
        return task

    def unresolved(self) -> list[TaskItem]:
        return [
            task for task in self.tasks.values()
            if task.status not in {TaskStatus.COMPLETED, TaskStatus.SKIPPED}
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "current_task_id": self.current_task_id,
            "board_goal": self.board_goal,
            "final_deliverable": self.final_deliverable,
            "plan_document": self.plan_document,
            "tasks": [task.to_dict() for task in self.tasks.values()],
        }


@dataclass(slots=True)
class ExecutionScope:
    request: AgentRequest
    plan: AgentPlan
    workspace: WorkspaceContext | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    event_callback: Callable[[dict[str, Any]], None] | None = None
    current_task_id: str | None = None
    allowed_tools: set[str] | None = None
    task_observations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    current_agent_id: str = ""
    current_agent_name: str = ""
    recovery_mode: bool = False
    checkpoint_callback: Callable[[], None] | None = None
    budget_guard: Callable[[str], None] | None = None
    harness_snapshot: Callable[[], dict[str, Any]] | None = None

    def record(self, event_type: str, **payload: Any) -> None:
        event = {
            "type": event_type,
            "agent_id": self.current_agent_id,
            "agent_name": self.current_agent_name,
            **payload,
        }
        self.events.append(event)
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception:
                pass
