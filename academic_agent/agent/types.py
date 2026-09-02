"""Agent Runtime 的请求、计划与执行作用域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import uuid


class AgentRoute(str, Enum):
    CHAT = "chat"
    DOCUMENT = "document"
    PROJECT = "project"
    DATA = "data"
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
    done_when: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentPlan:
    objective: str
    route: AgentRoute
    steps: tuple[PlanStep, ...]
    available_tools: tuple[str, ...] = ()
    requires_confirmation: bool = False
    configuration: dict[str, Any] | None = None
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
            lines.append("执行前必须遵守用户在算法确认窗口中选择的算法和参数。")
        if self.expected_outputs:
            lines.append("预期产物：" + "、".join(self.expected_outputs))
        if self.success_criteria:
            lines.append("完成条件：" + "；".join(self.success_criteria))
        return "\n".join(lines)


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
    task_goal: str = ""
    deliverable: str = ""
    done_when: str = ""
    max_attempts: int = 2
    attempts: int = 0
    feedback: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "depends_on": list(self.depends_on),
            "suggested_tools": list(self.suggested_tools),
            "task_goal": self.task_goal,
            "deliverable": self.deliverable,
            "done_when": self.done_when,
            "max_attempts": self.max_attempts,
            "attempts": self.attempts,
            "feedback": list(self.feedback),
            "evidence": list(self.evidence),
            "artifacts": list(self.artifacts),
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
            depends_on = (previous_id,) if previous_id else ()
            tasks[task_id] = TaskItem(
                id=task_id,
                title=title,
                description=goal,
                task_goal=goal,
                deliverable=deliverable,
                done_when=done_when,
                suggested_tools=suggested_tools,
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
    events: list[dict[str, Any]] = field(default_factory=list)
    event_callback: Callable[[dict[str, Any]], None] | None = None
    current_task_id: str | None = None
    allowed_tools: set[str] | None = None
    task_observations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        self.events.append(event)
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception:
                pass
