"""本地能力路由和 Plan 初始上下文。

复杂任务的语义 Plan、任务拆分和 Reflection 由 Runtime 交给无工具的规划模型；
这里的代码只负责建立安全的能力边界，避免模型直接决定越权工具。
"""

from __future__ import annotations

import re

from academic_agent.agent.planning.configuration import configuration_for_query
from academic_agent.agent.skills.catalog import SkillCatalog, skill_catalog
from academic_agent.agent.tooling.registry import ToolRegistry, tool_registry
from academic_agent.agent.types import (
    AgentPlan,
    AgentRequest,
    AgentRoute,
    PlanStep,
    RoutingDecision,
)


class TaskPlanner:
    """完成确定性的能力路由，不替模型硬编码完整任务清单。"""

    # “当前文件”通常是用户上传的数据文件，不应因为出现“文件”二字
    # 就被路由到项目工作区；只有明确提到工作区/项目/代码操作时才走 project。
    _PROJECT = re.compile(r"目录|工作区|项目|代码|glob|grep|读取|修改|生成|删除", re.I)
    _DOCUMENT = re.compile(r"文档|文献|学术|研究|PDF|pdf|手册|论文|报告|资料|知识库|检索|引用|原文", re.I)
    _VISUAL = re.compile(r"画图|图表|可视化|折线图|柱状图|词云|散点图|热力图", re.I)
    _ML = re.compile(r"机器学习|分类|回归|因果|预测|训练|特征", re.I)
    _DATA = re.compile(r"文本挖掘|聚类|情感|关键词|预处理|实体识别|命名实体|entity recognition|关系抽取|关系识别|relation extraction|知识图谱|knowledge graph|NER|数据|字段|表格", re.I)
    _WRITE = re.compile(r"生成|创建|写入|写代码|编写|新增代码|实现|重构|修改|编辑|删除|覆盖", re.I)

    _ACADEMIC_INTENT_OPTIONS = (
        {
            "label": "文本挖掘",
            "value": "文本挖掘",
            "description": "适合评论、问卷、访谈、论文摘要等文本数据。",
        },
        {
            "label": "机器学习",
            "value": "机器学习",
            "description": "适合明确目标变量、特征变量或研究假设的数据建模。",
        },
        {
            "label": "数据分析与可视化",
            "value": "数据分析与可视化",
            "description": "适合先了解数据质量、统计规律和图表呈现。",
        },
        {
            "label": "学术文献与研究",
            "value": "学术文献与研究",
            "description": "适合文献证据整理、论文资料问答和研究方案梳理。",
        },
    )

    def __init__(
        self,
        registry: ToolRegistry = tool_registry,
        skills: SkillCatalog = skill_catalog,
    ) -> None:
        self.registry = registry
        self.skills = skills

    def create_plan(self, request: AgentRequest, routing_hint: str = "") -> AgentPlan:
        query = request.query.strip() or "响应用户请求"
        route_query = f"{query}\n{routing_hint.strip()}" if routing_hint.strip() else query
        route = AgentRoute.CHAT if request.chat_only else self._route(route_query)
        secondary = () if request.chat_only else self._secondary_routes(route_query, route)
        skills = [self.skills.for_route(item) for item in (route, *secondary)]
        available = tuple(dict.fromkeys(
            name
            for skill in skills if skill
            for name in skill.tools
            if self.registry.get(name) is not None
        ))

        steps = [PlanStep("understand", "结合当前需求、会话上下文和已确认信息判断任务边界")]
        if route is AgentRoute.CHAT:
            steps.append(PlanStep("respond", "直接进行自然语言回答，不访问项目文件或工具"))
        else:
            skill = self.skills.for_route(route)
            if skill:
                steps.extend(
                    PlanStep(f"workflow_{index}", description)
                    for index, description in enumerate(skill.workflow, 1)
                )
            configuration = configuration_for_query(route_query, route.value)
            if configuration:
                steps.append(PlanStep("configure_algorithm", "在执行前确认算法与参数"))
            if available:
                steps.append(PlanStep(
                    "execute",
                    "由动态任务清单选择当前任务，调用最小必要工具并读取观察结果",
                    suggested_tools=available,
                    requires_observation=True,
                ))
            steps.append(PlanStep("respond", "根据已完成任务的结果确定性汇总最终产物"))

        configuration = None if request.chat_only else configuration_for_query(route_query, route.value)
        routing = RoutingDecision(
            primary_route=route,
            intent=query,
            confidence=1.0,
            secondary_routes=secondary,
            available_tools=available,
        )
        return AgentPlan(
            objective=query,
            route=route,
            steps=tuple(steps),
            available_tools=available,
            requires_confirmation=bool(self._WRITE.search(query)),
            configuration=configuration,
            route_decision=routing,
            expected_outputs=self._expected_outputs(route, query),
            success_criteria=(
                "Plan 与用户确认的信息保持一致",
                "每个任务结果经过 passed/feedback 检查",
                "最终产物只由已完成任务结果组装",
            ),
        )

    def requires_academic_intent_clarification(self, request: AgentRequest) -> bool:
        """Work 模式没有识别出学术能力时，先让用户选择研究方向。

        Chat 模式仍然可以自由进行自然语言问答；这个约束只用于 Work 模式，
        防止“帮我看看”“分析一下”等模糊请求直接进入一个没有明确目标的任务板。
        """
        return not request.chat_only and self._route(request.query.strip()) is AgentRoute.CHAT

    @classmethod
    def academic_intent_question(cls) -> dict[str, object]:
        """返回 UI 可直接渲染的学术方向单选问题。"""
        return {
            "info_id": "R0-F1",
            "topic": "先选择研究方向",
            "why_needed": "当前需求还没有明确要做哪类学术任务，先确定方向才能选择正确的分析流程和工具。",
            "question": "你更希望我从哪个方向开始？请选择最接近的一项。",
            "options": [dict(item) for item in cls._ACADEMIC_INTENT_OPTIONS],
            "expected_format": "点击一个研究方向即可",
            "example": "例如：评论数据选“文本挖掘”，有目标变量的数据选“机器学习”",
            "default_assumption": "",
            "answer": "",
            "status": "missing",
            "source": "local_router",
        }

    def _route(self, query: str) -> AgentRoute:
        if self._DOCUMENT.search(query):
            return AgentRoute.DOCUMENT
        if self._VISUAL.search(query):
            return AgentRoute.VISUALIZATION
        if self._ML.search(query):
            return AgentRoute.MACHINE_LEARNING
        if self._PROJECT.search(query):
            return AgentRoute.PROJECT
        if self._DATA.search(query):
            return AgentRoute.DATA
        return AgentRoute.CHAT

    def _secondary_routes(self, query: str, route: AgentRoute) -> tuple[AgentRoute, ...]:
        secondary: list[AgentRoute] = []
        if route is not AgentRoute.VISUALIZATION and self._VISUAL.search(query):
            secondary.append(AgentRoute.VISUALIZATION)
        if route is not AgentRoute.DATA and self._DATA.search(query):
            secondary.append(AgentRoute.DATA)
        if route is not AgentRoute.PROJECT and self._PROJECT.search(query):
            secondary.append(AgentRoute.PROJECT)
        return tuple(secondary)

    @staticmethod
    def _expected_outputs(route: AgentRoute, query: str) -> tuple[str, ...]:
        outputs = {
            AgentRoute.DOCUMENT: ("带引用的文档回答",),
            AgentRoute.PROJECT: ("项目操作结果或文件路径",),
            AgentRoute.DATA: ("分析结果摘要",),
            AgentRoute.MACHINE_LEARNING: ("模型结果、指标和限制",),
            AgentRoute.VISUALIZATION: ("图表文件",),
            AgentRoute.CHAT: ("自然语言回答",),
        }
        result = list(outputs.get(route, ("任务结果",)))
        if any(word in query for word in ("画图", "图表", "可视化", "词云")):
            result.append("图表产物")
        return tuple(dict.fromkeys(result))
