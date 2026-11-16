"""本地能力路由和 Plan 初始上下文。

复杂任务的语义 Plan、任务拆分和 Reflection 由 Runtime 交给无工具的规划模型；
这里的代码只负责建立安全的能力边界，避免模型直接决定越权工具。
"""

from __future__ import annotations

import re

from academic_agent.agent.planning.configuration import configuration_for_query
from academic_agent.agent.skills.catalog import SkillCatalog, skill_catalog
from academic_agent.agent.skills.manager import skill_manager
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

    # 轻量对话必须在语义规划模型之前识别。这里只匹配独立的问候、身份、
    # 能力和礼貌回应，不把“你好，帮我分析这份数据”误判成免规划请求。
    _LIGHTWEIGHT_CHAT = (
        re.compile(
            r"^(?:你好|您好|嗨|哈喽|hello|hi|hey|早上好|中午好|下午好|晚上好)"
            r"(?:呀|啊|哦|喽)?$",
            re.I,
        ),
        re.compile(r"^(?:你是谁|你叫什么|你是什么|你能做什么|介绍一下你自己)(?:呀|啊|哦)?$", re.I),
        re.compile(r"^(?:在吗|在不在|能听到吗|谢谢|谢谢你|辛苦了|好的|收到|明白了|再见|拜拜)(?:呀|啊|哦)?$", re.I),
    )
    _SIMPLE_DOCUMENT_READ = re.compile(
        r"^(?:(?:请|帮我|我想|我只想|只需要|只要))?"
        r"(?:读|阅读|看|查看|读取)(?:一下|一遍|下)?"
        r"(?:这个|这份|该|此|当前)?"
        r"(?:文件|文档|pdf|报告|论文|资料)"
        r"(?:即可|就好|就行)?$",
        re.I,
    )

    # “当前文件”通常是用户上传的数据文件，不应因为出现“文件”二字
    # 就被路由到项目工作区；只有明确提到工作区/项目/代码操作时才走 project。
    _PROJECT = re.compile(r"目录|工作区|项目|代码|glob|grep|读取|修改|生成|删除", re.I)
    _SKILL_MANAGEMENT = re.compile(
        r"(?:安装|列出|查看|更新|导入|添加|启用|覆盖|有哪些|已安装).{0,40}(?:技能|skills?|skill\.md)|"
        r"(?:技能|skills?|skill\.md).{0,40}(?:安装|列出|查看|更新|导入|添加|启用|覆盖|有哪些|已安装)",
        re.I,
    )
    _DOCUMENT = re.compile(r"文档|文献|学术|研究|PDF|pdf|手册|论文|报告|资料|知识库|检索|引用|原文", re.I)
    _TEXT2SQL = re.compile(
        r"\b(?:mysql|text2sql|sql)\b|数据库|数据表|表结构|"
        r"(?:查询|统计|汇总).{0,20}(?:表|数据库|指标)|从.{0,20}表",
        re.I,
    )
    _VISUAL = re.compile(r"画图|图表|可视化|折线图|柱状图|词云|散点图|热力图", re.I)
    _ML = re.compile(r"机器学习|分类|回归|因果|预测|训练|特征", re.I)
    _DATA = re.compile(r"文本挖掘|聚类|情感|关键词|预处理|实体识别|命名实体|entity recognition|关系抽取|关系识别|relation extraction|知识图谱|knowledge graph|NER|数据|字段|表格", re.I)
    _WRITE = re.compile(r"生成|创建|写入|写代码|编写|新增代码|实现|重构|修改|编辑|删除|覆盖", re.I)

    def __init__(
        self,
        registry: ToolRegistry = tool_registry,
        skills: SkillCatalog = skill_catalog,
    ) -> None:
        self.registry = registry
        self.skills = skills

    def create_plan(
        self,
        request: AgentRequest,
        routing_hint: str = "",
        *,
        capability_routes: tuple[AgentRoute, ...] | None = None,
    ) -> AgentPlan:
        query = request.query.strip() or "响应用户请求"
        route_query = f"{query}\n{routing_hint.strip()}" if routing_hint.strip() else query
        if request.chat_only:
            route, secondary = AgentRoute.CHAT, ()
        elif capability_routes:
            # 语义模型只建议已知能力类别；工具白名单仍从本地注册表重建。
            routes = tuple(dict.fromkeys(AgentRoute(item) for item in capability_routes))
            route = routes[0]
            secondary = () if route is AgentRoute.CHAT else tuple(
                item for item in routes[1:] if item is not AgentRoute.CHAT
            )
        else:
            route = self._route(route_query, request.workspace_path)
            secondary = self._secondary_routes(route_query, route)
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

        configuration = None if route is AgentRoute.CHAT else configuration_for_query(route_query, route.value)
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
            requires_confirmation=route is not AgentRoute.CHAT and bool(self._WRITE.search(query)),
            configuration=configuration,
            route_decision=routing,
            expected_outputs=self._expected_outputs(route, query),
            success_criteria=(
                "Plan 与用户确认的信息保持一致",
                "每个任务结果经过 passed/feedback 检查",
                "最终产物只由已完成任务结果组装",
            ),
        )

    def recognize_intent(self, request: AgentRequest) -> str:
        """在调用语义规划模型前识别是否可以直接聊天。

        返回值是运行时分流标签：``lightweight_chat`` 只调用无工具聊天模型，
        ``task`` 保留现有 Work 路由、Plan 和任务板流程。Chat 模式本身已经
        是无工具模式，因此单独返回 ``chat`` 便于日志和测试区分。
        """
        if request.chat_only:
            return "chat"
        if self.is_lightweight_chat(request.query):
            return "lightweight_chat"
        if self.is_simple_document_read(request.query):
            return "simple_document_read"
        return "task"

    @classmethod
    def is_lightweight_chat(cls, query: str) -> bool:
        """判断短问候/闲聊是否不需要进入 Work 规划流程。"""
        normalized = re.sub(r"\s+", "", str(query or "")).strip()
        normalized = re.sub(r"[!！,，。?？：:；;、~～…]+$", "", normalized)
        if not normalized or len(normalized) > 24:
            return False
        return any(pattern.fullmatch(normalized) for pattern in cls._LIGHTWEIGHT_CHAT)

    @classmethod
    def is_simple_document_read(cls, query: str) -> bool:
        """识别只读一个文件的低风险请求，不要求用户确认完整 Plan。"""
        normalized = re.sub(r"\s+", "", str(query or "")).strip()
        normalized = re.sub(r"[!！,，。?？：:；;、~～…]+$", "", normalized)
        if not normalized or len(normalized) > 40:
            return False
        return bool(cls._SIMPLE_DOCUMENT_READ.fullmatch(normalized))

    def requires_plan_review(self, request: AgentRequest, plan: AgentPlan | None = None) -> bool:
        """判断本次请求是否需要展示人工 Plan 确认卡片。"""
        if request.chat_only:
            return False
        if plan is not None and plan.route is AgentRoute.CHAT:
            return False
        if self.is_lightweight_chat(request.query) or self.is_simple_document_read(request.query):
            return False
        return True

    def _route(self, query: str, workspace_path: str | None = None) -> AgentRoute:
        if self._SKILL_MANAGEMENT.search(query):
            return AgentRoute.SKILL_MANAGEMENT
        # 用户只输入一个 Skill 名称时，默认到项目 skills/ 目录中查找。
        # 只有扫描到真实的 SKILL.md 才触发该路由，避免普通短句被误判。
        if len(query.split()) == 1 and skill_manager.find_by_name(query, workspace_root=workspace_path):
            return AgentRoute.SKILL_MANAGEMENT
        if self.is_simple_document_read(query) or self._DOCUMENT.search(query):
            return AgentRoute.DOCUMENT
        if self._TEXT2SQL.search(query):
            return AgentRoute.TEXT2SQL
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
        if route is AgentRoute.SKILL_MANAGEMENT:
            return ()
        secondary: list[AgentRoute] = []
        if route is AgentRoute.TEXT2SQL:
            if self._VISUAL.search(query):
                secondary.append(AgentRoute.VISUALIZATION)
            return tuple(secondary)
        if route is not AgentRoute.VISUALIZATION and self._VISUAL.search(query):
            secondary.append(AgentRoute.VISUALIZATION)
        if route is not AgentRoute.DATA and self._DATA.search(query):
            secondary.append(AgentRoute.DATA)
        if route is not AgentRoute.MACHINE_LEARNING and self._ML.search(query):
            secondary.append(AgentRoute.MACHINE_LEARNING)
        if route is not AgentRoute.PROJECT and self._PROJECT.search(query):
            secondary.append(AgentRoute.PROJECT)
        return tuple(secondary)

    @staticmethod
    def _expected_outputs(route: AgentRoute, query: str) -> tuple[str, ...]:
        outputs = {
            AgentRoute.SKILL_MANAGEMENT: ("Skill 安装结果或已安装 Skill 列表",),
            AgentRoute.DOCUMENT: ("带引用的文档回答",),
            AgentRoute.PROJECT: ("项目操作结果或文件路径",),
            AgentRoute.DATA: ("分析结果摘要",),
            AgentRoute.TEXT2SQL: ("业务口径、执行 SQL 和查询结果",),
            AgentRoute.MACHINE_LEARNING: ("模型结果、指标和限制",),
            AgentRoute.VISUALIZATION: ("图表文件",),
            AgentRoute.CHAT: ("自然语言回答",),
        }
        result = list(outputs.get(route, ("任务结果",)))
        if any(word in query for word in ("画图", "图表", "可视化", "词云")):
            result.append("图表产物")
        return tuple(dict.fromkeys(result))
