"""零额外模型调用的任务 Planner。"""

from __future__ import annotations

import re

from academic_agent.agent.types import AgentPlan, AgentRequest, AgentRoute, PlanStep
from academic_agent.agent.tooling.registry import ToolRegistry, tool_registry
from academic_agent.agent.skills.catalog import SkillCatalog, skill_catalog


class TaskPlanner:
    """先完成本地路由和能力约束，再交给 Function Calling 动态执行。"""

    _PROJECT = re.compile(r"文件|目录|工作区|项目|代码|glob|grep|读取|修改|生成|删除", re.I)
    _VISUAL = re.compile(r"画图|图表|可视化|折线图|柱状图|词云|散点图|热力图", re.I)
    _ML = re.compile(r"机器学习|分类|回归|因果|预测|训练|特征", re.I)
    _DATA = re.compile(r"文本挖掘|聚类|情感|关键词|预处理|数据|字段|表格", re.I)
    _WRITE = re.compile(r"生成|创建|写入|修改|编辑|删除|覆盖", re.I)

    def __init__(
        self,
        registry: ToolRegistry = tool_registry,
        skills: SkillCatalog = skill_catalog,
    ) -> None:
        self.registry = registry
        self.skills = skills

    def create_plan(self, request: AgentRequest) -> AgentPlan:
        query = request.query.strip() or "响应用户请求"
        if request.chat_only:
            return AgentPlan(
                objective=query,
                route=AgentRoute.CHAT,
                steps=(PlanStep("respond", "直接进行自然语言回答，不访问项目文件或工具"),),
            )

        if self._VISUAL.search(query):
            route = AgentRoute.VISUALIZATION
        elif self._ML.search(query):
            route = AgentRoute.MACHINE_LEARNING
        elif self._PROJECT.search(query):
            route = AgentRoute.PROJECT
        elif self._DATA.search(query):
            route = AgentRoute.DATA
        else:
            route = AgentRoute.CHAT

        skill = self.skills.for_route(route)
        tools = skill.tools if skill else ()
        available = tuple(name for name in tools if self.registry.get(name) is not None)
        steps = [PlanStep("understand", "结合当前会话与项目上下文确认目标")]
        if skill:
            steps.extend(
                PlanStep(f"workflow_{index}", description)
                for index, description in enumerate(skill.workflow, 1)
            )
        if available:
            steps.append(PlanStep(
                "execute",
                "选择最小必要工具执行，并读取工具观察结果",
                suggested_tools=available,
                requires_observation=True,
            ))
        steps.append(PlanStep("respond", "汇总结果、限制与下一步建议"))
        return AgentPlan(
            objective=query,
            route=route,
            steps=tuple(steps),
            available_tools=available,
            requires_confirmation=bool(self._WRITE.search(query)),
        )
