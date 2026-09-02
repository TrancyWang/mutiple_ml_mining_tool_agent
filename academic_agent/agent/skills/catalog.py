"""将产品能力组织为 Planner 可消费的标准工作流。"""

from __future__ import annotations

from dataclasses import dataclass

from academic_agent.agent.types import AgentRoute


@dataclass(frozen=True, slots=True)
class AgentSkill:
    name: str
    route: AgentRoute
    description: str
    tools: tuple[str, ...]
    workflow: tuple[str, ...]


class SkillCatalog:
    def __init__(self) -> None:
        definitions = (
            AgentSkill(
                "document_rag",
                AgentRoute.DOCUMENT,
                "对用户上传或当前工作区中的文档进行结构化解析和带引用检索",
                ("document_rag",),
                ("确定文档范围", "解析并检索相关证据", "根据证据回答并保留引用"),
            ),
            AgentSkill(
                "project_workspace",
                AgentRoute.PROJECT,
                "在当前项目边界内检索、读取与生成文件",
                (
                    "list_project_files", "glob_project_files", "grep_project_files",
                    "read_project_file", "generate_project_file", "edit_project_file",
                ),
                ("定位目标文件", "读取最小必要内容", "执行或预览文件操作"),
            ),
            AgentSkill(
                "data_mining",
                AgentRoute.DATA,
                "完成数据理解、文本预处理与文本挖掘",
                (
                    "profile_data", "load_data", "data_statistics", "data_preprocess",
                    "preprocess_text", "sentiment_analysis",
                    "text_clustering", "extract_keywords",
                ),
                ("识别文件结构与字段", "确认文本列和参数", "执行分析并汇总质量指标"),
            ),
            AgentSkill(
                "machine_learning",
                AgentRoute.MACHINE_LEARNING,
                "完成特征选择、建模与评估",
                ("data_preprocess", "feature_processing", "classification", "regression", "causal_inference"),
                ("确认目标变量与特征", "选择模型并执行", "解释指标与限制"),
            ),
            AgentSkill(
                "visualization",
                AgentRoute.VISUALIZATION,
                "生成可在对话框和工作区查看的图表",
                ("line_chart", "wordcloud", "cluster_plot", "sentiment_plot"),
                ("确认横纵轴或文本列", "生成图表文件", "返回状态与图片产物"),
            ),
        )
        self._by_route = {item.route: item for item in definitions}

    def for_route(self, route: AgentRoute) -> AgentSkill | None:
        return self._by_route.get(route)

    def all(self) -> tuple[AgentSkill, ...]:
        return tuple(self._by_route.values())


skill_catalog = SkillCatalog()
