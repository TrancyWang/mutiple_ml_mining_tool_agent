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
                "skill_management",
                AgentRoute.SKILL_MANAGEMENT,
                "列出并安装包含 SKILL.md 的指令型 Skill 包",
                ("list_installed_skills", "install_skill"),
                ("判断 Skill 来源和安装目标", "校验并安装 Skill 包", "返回安装结果并说明后续生效方式"),
            ),
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
                "在当前项目边界内检索、读取、创建与修改代码文件",
                (
                    "list_project_files", "glob_project_files", "grep_project_files",
                    "read_project_file", "write_project_file", "edit_project_file",
                    "generate_project_file",
                ),
                ("定位目标文件", "读取最小必要内容", "创建或修改代码并预览文件操作"),
            ),
            AgentSkill(
                "data_mining",
                AgentRoute.DATA,
                "完成数据理解、文本预处理与文本挖掘",
                (
                    "profile_data", "load_data", "get_data_info", "data_statistics", "data_preprocess",
                    "preprocess_text", "sentiment_analysis",
                    "text_clustering", "extract_keywords", "entity_recognition",
                    "relation_extraction",
                ),
                ("识别文件结构与字段", "确认文本列和参数", "执行分析并汇总质量指标"),
            ),
            AgentSkill(
                "text2sql",
                AgentRoute.TEXT2SQL,
                "把自然语言对齐到已注册业务指标和维度，再生成受限 SQL 查询 MySQL",
                ("text2sql_context", "text2sql_execute"),
                (
                    "读取当前 MySQL Schema 和业务指标目录",
                    "完成业务语义对齐并拒绝未知或歧义口径",
                    "执行受限查询并返回指标口径、来源表、SQL 和结果",
                ),
            ),
            AgentSkill(
                "machine_learning",
                AgentRoute.MACHINE_LEARNING,
                "完成特征选择、建模与评估",
                ("profile_data", "load_data", "get_data_info", "data_statistics", "data_preprocess",
                 "feature_processing", "classification", "regression", "causal_inference"),
                ("确认目标变量与特征", "选择模型并执行", "解释指标与限制"),
            ),
            AgentSkill(
                "visualization",
                AgentRoute.VISUALIZATION,
                "生成可在对话框和工作区查看的图表",
                ("profile_data", "load_data", "get_data_info", "data_statistics",
                 "line_chart", "wordcloud", "cluster_plot", "sentiment_plot"),
                ("确认横纵轴或文本列", "生成图表文件", "返回状态与图片产物"),
            ),
        )
        self._by_route = {item.route: item for item in definitions}

    def for_route(self, route: AgentRoute) -> AgentSkill | None:
        return self._by_route.get(route)

    def all(self) -> tuple[AgentSkill, ...]:
        return tuple(self._by_route.values())


skill_catalog = SkillCatalog()
