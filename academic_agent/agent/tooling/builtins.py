"""注册 Agent 工具并处理会话作用域，不承载具体业务实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from academic_agent.agent.session import SessionContext
from academic_agent.agent.tooling.handlers import data, mining, ml, project, visualization
from academic_agent.agent.tooling.registry import ToolSpec, tool_registry
from academic_agent.infrastructure.workspace_manager import workspace_manager


def register_builtin_tools() -> None:
    specs = [
        ToolSpec("profile_data", "数据概览", "数据理解", "识别文件结构、字段角色和数据质量", data.profile, requires_data=False),
        ToolSpec("load_data", "加载数据", "数据理解", "加载 CSV、Excel、JSON 或文本数据", data.load_data, requires_data=False),
        ToolSpec("get_data_info", "数据状态", "数据理解", "读取当前已加载数据的字段和规模", data.get_data_info),
        ToolSpec("data_statistics", "数据统计分析", "数据理解", "生成数据质量、描述统计、类别分布和相关性报告", data.data_statistics),
        ToolSpec("data_preprocess", "数据预处理", "数据理解", "执行缺失值处理和重复行处理，原始数据不覆盖", data.data_preprocess),
        ToolSpec("feature_processing", "特征处理", "机器学习", "识别数值/类别特征并完成填充、编码和标准化", data.feature_processing),
        ToolSpec("preprocess_text", "文本预处理", "文本挖掘", "调用 video_text_mutiplemodal_agent.Data_PreProcessor", mining.preprocess),
        ToolSpec("sentiment_analysis", "情感分析", "文本挖掘", "调用 video_text_mutiplemodal_agent 情感模型", mining.sentiment),
        ToolSpec("text_clustering", "文本聚类", "文本挖掘", "调用 video_text_mutiplemodal_agent.VideoTextClustering", mining.clustering),
        ToolSpec("extract_keywords", "关键词提取", "文本挖掘", "调用 video_text_mutiplemodal_agent.TextKeyBert", mining.keywords),
        ToolSpec("regression", "回归分析", "机器学习", "调用项目 algorithms/machine_learning 回归类", ml.regression),
        ToolSpec("classification", "分类分析", "机器学习", "调用项目 SVMClassificationTool", ml.classification),
        ToolSpec("causal_inference", "因果推断", "机器学习", "调用项目因果推断类", ml.causal),
        ToolSpec("wordcloud", "词云图", "可视化", "生成文本词云", visualization.wordcloud),
        ToolSpec("cluster_plot", "聚类分布图", "可视化", "生成聚类分布图", visualization.cluster_plot),
        ToolSpec("sentiment_plot", "情感分布图", "可视化", "生成情感分布图", visualization.sentiment_plot),
        ToolSpec("line_chart", "折线图", "可视化", "按时间或顺序字段绘制折线图", visualization.line_chart),
        ToolSpec("list_project_files", "列出项目文件", "项目工作区", "列出当前项目工作区内可操作的文件", project.list_project_files, requires_data=False),
        ToolSpec("search_project_files", "检索项目文件", "项目工作区", "在当前项目文件中按关键词检索", project.search_project_files, requires_data=False),
        ToolSpec("glob_project_files", "按名称查找项目文件", "项目工作区", "按路径或文件名模式查找当前项目文件", project.glob_project_files, requires_data=False),
        ToolSpec("grep_project_files", "检索项目文件内容", "项目工作区", "按关键词检索项目文件内容并返回行号", project.grep_project_files, requires_data=False),
        ToolSpec("read_project_file", "读取项目文件", "项目工作区", "读取当前项目中的指定文件", project.read_project_file, requires_data=False),
        ToolSpec("generate_project_file", "生成项目文件", "项目工作区", "生成文档并写入当前工作区 output 目录", project.generate_project_file, requires_data=False),
        ToolSpec("execute_python_analysis", "临时代码分析", "项目工作区", "在临时工作区执行受控 Python 分析代码", project.execute_python_analysis, requires_data=False),
        ToolSpec("edit_project_file", "编辑项目文件", "项目工作区", "预览并编辑当前项目文件，写入前需要确认", project.edit_project_file, requires_data=False),
        ToolSpec("delete_project_file", "删除项目文件", "项目工作区", "预览并删除当前项目文件，删除前需要确认", project.delete_project_file, requires_data=False),
        ToolSpec("confirm_workspace_operation", "确认文件操作", "项目工作区", "确认执行待处理的生成、编辑或删除操作", project.confirm_workspace_operation, requires_data=False),
    ]
    for spec in specs:
        tool_registry.register(spec)


def execute_tool(name: str, context: SessionContext | None = None, **kwargs: Any) -> dict[str, Any]:
    spec = tool_registry.get(name)
    if spec is None:
        return {"success": False, "error": f"未知工具：{name}"}
    if spec.requires_data:
        from academic_agent.tools.text_mining_tools import text_mining_tools

        if text_mining_tools.current_data is None:
            return {"success": False, "error": "请先上传并加载数据文件。"}
    if context is not None:
        if name == "load_data":
            candidate = Path(str(kwargs.get("file_path", ""))).expanduser()
            if not candidate.is_absolute():
                candidate = workspace_manager.root / candidate
            uploaded = {Path(path).expanduser().resolve() for path in context.uploaded_files}
            candidate = candidate.resolve()
            kwargs["file_path"] = str(candidate)
            kwargs["_source_scope"] = "upload" if candidate in uploaded else "workspace"
        elif name == "execute_python_analysis":
            kwargs["_allowed_uploaded_files"] = list(context.uploaded_files)
    result = tool_registry.execute(name, **kwargs)
    if context is not None:
        context.record_task(name, result)
    return result


register_builtin_tools()
