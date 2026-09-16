"""LLM 算法调度计划的目录、规范化和安全兜底。

这个模块不实现算法，也不引入外部图数据库。它只把当前平台已经注册的
分析工具描述成可供规划模型选择的执行节点，并把模型返回的计划限制在
当前 Agent Plan 的 ``available_tools`` 范围内。
"""

from __future__ import annotations

from typing import Any, Iterable


# 这些是可以由调度器直接触发的无外部副作用分析工具。工作区写入工具
# 不进入这个集合，即使模型把它写入 schedule，也会被丢弃。
SCHEDULABLE_TOOLS = frozenset({
    "get_data_info",
    "data_statistics",
    "data_preprocess",
    "feature_processing",
    "preprocess_text",
    "sentiment_analysis",
    "text_clustering",
    "extract_keywords",
    "entity_recognition",
    "relation_extraction",
    "regression",
    "classification",
    "causal_inference",
    "wordcloud",
    "cluster_plot",
    "sentiment_plot",
    "line_chart",
})


ALGORITHM_CATALOG: dict[str, dict[str, Any]] = {
    "get_data_info": {
        "stage": "inspect",
        "label": "读取数据状态",
        "purpose": "读取当前数据的字段、行数和列数，为算法选择提供上下文。",
        "depends_on": [],
        "required_arguments": [],
    },
    "data_statistics": {
        "stage": "inspect",
        "label": "数据统计分析",
        "purpose": "查看数据质量、描述统计、类别分布和相关性。",
        "depends_on": [],
        "required_arguments": [],
    },
    "data_preprocess": {
        "stage": "preprocess",
        "label": "数据预处理",
        "purpose": "处理缺失值和重复行，为后续建模准备数据。",
        "depends_on": [],
        "required_arguments": [],
    },
    "feature_processing": {
        "stage": "feature",
        "label": "特征处理",
        "purpose": "识别数值/类别特征，执行填充、编码和标准化。",
        "depends_on": ["data_preprocess"],
        "required_arguments": [],
    },
    "preprocess_text": {
        "stage": "preprocess",
        "label": "文本预处理",
        "purpose": "清洗和规范化文本列。",
        "depends_on": [],
        "required_arguments": ["text_column"],
    },
    "sentiment_analysis": {
        "stage": "model",
        "label": "情感分析",
        "purpose": "对文本进行情感或情绪分类。",
        "depends_on": ["preprocess_text"],
        "required_arguments": ["text_column"],
    },
    "text_clustering": {
        "stage": "model",
        "label": "文本聚类",
        "purpose": "使用文本向量和聚类算法发现主题簇。",
        "depends_on": ["preprocess_text"],
        "required_arguments": ["text_column"],
    },
    "extract_keywords": {
        "stage": "model",
        "label": "关键词提取",
        "purpose": "从文本中提取具有代表性的关键词。",
        "depends_on": ["preprocess_text"],
        "required_arguments": ["text_column"],
    },
    "entity_recognition": {
        "stage": "model",
        "label": "实体识别",
        "purpose": "从文本中抽取实体及其类型。",
        "depends_on": ["preprocess_text"],
        "required_arguments": ["text_column"],
    },
    "relation_extraction": {
        "stage": "model",
        "label": "关系抽取",
        "purpose": "从文本中抽取主语、关系和宾语三元组。",
        "depends_on": ["entity_recognition"],
        "required_arguments": ["text_column"],
    },
    "regression": {
        "stage": "model",
        "label": "回归分析",
        "purpose": "预测连续目标，并返回 R2、RMSE、MAE 和特征重要性。",
        "depends_on": ["feature_processing"],
        "required_arguments": ["target_var", "feature_vars"],
        "algorithm_argument": "model_type",
    },
    "classification": {
        "stage": "model",
        "label": "分类分析",
        "purpose": "预测类别目标，并返回 Accuracy、Precision、Recall 和 F1。",
        "depends_on": ["feature_processing"],
        "required_arguments": ["target_var", "feature_vars"],
        "algorithm_argument": "model_type",
    },
    "causal_inference": {
        "stage": "causal",
        "label": "因果推断",
        "purpose": "在明确处理、结果和控制变量后估计处理效应。",
        "depends_on": ["feature_processing"],
        "required_arguments": ["treatment_var", "outcome_var", "control_vars"],
        "algorithm_argument": "method",
    },
    "wordcloud": {
        "stage": "visualize",
        "label": "词云图",
        "purpose": "把文本词频结果转换为词云图。",
        "depends_on": ["extract_keywords"],
        "required_arguments": ["text_column"],
    },
    "cluster_plot": {
        "stage": "visualize",
        "label": "聚类分布图",
        "purpose": "把聚类标签转换为分布图。",
        "depends_on": ["text_clustering"],
        "required_arguments": ["cluster_column"],
    },
    "sentiment_plot": {
        "stage": "visualize",
        "label": "情感分布图",
        "purpose": "把情感标签转换为分布图。",
        "depends_on": ["sentiment_analysis"],
        "required_arguments": ["sentiment_column"],
    },
    "line_chart": {
        "stage": "visualize",
        "label": "折线图",
        "purpose": "按横纵轴字段生成趋势图。",
        "depends_on": [],
        "required_arguments": ["x_column", "y_column"],
    },
}


def build_algorithm_catalog(available_tools: Iterable[str]) -> list[dict[str, Any]]:
    """只向规划模型展示当前 Plan 允许调用的工具。"""

    allowed = set(str(item) for item in available_tools)
    result: list[dict[str, Any]] = []
    for name, definition in ALGORITHM_CATALOG.items():
        if name not in allowed:
            continue
        result.append({"name": name, **definition})
    return result


def _as_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    # 保持参数为 JSON 可序列化的基础容器，避免模型返回对象被带入执行层。
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
        elif isinstance(item, list):
            result[key] = [entry for entry in item if isinstance(entry, (str, int, float, bool)) or entry is None]
    return result


def normalize_algorithm_schedule(
    raw: Any,
    available_tools: Iterable[str],
    max_tasks: int = 12,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """校验并规范化规划模型返回的算法调度计划。

    调度计划采用一个任务对应一个工具的形式。任务顺序由程序按列表顺序
    转换为 TaskBoard 的依赖顺序；模型不能自行放入未授权工具。
    """

    allowed = set(str(item) for item in available_tools)
    confirmed = configuration if isinstance(configuration, dict) else {}
    confirmed_kind_candidates = (
        str(confirmed.get("task_type") or ""),
        str(confirmed.get("kind") or ""),
    )
    confirmed_tools = {
        "regression": "regression",
        "classification": "classification",
        "causal_inference": "causal_inference",
    }
    confirmed_kind = next(
        (item for item in confirmed_kind_candidates if item in confirmed_tools),
        "",
    )
    confirmed_tool = confirmed_tools.get(confirmed_kind, "")
    confirmed_arguments = _as_arguments(confirmed.get("tool_arguments"))
    confirmed_algorithm = str(confirmed.get("algorithm") or "").strip()
    payload = raw if isinstance(raw, dict) else {}
    raw_tasks = payload.get("tasks") or payload.get("schedule") or []
    if not isinstance(raw_tasks, list):
        raw_tasks = []

    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    legacy_used = False
    for index, item in enumerate(raw_tasks[:max_tasks], start=1):
        if not isinstance(item, dict):
            errors.append(f"第 {index} 个调度项不是对象")
            continue
        tool = str(
            item.get("scheduled_tool")
            or item.get("algorithm_tool")
            or item.get("tool")
            or ""
        ).strip()
        if not tool and all(
            str(item.get(key) or "").strip()
            for key in ("task_title", "task_goal", "deliverable", "done_when")
        ):
            # 兼容旧版 Plan-to-Task 输出：它只描述任务，不绑定调度工具。
            # 这种任务仍交给主 Agent 按原逻辑补参和调用，避免一次升级
            # 因规划模型没有返回新字段而丢失整个任务板。
            raw_tools = item.get("suggested_tools") or item.get("tool_hints") or ()
            if isinstance(raw_tools, str):
                raw_tools = [raw_tools]
            tasks.append({
                "task_title": str(item["task_title"]).strip(),
                "task_goal": str(item["task_goal"]).strip(),
                "deliverable": str(item["deliverable"]).strip(),
                "done_when": str(item["done_when"]).strip(),
                "suggested_tools": [
                    str(value) for value in raw_tools if str(value) in allowed
                ],
                "scheduled_tool": "",
                "scheduled_arguments": {},
            })
            legacy_used = True
            errors.append(f"第 {index} 个调度项未绑定工具；保留旧的模型调用路径")
            continue
        if tool not in allowed or tool not in SCHEDULABLE_TOOLS:
            errors.append(f"第 {index} 个调度项使用了未授权或不可调度工具: {tool or '空'}")
            continue
        if confirmed_tool and tool in {"regression", "classification", "causal_inference"} and tool != confirmed_tool:
            errors.append(f"第 {index} 个调度项未遵守用户确认的任务类型: {confirmed_kind}")
            continue

        definition = ALGORITHM_CATALOG[tool]
        arguments = _as_arguments(
            item.get("scheduled_arguments")
            or item.get("tool_arguments")
            or item.get("arguments")
        )
        # 用户在算法确认窗口中给出的选择是硬约束，不让 LLM 在调度时
        # 悄悄替换成另一种模型或覆盖参数。
        if confirmed_tool == tool:
            for key, value in confirmed_arguments.items():
                arguments[key] = value
            if confirmed_algorithm:
                arguments.setdefault(
                    "method" if tool == "causal_inference" else "model_type",
                    confirmed_algorithm,
                )
        missing = [
            name for name in definition.get("required_arguments", [])
            if name not in arguments or arguments[name] in (None, "", [])
        ]
        # 缺少数据字段时保留任务，但不直接执行；主 Agent 可在任务上下文中
        # 补齐字段后调用同一个工具。这样不会让调度层替用户猜列名。
        direct_tool = tool if not missing else ""
        if missing:
            errors.append(f"{tool} 缺少参数: {', '.join(missing)}；转为模型补参执行")

        purpose = str(item.get("purpose") or item.get("task_goal") or definition["purpose"]).strip()
        label = str(item.get("label") or definition["label"]).strip()
        tasks.append({
            "task_title": str(item.get("task_title") or item.get("title") or label).strip(),
            "task_goal": purpose,
            "deliverable": str(item.get("deliverable") or f"{label}的结构化结果与必要产物").strip(),
            "done_when": str(item.get("done_when") or f"{tool} 返回 success=true 且有可解释结果").strip(),
            "suggested_tools": [tool],
            "scheduled_tool": direct_tool,
            "scheduled_arguments": arguments,
        })

    decision = payload.get("algorithm_decision") or payload.get("decision") or {}
    if not isinstance(decision, dict):
        decision = {"summary": str(decision)}
    return {
        "tasks": tasks,
        "algorithm_decision": decision,
        "errors": errors,
        "fallback_used": not bool(tasks) or legacy_used,
    }


def fallback_algorithm_schedule(
    route: str,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """模型输出失配时保留旧的模型驱动执行路径。"""

    config = configuration if isinstance(configuration, dict) else {}
    tool = {
        "regression": "regression",
        "classification": "classification",
        "causal_inference": "causal_inference",
    }.get(str(route), "")
    if not tool:
        return {"tasks": [], "algorithm_decision": {}, "errors": [], "fallback_used": True}

    arguments = config.get("tool_arguments")
    arguments = _as_arguments(arguments)
    algorithm = config.get("algorithm")
    if algorithm:
        arguments.setdefault(
            "method" if tool == "causal_inference" else "model_type",
            str(algorithm),
        )
    return {
        "tasks": [{
            "task_title": f"执行{config.get('algorithm_label') or ALGORITHM_CATALOG[tool]['label']}",
            "task_goal": f"调用 {tool} 完成当前任务",
            "deliverable": "算法结构化结果与必要产物",
            "done_when": f"{tool} 返回可用结果",
            "suggested_tools": [tool],
            # 兜底不直接执行，避免缺少 target_var 等字段时替用户猜测。
            "scheduled_tool": "",
            "scheduled_arguments": arguments,
        }],
        "algorithm_decision": {
            "summary": "调度模型输出不可用，保留现有工具调用路径",
            "algorithm": algorithm or "",
        },
        "errors": [],
        "fallback_used": True,
    }
