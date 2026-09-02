"""Compare a direct tool-using Agent with the planned Academic Agent."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from academic_agent.agent.executor import tool_executor
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.session import session_store
from academic_agent.agent.tooling.builtins import register_builtin_tools
from academic_agent.agent.tooling.registry import tool_registry
from academic_agent.agent.types import AgentPlan, AgentRequest
from academic_agent.controllers.auth import AuthController
from academic_agent.agent.providers.config import get_llm_config_priority


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analysis_tool",
        "description": "执行学术文本挖掘和机器学习工具。必须先选择 action，再提供 params。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": [spec.name for spec in tool_registry.all()]},
                "params": {"type": "object"},
            },
            "required": ["action"],
        },
    },
}

DIRECT_SYSTEM = (
    "你是一个通用数据分析 Agent。请根据用户需求选择合适的 analysis_tool，"
    "完成文本挖掘或机器学习任务。工具返回结果后继续完成任务，最后用中文总结。"
)
PLANNED_SYSTEM = (
    "你是 Academic Agent 的实验版本。请严格参考运行时任务计划，"
    "优先使用计划中的候选工具，选择最小必要的调用序列。工具返回错误时读取错误信息并修正参数，"
    "最后用中文总结结果和生成的分析产物。"
)

TASKS = [
    {"task_id": "text-preprocess", "domain": "text", "file": "text.csv", "expected": {"load_data", "preprocess_text"}, "query": "加载文件 {path}，对 text 列进行中文文本预处理并生成结果文件。"},
    {"task_id": "text-sentiment", "domain": "text", "file": "text.csv", "expected": {"load_data", "sentiment_analysis"}, "query": "加载文件 {path}，对 text 列进行中文情感分析并输出结果。"},
    {"task_id": "text-cluster", "domain": "text", "file": "text.csv", "expected": {"load_data", "text_clustering"}, "query": "加载文件 {path}，将 text 列聚类为 3 个主题并保存聚类结果。"},
    {"task_id": "text-keywords", "domain": "text", "file": "text.csv", "expected": {"load_data", "extract_keywords"}, "query": "加载文件 {path}，从 text 列中提取 Top 5 关键词。"},
    # feature_processing is useful but not required: the current model
    # adapters can train directly on numeric columns. The gold sequence
    # therefore measures task completion, not one rigid implementation path.
    {"task_id": "ml-classification", "domain": "ml", "file": "ml.csv", "expected": {"load_data", "classification"}, "query": "加载文件 {path}，使用 f1、f2、f3 预测 label，训练 SVM 分类模型并报告指标。"},
    {"task_id": "ml-regression", "domain": "ml", "file": "ml.csv", "expected": {"load_data", "regression"}, "query": "加载文件 {path}，使用 f1、f2、f3 预测 target，训练线性回归模型并报告指标。"},
]


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    return str(value)


def _write_datasets(directory: str) -> dict[str, Path]:
    root = Path(directory)
    text_path = root / "text.csv"
    templates = [
        "数据分析工具提升研究效率和报告质量",
        "自然语言交互降低文本挖掘的使用门槛",
        "模型结果需要保留参数和过程才能复现",
        "研究者认为可视化结果有助于理解主题",
        "自动化流程减少重复的数据清洗工作",
        "准确的情感识别可以辅助舆情研究",
    ]
    with text_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows({"id": i + 1, "text": templates[i % len(templates)]} for i in range(36))

    ml_path = root / "ml.csv"
    rows = []
    for index in range(80):
        f1 = (index % 10) / 10
        f2 = ((index * 3) % 11) / 10
        f3 = ((index * 7) % 13) / 10
        rows.append({"id": index + 1, "f1": round(f1, 4), "f2": round(f2, 4), "f3": round(f3, 4), "label": int(f1 + f2 > 0.9), "target": round(2.0 * f1 - 0.8 * f2 + 0.4 * f3, 4)})
    with ml_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "f1", "f2", "f3", "label", "target"])
        writer.writeheader()
        writer.writerows(rows)
    return {"text.csv": text_path, "ml.csv": ml_path}


def _reset_global_data() -> None:
    from academic_agent.tools.text_mining_tools import text_mining_tools

    text_mining_tools.current_data = None
    text_mining_tools.current_file_path = None
    text_mining_tools.current_source_scope = "upload"


def _config() -> dict[str, Any]:
    AuthController().load_root_env()
    default_models = REPO_ROOT.parent / "video_text_mutiplemodal_agent" / "pretrain_models"
    if not os.getenv("PRETRAINED_MODELS_DIR") and default_models.is_dir():
        os.environ["PRETRAINED_MODELS_DIR"] = str(default_models)
    return get_llm_config_priority(model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"), provider="gemini")


def _run_condition(client: OpenAI, config: dict[str, Any], task: dict[str, Any], path: Path, condition: str) -> dict[str, Any]:
    _reset_global_data()
    session_id = f"comparison-{condition}-{uuid.uuid4().hex[:10]}"
    session_store.get(session_id).attach_file(str(path))
    query = task["query"].format(path=path)
    request = AgentRequest(messages=[{"role": "user", "content": query}], user_id="comparison-user", session_id=session_id, workspace_path=str(REPO_ROOT))
    plan = TaskPlanner().create_plan(request)
    system = DIRECT_SYSTEM if condition == "direct_agent" else f"{PLANNED_SYSTEM}\n\n【运行时执行计划】\n{plan.as_prompt()}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"当前工作区是 {REPO_ROOT}。用户上传文件是 {path}。\n{query}"},
    ]
    request = AgentRequest(messages=messages, user_id="comparison-user", session_id=session_id, workspace_path=str(REPO_ROOT))
    from academic_agent.agent.types import ExecutionScope
    scope = ExecutionScope(request=request, plan=plan)
    calls: list[dict[str, Any]] = []
    assistant_text = ""
    model_error = ""
    started = time.perf_counter()
    try:
        with tool_executor.bind(scope):
            for _ in range(8):
                response = client.chat.completions.create(model=config["model"], messages=messages, tools=[TOOL_SCHEMA], tool_choice="auto", temperature=0, max_tokens=1200)
                message = response.choices[0].message
                assistant_text = message.content or assistant_text
                tool_calls = message.tool_calls or []
                if not tool_calls:
                    break
                messages.append(message.model_dump(exclude_none=True))
                for call in tool_calls:
                    try:
                        payload = json.loads(call.function.arguments or "{}")
                        action = str(payload.get("action", ""))
                        result = tool_executor.execute(action, **(payload.get("params") or {}))
                        calls.append({"action": action, "success": bool(result.get("success")), "requires_confirmation": bool(result.get("requires_confirmation")), "error": result.get("error", ""), "has_artifact": bool(result.get("output_file") or result.get("artifacts") or result.get("output_files"))})
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(_safe(result), ensure_ascii=False)})
                    except Exception as exc:
                        model_error = f"tool_call_error: {type(exc).__name__}: {exc}"
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"success": False, "error": model_error})})
    except Exception as exc:
        model_error = f"model_error: {type(exc).__name__}: {exc}"

    expected = set(task["expected"])
    actual = {call["action"] for call in calls}
    successful = {call["action"] for call in calls if call["success"] or call["requires_confirmation"]}
    failed_calls = sum(not call["success"] and not call["requires_confirmation"] for call in calls)
    return {"task_id": task["task_id"], "domain": task["domain"], "condition": condition, "expected_tools": sorted(expected), "actual_tools": sorted(actual), "successful_tools": sorted(successful), "required_tools_success": expected.issubset(successful), "task_success": expected.issubset(successful) and not model_error, "clean_execution": expected.issubset(successful) and not model_error and failed_calls == 0, "tool_call_count": len(calls), "failed_tool_calls": failed_calls, "has_artifact": any(call["has_artifact"] for call in calls), "elapsed_seconds": round(time.perf_counter() - started, 3), "plan_route": plan.route.value, "plan_scope_size": len(plan.available_tools), "calls": calls, "assistant_text": assistant_text, "error": model_error}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    result: dict[str, Any] = {"runs": len(rows), "conditions": {}}
    for condition, items in groups.items():
        n = len(items)
        result["conditions"][condition] = {"task_success_rate": sum(i["task_success"] for i in items) / n, "clean_execution_rate": sum(i["clean_execution"] for i in items) / n, "required_tools_success_rate": sum(i["required_tools_success"] for i in items) / n, "artifact_rate": sum(i["has_artifact"] for i in items) / n, "average_tool_calls": sum(i["tool_call_count"] for i in items) / n, "average_failed_tool_calls": sum(i["failed_tool_calls"] for i in items) / n, "average_elapsed_seconds": sum(i["elapsed_seconds"] for i in items) / n}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare direct and planned Gemini Agents on text/ML workflows")
    parser.add_argument("--limit", type=int, default=len(TASKS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--task-ids", default="", help="comma-separated task IDs; default runs tasks in catalog order")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results_comparison"))
    args = parser.parse_args()
    register_builtin_tools()
    config = _config()
    client = OpenAI(api_key=config["api_key"], base_url=config["model_server"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = TASKS
    if args.task_ids.strip():
        wanted = {item.strip() for item in args.task_ids.split(",") if item.strip()}
        selected_tasks = [task for task in TASKS if task["task_id"] in wanted]
        missing = wanted - {task["task_id"] for task in selected_tasks}
        if missing:
            raise SystemExit(f"unknown task ids: {sorted(missing)}")
    selected_tasks = selected_tasks[: max(0, args.limit)]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="academic-agent-comparison-") as temp_dir:
        files = _write_datasets(temp_dir)
        for repeat in range(max(1, args.repeats)):
            for task in selected_tasks:
                for condition in ("direct_agent", "planned_agent"):
                    result = _run_condition(client, config, task, files[task["file"]], condition)
                    result["repeat"] = repeat + 1
                    rows.append(result)
                    print(json.dumps({k: result[k] for k in ("repeat", "task_id", "condition", "task_success", "clean_execution", "tool_call_count", "failed_tool_calls")}, ensure_ascii=False))
    payload = {"provider": "gemini", "model": config["model"], "tasks": [task["task_id"] for task in selected_tasks], "task_count_per_condition": len(selected_tasks) * max(1, args.repeats), "summary": _summarize(rows), "results": rows}
    (output_dir / "comparison_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "output": str(output_dir / "comparison_results.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
