"""Run a small end-to-end benchmark against the configured Gemini account.

This runner uses the project's existing ``analysis_tool`` contract and the
same ToolExecutor/AgentPolicy stack as the desktop application, but talks to
Gemini through its OpenAI-compatible endpoint directly. It intentionally uses
small, safe tasks and a temporary uploaded CSV so the experiment does not
modify the repository.
"""

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
from academic_agent.agent.policies.tool_access import AgentPolicy
from academic_agent.agent.session import session_store
from academic_agent.agent.tooling.builtins import register_builtin_tools
from academic_agent.agent.tooling.registry import tool_registry
from academic_agent.agent.types import AgentPlan, AgentRequest, AgentRoute, ExecutionScope
from academic_agent.controllers.auth import AuthController
from academic_agent.agent.providers.config import get_llm_config_priority
from academic_agent.infrastructure.workspace_manager import workspace_manager


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analysis_tool",
        "description": (
            "Academic Agent 的统一分析工具。先选择 action，再在 params 中传递参数。"
            "只能使用当前工作区或用户上传文件；生成、编辑和删除文件会先返回确认预览。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [spec.name for spec in tool_registry.all()],
                },
                "params": {"type": "object"},
            },
            "required": ["action"],
        },
    },
}


SYSTEM_PROMPT = """你是用于实验的 Academic Agent。你必须使用 analysis_tool 完成能由工具完成的任务。
始终用中文回答。先根据用户任务选择最小必要工具；工具返回结果后再总结。
不要调用未提供的工具，不要访问工作区之外的文件。生成、编辑或删除文件时不要确认执行，
只保留工具返回的预览结果，由实验程序记录待确认状态。"""


TASKS = [
    {
        "task_id": "live-01",
        "query": "请加载我上传的 CSV 文件 {csv_path}，然后查看数据字段和规模。",
        # load_data itself returns rows, columns and column_names, so requiring
        # profile_data here would penalize a valid minimal tool sequence.
        "expected_tools": {"load_data"},
        "route": "data",
    },
    {
        "task_id": "live-02",
        "query": "基于当前已经加载的数据，生成描述统计和数据质量报告。",
        "expected_tools": {"data_statistics"},
        "route": "data",
    },
    {
        "task_id": "live-03",
        "query": "读取项目中的 academic_agent/agent/planning/planner.py，并告诉我它如何进行任务路由。",
        "expected_tools": {"read_project_file"},
        "route": "project",
    },
    {
        "task_id": "live-04",
        "query": "在项目文件中检索 ToolExecutor 的定义位置，并汇报命中的文件和行号。",
        "expected_tools": {"grep_project_files"},
        "acceptable_tools": {"grep_project_files", "search_project_files"},
        "route": "project",
    },
    {
        "task_id": "live-05",
        "query": "根据当前数据的 year 和 score 字段绘制一张折线图。",
        "expected_tools": {"line_chart"},
        "route": "visualization",
    },
    {
        "task_id": "live-06",
        "query": "生成一份名为 output/live_experiment_notes.md 的简短分析说明文档。",
        "expected_tools": {"generate_project_file"},
        "route": "project",
    },
]


def _load_config() -> dict[str, Any]:
    AuthController().load_root_env()
    # Explicitly request Gemini so a local-model setting in .env cannot cause
    # this experiment to silently use Ollama.
    config = get_llm_config_priority(model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"), provider="gemini")
    return config


def _make_csv(directory: str) -> Path:
    path = Path(directory) / "academic_agent_eval.csv"
    rows = [
        {"id": 1, "year": 2021, "score": 0.72, "text": "数据分析提升研究效率"},
        {"id": 2, "year": 2022, "score": 0.81, "text": "智能工具降低分析门槛"},
        {"id": 3, "year": 2023, "score": 0.86, "text": "可追溯流程有助于复现"},
        {"id": 4, "year": 2024, "score": 0.91, "text": "自然语言交互改善体验"},
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _run_one(client: OpenAI, config: dict[str, Any], task: dict[str, Any], csv_path: Path) -> dict[str, Any]:
    session_id = f"live-eval-{uuid.uuid4().hex[:10]}"
    session = session_store.get(session_id)
    session.attach_file(str(csv_path))
    query = task["query"].format(csv_path=str(csv_path))
    request = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"当前项目工作区是 {REPO_ROOT}。你有一个用户上传文件：{csv_path}。\n\n{query}"
                ),
            },
        ],
        "user_id": "trancy",
        "session_id": session_id,
        "workspace_path": str(REPO_ROOT),
    }
    agent_request = AgentRequest(
        messages=request["messages"],
        user_id="trancy",
        session_id=session_id,
        workspace_path=str(REPO_ROOT),
    )
    scope = ExecutionScope(
        request=agent_request,
        plan=AgentPlan(task["query"], AgentRoute(task["route"]), ()),
    )
    messages = list(request["messages"])
    tool_calls = []
    assistant_text = ""
    error = ""
    started = time.perf_counter()

    try:
        with tool_executor.bind(scope):
            for _ in range(6):
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    tools=[TOOL_SCHEMA],
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=1200,
                )
                message = response.choices[0].message
                assistant_text = message.content or assistant_text
                calls = message.tool_calls or []
                if not calls:
                    break
                messages.append(message.model_dump(exclude_none=True))
                for call in calls:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        action = str(arguments.get("action", ""))
                        params = arguments.get("params") or {}
                        result = tool_executor.execute(action, **params)
                        tool_calls.append({
                            "action": action,
                            "params": _json_safe(params),
                            "success": bool(result.get("success")),
                            "requires_confirmation": bool(result.get("requires_confirmation")),
                            "error": result.get("error", ""),
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(_json_safe(result), ensure_ascii=False),
                        })
                    except Exception as exc:
                        error = f"tool_call_error: {type(exc).__name__}: {exc}"
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"success": False, "error": error})})
    except Exception as exc:
        error = f"model_error: {type(exc).__name__}: {exc}"

    actions = {item["action"] for item in tool_calls}
    successful_actions = {
        item["action"] for item in tool_calls
        if item["success"] or item["requires_confirmation"]
    }
    expected = set(task["expected_tools"])
    acceptable = set(task.get("acceptable_tools", expected))
    required_found = expected.issubset(actions)
    required_success = expected.issubset(successful_actions)
    acceptable_success = bool(acceptable.intersection(successful_actions))
    unexpected = actions - expected
    operation_preview = any(item["requires_confirmation"] for item in tool_calls)
    clean_execution = acceptable_success and not error and all(
        item["success"] or item["requires_confirmation"] for item in tool_calls
    )
    # Functional success allows a semantically equivalent tool and recovery
    # after a parameter error; clean_execution reports the stricter metric.
    success = acceptable_success and not error
    return {
        "task_id": task["task_id"],
        "query": query,
        "expected_route": task["route"],
        "expected_tools": sorted(expected),
        "acceptable_tools": sorted(acceptable),
        "actual_tools": sorted(actions),
        "successful_tools": sorted(successful_actions),
        "tool_call_count": len(tool_calls),
        "required_tools_found": required_found,
        "required_tools_success": required_success,
        "acceptable_tool_success": acceptable_success,
        "clean_execution": clean_execution,
        "unexpected_tools": sorted(unexpected),
        "confirmation_preview_created": operation_preview,
        "task_success": success,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tool_calls": tool_calls,
        "assistant_text": assistant_text,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Gemini experiments using the configured project account")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results_live_gemini"))
    parser.add_argument("--limit", type=int, default=len(TASKS), help="number of live tasks to run")
    args = parser.parse_args()

    register_builtin_tools()
    config = _load_config()
    client = OpenAI(api_key=config["api_key"], base_url=config["model_server"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The CSV exists only in a temporary upload scope and is deleted afterwards.
    with tempfile.TemporaryDirectory(prefix="academic-agent-live-") as temp_dir:
        csv_path = _make_csv(temp_dir)
        results = [_run_one(client, config, task, csv_path) for task in TASKS[: max(0, args.limit)]]

    summary = {
        "provider": "gemini",
        "model": config["model"],
        "task_count": len(results),
        "task_success_rate": sum(item["task_success"] for item in results) / max(len(results), 1),
        "required_tool_selection_rate": sum(item["required_tools_found"] for item in results) / max(len(results), 1),
        "required_tool_success_rate": sum(item["required_tools_success"] for item in results) / max(len(results), 1),
        "clean_execution_rate": sum(item["clean_execution"] for item in results) / max(len(results), 1),
        "average_tool_call_count": sum(item["tool_call_count"] for item in results) / max(len(results), 1),
        "average_elapsed_seconds": sum(item["elapsed_seconds"] for item in results) / max(len(results), 1),
        "confirmation_previews": sum(item["confirmation_preview_created"] for item in results),
    }
    payload = {"summary": summary, "results": results}
    (output_dir / "live_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output_dir / 'live_results.json')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
