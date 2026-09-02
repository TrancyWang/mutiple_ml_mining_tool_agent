"""Offline evaluation for the first system-paper experiments.

The script deliberately evaluates deterministic system layers first. It does
not call a remote LLM, so results can be reproduced on a clean machine with
the project dependencies already installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.policies.tool_access import AgentPolicy, PolicyViolation
from academic_agent.agent.session import session_store
from academic_agent.agent.tooling.builtins import register_builtin_tools
from academic_agent.agent.tooling.registry import tool_registry
from academic_agent.agent.types import AgentPlan, AgentRequest, AgentRoute, ExecutionScope
from academic_agent.infrastructure.workspace_manager import WorkspaceManager, workspace_manager


@dataclass(frozen=True)
class PlannerTask:
    task_id: str
    category: str
    query: str
    expected_route: str
    expected_tools: tuple[str, ...] = ()
    requires_confirmation: bool = False
    chat_only: bool = False


@dataclass(frozen=True)
class SecurityCase:
    case_id: str
    description: str
    tool: str
    params: dict[str, Any]
    workspace_path: str | None
    chat_only: bool
    expected_denied: bool


def planner_tasks() -> list[PlannerTask]:
    """Small, hand-audited benchmark with a gold route/tool scope."""

    return [
        PlannerTask("data-01", "data", "加载当前工作区的 CSV 数据并查看字段结构", "data", ("load_data", "profile_data")),
        PlannerTask("data-02", "data", "对数据进行缺失值处理和重复行清理", "data", ("data_preprocess",)),
        PlannerTask("data-03", "data", "对评论文本做中文文本预处理", "data", ("preprocess_text",)),
        PlannerTask("data-04", "data", "进行中文情感分析并统计情绪分布", "data", ("sentiment_analysis",)),
        PlannerTask("data-05", "data", "对文本进行聚类并提取每个主题的关键词", "data", ("text_clustering", "extract_keywords")),
        PlannerTask("ml-01", "machine_learning", "使用特征训练一个分类模型", "machine_learning", ("feature_processing", "classification")),
        PlannerTask("ml-02", "machine_learning", "使用特征训练回归模型并评估预测效果", "machine_learning", ("feature_processing", "regression")),
        PlannerTask("ml-03", "machine_learning", "估计处理变量对结果变量的因果影响", "machine_learning", ("causal_inference",)),
        PlannerTask("ml-04", "machine_learning", "进行机器学习预测并选择重要特征", "machine_learning", ("feature_processing",)),
        PlannerTask("ml-05", "machine_learning", "训练模型并分析目标变量和特征之间的关系", "machine_learning", ("feature_processing",)),
        PlannerTask("viz-01", "visualization", "根据时间字段绘制折线图", "visualization", ("line_chart",)),
        PlannerTask("viz-02", "visualization", "生成评论文本词云", "visualization", ("wordcloud",)),
        PlannerTask("viz-03", "visualization", "绘制聚类分布图", "visualization", ("cluster_plot",)),
        PlannerTask("viz-04", "visualization", "绘制情感分布图", "visualization", ("sentiment_plot",)),
        PlannerTask("project-01", "project", "列出当前项目工作区中的 Python 文件", "project", ("list_project_files",)),
        PlannerTask("project-02", "project", "按文件名模式查找项目中的分析脚本", "project", ("glob_project_files",)),
        PlannerTask("project-03", "project", "在项目文件中检索 Planner 的实现位置", "project", ("grep_project_files",)),
        PlannerTask("project-04", "project", "读取当前项目中的配置文件", "project", ("read_project_file",)),
        PlannerTask("project-05", "project", "生成一份分析说明文档", "project", ("generate_project_file",), True),
        PlannerTask("project-06", "project", "编辑项目中的分析脚本", "project", ("edit_project_file",), True),
        PlannerTask("chat-01", "chat", "解释什么是交叉验证", "chat", (), False, True),
        PlannerTask("chat-02", "chat", "帮我比较回归和分类的区别", "chat", (), False, True),
    ]


def _request(task: PlannerTask) -> AgentRequest:
    return AgentRequest(
        messages=[{"role": "user", "content": task.query}],
        user_id="offline-eval-user",
        session_id=f"offline-eval-{task.task_id}",
        workspace_path=str(REPO_ROOT),
        chat_only=task.chat_only,
    )


def evaluate_planner(tasks: list[PlannerTask]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planner = TaskPlanner()
    all_tools = {spec.name for spec in tool_registry.all()}
    rows: list[dict[str, Any]] = []

    for task in tasks:
        plan = planner.create_plan(_request(task))
        available = set(plan.available_tools)
        expected = set(task.expected_tools)
        overlap = expected.intersection(available)
        route_correct = plan.route.value == task.expected_route
        confirmation_correct = plan.requires_confirmation == task.requires_confirmation
        coverage = len(overlap) / len(expected) if expected else 1.0
        scope_precision = len(overlap) / len(available) if available else (1.0 if not expected else 0.0)
        planner_success = route_correct and confirmation_correct and coverage == 1.0

        rows.append({
            "task_id": task.task_id,
            "category": task.category,
            "query": task.query,
            "expected_route": task.expected_route,
            "predicted_route": plan.route.value,
            "route_correct": route_correct,
            "expected_tools": ",".join(task.expected_tools),
            "available_tools": ",".join(plan.available_tools),
            "expected_tool_coverage": round(coverage, 4),
            "scope_precision": round(scope_precision, 4),
            "scope_size": len(available),
            "all_tools_size": len(all_tools),
            "confirmation_expected": task.requires_confirmation,
            "confirmation_predicted": plan.requires_confirmation,
            "confirmation_correct": confirmation_correct,
            "planner_success": planner_success,
        })

    total = len(rows)
    summary = {
        "task_count": total,
        "route_accuracy": round(sum(row["route_correct"] for row in rows) / total, 4),
        "confirmation_accuracy": round(sum(row["confirmation_correct"] for row in rows) / total, 4),
        "tool_coverage": round(sum(row["expected_tool_coverage"] for row in rows) / total, 4),
        "scope_precision": round(sum(row["scope_precision"] for row in rows) / total, 4),
        "average_planner_scope_size": round(sum(row["scope_size"] for row in rows) / total, 4),
        "registered_tool_count": len(all_tools),
        "planner_success_rate": round(sum(row["planner_success"] for row in rows) / total, 4),
    }
    return rows, summary


def security_cases() -> list[SecurityCase]:
    workspace_file = str(REPO_ROOT / "academic_agent" / "agent" / "runtime.py")
    outside_file = str(Path(tempfile.gettempdir()) / "academic_agent_outside_input.csv")
    return [
        SecurityCase("sec-01", "Chat 模式调用数据工具", "load_data", {"file_path": workspace_file}, None, True, True),
        SecurityCase("sec-02", "没有工作区时读取项目文件", "read_project_file", {"path": "academic_agent/agent/runtime.py"}, None, False, True),
        SecurityCase("sec-03", "从工作区外加载未上传文件", "load_data", {"file_path": outside_file}, str(REPO_ROOT), False, True),
        SecurityCase("sec-04", "读取当前工作区文件", "read_project_file", {"path": "academic_agent/agent/runtime.py"}, str(REPO_ROOT), False, False),
        SecurityCase("sec-05", "Chat 模式执行临时代码", "execute_python_analysis", {"script": "print(1)"}, str(REPO_ROOT), True, True),
        SecurityCase("sec-06", "Work 模式执行临时代码", "execute_python_analysis", {"script": "print(1)"}, str(REPO_ROOT), False, False),
    ]


def evaluate_security(cases: list[SecurityCase]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = AgentPolicy()
    rows: list[dict[str, Any]] = []
    session_store.clear("security-eval")
    for case in cases:
        request = AgentRequest(
            messages=[{"role": "user", "content": case.description}],
            user_id="offline-security-user",
            session_id="security-eval",
            workspace_path=case.workspace_path,
            chat_only=case.chat_only,
        )
        route = AgentRoute.CHAT if case.chat_only else AgentRoute.PROJECT
        plan = AgentPlan(case.description, route, ())
        scope = ExecutionScope(request=request, plan=plan)
        spec = tool_registry.get(case.tool)
        if spec is None:
            raise RuntimeError(f"工具未注册：{case.tool}")
        denied = False
        error = ""
        try:
            policy.authorize(spec, case.params, scope)
        except PolicyViolation as exc:
            denied = True
            error = str(exc)
        passed = denied == case.expected_denied
        rows.append({
            "case_id": case.case_id,
            "description": case.description,
            "tool": case.tool,
            "expected_denied": case.expected_denied,
            "actual_denied": denied,
            "passed": passed,
            "error": error,
        })

    total = len(rows)
    summary = {
        "case_count": total,
        "security_pass_rate": round(sum(row["passed"] for row in rows) / total, 4),
        "unauthorized_cases": sum(row["expected_denied"] for row in rows),
        "unauthorized_intercepted": sum(row["expected_denied"] and row["actual_denied"] for row in rows),
        "false_positive_allowed_cases": sum((not row["expected_denied"]) and row["actual_denied"] for row in rows),
    }
    return rows, summary


def evaluate_confirmation() -> dict[str, Any]:
    """Test preview-before-write and explicit confirmation in a temp workspace."""
    with tempfile.TemporaryDirectory(prefix="academic-agent-confirmation-") as temp_dir:
        manager = WorkspaceManager(temp_dir)
        preview = manager.prepare_generate("draft/result.txt", "experiment result\n")
        target = Path(temp_dir) / "draft" / "result.txt"
        preview_passed = bool(preview.get("requires_confirmation")) and not target.exists()
        confirmed = manager.confirm(str(preview["operation_id"])) if preview.get("operation_id") else {"success": False}
        confirm_passed = bool(confirmed.get("success")) and target.is_file() and target.read_text(encoding="utf-8") == "experiment result\n"
        return {
            "permission_mode": manager.permission_mode,
            "preview": preview,
            "confirmed": confirmed,
            "preview_before_write_passed": preview_passed,
            "explicit_confirmation_passed": confirm_passed,
            "passed": preview_passed and confirm_passed,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, planner: dict[str, Any], security: dict[str, Any], confirmation: dict[str, Any]) -> None:
    path.write_text(
        "# Offline agent system evaluation\n\n"
        "## Planner and tool scope\n\n"
        f"- Tasks: {planner['task_count']}\n"
        f"- Route accuracy: {planner['route_accuracy']:.2%}\n"
        f"- Confirmation accuracy: {planner['confirmation_accuracy']:.2%}\n"
        f"- Required tool coverage: {planner['tool_coverage']:.2%}\n"
        f"- Average scope size: {planner['average_planner_scope_size']:.2f} / {planner['registered_tool_count']} registered tools\n"
        f"- Planner success rate: {planner['planner_success_rate']:.2%}\n\n"
        "## Security policy\n\n"
        f"- Cases: {security['case_count']}\n"
        f"- Security pass rate: {security['security_pass_rate']:.2%}\n"
        f"- Unauthorized operations intercepted: {security['unauthorized_intercepted']} / {security['unauthorized_cases']}\n"
        f"- False-positive allowed cases: {security['false_positive_allowed_cases']}\n\n"
        "## Confirmation workflow\n\n"
        f"- Preview before write: {confirmation['preview_before_write_passed']}\n"
        f"- Explicit confirmation: {confirmation['explicit_confirmation_passed']}\n\n"
        "## Interpretation\n\n"
        "These are deterministic offline system-layer results. They should be reported separately from a future end-to-end experiment with a real LLM, where task success, latency, retries and token cost are measured over repeated runs.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline experiments for the academic agent system paper")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"), help="directory for generated reports")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importing/registering once makes the experiment reflect the actual tool surface.
    register_builtin_tools()
    planner_rows, planner_summary = evaluate_planner(planner_tasks())
    security_rows, security_summary = evaluate_security(security_cases())
    confirmation_summary = evaluate_confirmation()
    summary = {
        "experiment": "offline_agent_system_evaluation",
        "repository": str(REPO_ROOT),
        "planner": planner_summary,
        "security": security_summary,
        "confirmation": {
            key: value for key, value in confirmation_summary.items()
            if key not in {"preview", "confirmed"}
        },
    }

    write_csv(output_dir / "planner_results.csv", planner_rows)
    write_csv(output_dir / "security_results.csv", security_rows)
    (output_dir / "confirmation_results.json").write_text(json.dumps(confirmation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", planner_summary, security_summary, confirmation_summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
