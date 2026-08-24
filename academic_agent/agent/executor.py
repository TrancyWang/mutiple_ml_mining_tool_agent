"""统一 Tool Executor 与执行作用域。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import time
from typing import Any, Iterator

from academic_agent.agent.policies.tool_access import AgentPolicy, PolicyViolation
from academic_agent.agent.types import ExecutionScope
from academic_agent.agent.tooling.builtins import execute_tool
from academic_agent.agent.session import session_store
from academic_agent.agent.tooling.registry import ToolRegistry, tool_registry


_CURRENT_SCOPE: ContextVar[ExecutionScope | None] = ContextVar(
    "academic_agent_execution_scope", default=None
)


def emit_tool_log(tool: str, message: str, level: str = "info") -> None:
    """Allow long-running project algorithms to publish safe, live progress."""
    scope = _CURRENT_SCOPE.get()
    if scope is not None:
        scope.record("tool_log", tool=tool, level=level, message=str(message))

_ANALYSIS_TOOLS = {
    "load_data", "preprocess_text", "data_statistics", "data_preprocess",
    "feature_processing", "sentiment_analysis", "text_clustering", "extract_keywords",
    "regression", "classification", "causal_inference",
}


def _analysis_input_log(name: str, params: dict[str, Any]) -> str:
    from academic_agent.tools.text_mining_tools import text_mining_tools

    parts = [f"启动 {name}"]
    if text_mining_tools.current_file_path:
        parts.append(f"输入={Path(text_mining_tools.current_file_path).name}")
    if text_mining_tools.current_data is not None:
        rows, columns = text_mining_tools.current_data.shape
        parts.append(f"数据={rows} 行 × {columns} 列")
    for key in (
        "text_column", "n_clusters", "top_n", "target_var", "feature_vars",
        "model_type", "method", "treatment_var", "outcome_var",
    ):
        value = params.get(key)
        if value is not None and value != "" and value != []:
            parts.append(f"{key}={value}")
    return "；".join(parts)


def _analysis_result_logs(result: dict[str, Any], elapsed: float) -> list[str]:
    logs = [f"计算结束；耗时={elapsed:.2f}s"]
    if result.get("rows") is not None:
        logs.append(f"结果数据={result.get('rows')} 行 × {result.get('columns', '?')} 列")
    implementation = result.get("implementation") or result.get("algorithm") or result.get("model_type")
    if implementation:
        logs.append(f"实现={implementation}")
    artifacts = result.get("artifacts") or result.get("output_files") or []
    if not artifacts and result.get("output_file"):
        artifacts = [result["output_file"]]
    if artifacts:
        logs.append("输出=" + "、".join(Path(str(path)).name for path in artifacts))
    if result.get("error"):
        logs.append(f"错误={result['error']}")
    return logs


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry = tool_registry,
        policy: AgentPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or AgentPolicy()

    @contextmanager
    def bind(self, scope: ExecutionScope) -> Iterator[None]:
        token = _CURRENT_SCOPE.set(scope)
        try:
            yield
        finally:
            _CURRENT_SCOPE.reset(token)

    def execute(self, name: str, **params: Any) -> dict[str, Any]:
        scope = _CURRENT_SCOPE.get()
        spec = self.registry.get(str(name))
        if spec is None:
            return {"success": False, "error": f"未知工具：{name}"}
        try:
            self.policy.authorize(spec, params, scope)
            context = session_store.get(scope.request.session_id) if scope else None
            started_at = time.monotonic()
            if scope:
                scope.record("tool_started", tool=name, label=spec.label)
                if name in _ANALYSIS_TOOLS:
                    scope.record(
                        "tool_log", tool=name, level="info",
                        message=_analysis_input_log(name, params),
                    )
            result = execute_tool(str(name), context=context, **params)
            if scope:
                if name in _ANALYSIS_TOOLS:
                    for message in _analysis_result_logs(result, time.monotonic() - started_at):
                        scope.record(
                            "tool_log",
                            tool=name,
                            level="error" if message.startswith("错误=") else "info",
                            message=message,
                        )
                scope.record(
                    "tool_finished",
                    tool=name,
                    label=spec.label,
                    success=result.get("success", False),
                )
            return result
        except PolicyViolation as exc:
            if scope:
                scope.record("tool_denied", tool=name, label=spec.label, reason=str(exc))
            return {"success": False, "tool": name, "error": str(exc), "denied": True}
        except Exception as exc:
            if scope:
                if name in _ANALYSIS_TOOLS:
                    scope.record("tool_log", tool=name, level="error", message=f"执行异常={exc}")
                scope.record("tool_failed", tool=name, label=spec.label, error=str(exc))
            return {"success": False, "tool": name, "error": str(exc)}


tool_executor = ToolExecutor()
