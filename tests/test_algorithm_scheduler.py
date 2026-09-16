import json

from academic_agent.agent.planning.algorithm_scheduler import (
    fallback_algorithm_schedule,
    normalize_algorithm_schedule,
)
from academic_agent.agent.runtime import AgentRuntime
from academic_agent.agent.types import AgentPlan, AgentRequest, AgentRoute, TaskBoard


def test_scheduler_rejects_unknown_tools_and_keeps_local_tool_only():
    result = normalize_algorithm_schedule(
        {
            "algorithm_decision": {"algorithm": "random_forest"},
            "tasks": [
                {
                    "task_title": "先清洗数据",
                    "task_goal": "处理缺失值",
                    "scheduled_tool": "data_preprocess",
                    "scheduled_arguments": {},
                },
                {
                    "task_title": "越权工具",
                    "task_goal": "不应执行",
                    "scheduled_tool": "shell_exec",
                    "scheduled_arguments": {"command": "rm -rf /"},
                },
            ],
        },
        ["data_preprocess", "regression"],
    )

    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["scheduled_tool"] == "data_preprocess"
    assert result["errors"]


def test_scheduler_preserves_user_confirmed_algorithm_as_hard_constraint():
    result = normalize_algorithm_schedule(
        {
            "tasks": [{
                "task_title": "训练模型",
                "task_goal": "预测销售额",
                "scheduled_tool": "regression",
                "scheduled_arguments": {
                    "target_var": "销售额",
                    "feature_vars": ["广告费"],
                    "model_type": "linear",
                },
            }],
        },
        ["regression", "classification"],
        configuration={
            "kind": "regression",
            "algorithm": "xgboost",
            "tool_arguments": {"model_type": "xgboost", "multiple_folds": 3},
        },
    )

    task = result["tasks"][0]
    assert task["scheduled_arguments"]["model_type"] == "xgboost"
    assert task["scheduled_arguments"]["multiple_folds"] == 3


def test_scheduler_moves_missing_model_fields_to_model_completion_path():
    result = normalize_algorithm_schedule(
        {
            "tasks": [{
                "task_title": "训练模型",
                "task_goal": "预测销售额",
                "scheduled_tool": "regression",
                "scheduled_arguments": {"model_type": "random_forest"},
            }],
        },
        ["regression"],
    )

    task = result["tasks"][0]
    assert task["suggested_tools"] == ["regression"]
    assert task["scheduled_tool"] == ""
    assert result["errors"]


def test_algorithm_schedule_becomes_program_managed_task_board():
    plan = AgentPlan(
        "预测销售额",
        AgentRoute.MACHINE_LEARNING,
        (),
        available_tools=("regression",),
        algorithm_schedule=(
            {
                "task_title": "训练随机森林",
                "task_goal": "预测销售额",
                "deliverable": "回归指标",
                "done_when": "工具返回 success=true",
                "suggested_tools": ["regression"],
                "scheduled_tool": "regression",
                "scheduled_arguments": {
                    "target_var": "销售额",
                    "feature_vars": ["广告费"],
                    "model_type": "random_forest",
                },
            },
        ),
    )
    board = TaskBoard.from_task_plan(
        plan,
        {
            "tasks": list(plan.algorithm_schedule),
            "board_goal": plan.objective,
            "final_deliverable": "回归指标",
        },
    )

    task = board.tasks["T1"]
    assert task.scheduled_tool == "regression"
    assert task.scheduled_arguments["model_type"] == "random_forest"
    assert task.depends_on == ()


def test_fallback_preserves_old_model_driven_execution():
    result = fallback_algorithm_schedule(
        "classification",
        {"algorithm": "svm", "tool_arguments": {"model_type": "svm"}},
    )

    assert result["fallback_used"] is True
    assert result["tasks"][0]["suggested_tools"] == ["classification"]
    assert result["tasks"][0]["scheduled_tool"] == ""


def test_runtime_uses_llm_schedule_as_program_managed_plan():
    runtime = AgentRuntime.__new__(AgentRuntime)
    plan = AgentPlan(
        "预测销售额",
        AgentRoute.MACHINE_LEARNING,
        (),
        available_tools=("feature_processing", "regression"),
    )
    payload = {
        "algorithm_decision": {"algorithm": "random_forest"},
        "tasks": [{
            "task_title": "训练随机森林",
            "task_goal": "预测销售额",
            "deliverable": "回归指标",
            "done_when": "工具返回 success=true",
            "scheduled_tool": "regression",
            "scheduled_arguments": {
                "target_var": "销售额",
                "feature_vars": ["广告费"],
                "model_type": "random_forest",
            },
        }],
    }
    events = []

    scheduled = runtime._schedule_algorithms(
        AgentRequest([{"role": "user", "content": "预测销售额"}], "u", "s"),
        plan,
        [],
        None,
        lambda _messages: [{"role": "assistant", "content": json.dumps(payload)}],
        events.append,
    )

    assert scheduled.algorithm_decision["algorithm"] == "random_forest"
    assert scheduled.algorithm_schedule[0]["scheduled_tool"] == "regression"
    assert events[0]["type"] == "algorithm_schedule_created"
