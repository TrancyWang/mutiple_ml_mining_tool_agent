from __future__ import annotations

from contextlib import contextmanager

from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.runtime import AgentRuntime
from academic_agent.agent.types import AgentRequest


def _request(text: str) -> AgentRequest:
    return AgentRequest(
        messages=[{"role": "user", "content": text}],
        user_id="test-user",
        session_id="test-session",
    )


def test_planner_exposes_algorithm_choices_for_text_and_ml_tasks():
    planner = TaskPlanner()

    clustering = planner.create_plan(_request("对文本进行聚类"))
    assert clustering.configuration["kind"] == "text_clustering"
    assert [item["key"] for item in clustering.configuration["options"]] == [
        "kmeans", "agglomerative", "dbscan"
    ]

    regression = planner.create_plan(_request("训练回归模型"))
    assert regression.configuration["kind"] == "regression"
    assert "random_forest" in [item["key"] for item in regression.configuration["options"]]


class _Memory:
    def recall(self, *args, **kwargs):
        return ""

    def commit(self, *args, **kwargs):
        return None


class _ContextBuilder:
    def __init__(self):
        self.configuration = None

    def build(self, request, plan, memory_context="", plan_configuration=None):
        self.configuration = plan_configuration
        return [{"role": "system", "content": "configured"}, *request.messages]


class _Executor:
    @contextmanager
    def bind(self, scope):
        yield


def test_runtime_passes_confirmed_configuration_to_model_context():
    context_builder = _ContextBuilder()
    runtime = AgentRuntime(
        memory=_Memory(), context_builder=context_builder, executor=_Executor()
    )
    selection = {
        "kind": "text_clustering",
        "algorithm": "dbscan",
        "parameters": {"eps": 0.3, "min_samples": 4},
        "tool_arguments": {"algorithm": "dbscan", "eps": 0.3, "min_samples": 4},
    }
    events = []
    responses = list(runtime.stream(
        _request("对文本进行聚类"),
        lambda messages: [[{"role": "assistant", "content": "完成"}]],
        event_callback=events.append,
        plan_confirmation_callback=lambda plan: selection,
    ))

    assert responses[-1][-1]["content"] == "完成"
    assert context_builder.configuration == selection
    assert any(event["type"] == "configuration_required" for event in events)


def test_runtime_stops_when_algorithm_configuration_is_cancelled():
    runtime = AgentRuntime(memory=_Memory(), executor=_Executor())
    responses = list(runtime.stream(
        _request("做中文情感分析"),
        lambda messages: (_ for _ in ()).throw(AssertionError("不应调用模型")),
        plan_confirmation_callback=lambda plan: {"cancelled": True},
    ))

    assert responses == [{"error": "用户取消了算法配置，本次任务未执行。"}]
