from academic_agent.agent.orchestration.plan_loop import (
    apply_user_answer,
    next_missing_info,
    normalize_information_frame,
    normalize_plan_review,
)
from academic_agent.agent.orchestration.verifier import TaskVerifier
from academic_agent.agent.context import AgentContextBuilder
from academic_agent.agent.runtime import AgentRuntime
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.types import (
    AgentPlan,
    AgentRequest,
    AgentRoute,
    TaskBoard,
    TaskItem,
    TaskStatus,
)


class MemoryStub:
    def recall(self, *args, **kwargs):
        return ""

    def commit(self, *args, **kwargs):
        return None


def response(content):
    return [{"role": "assistant", "content": content}]


def test_plan_information_frame_is_numbered_and_limited():
    frame = normalize_information_frame(
        {
            "plan_ready": False,
            "information_frame": [
                {"topic": str(i), "why_needed": "why", "question": f"q{i}"}
                for i in range(5)
            ],
        },
        2,
    )
    assert len(frame) == 3
    assert frame[0]["info_id"] == "R2-F1"
    answer = apply_user_answer(frame, "R2-F1", "用户回答")
    assert answer["status"] == "filled"
    assert next_missing_info(frame)["info_id"] == "R2-F2"


def test_plan_review_normalizes_safe_model_variants():
    review = normalize_plan_review({
        "plan_review": {
            "plan_ready": "true",
            "information_frame": None,
        }
    })
    assert review["plan_ready"] is True
    assert review["information_frame"] == []


def test_task_board_assigns_program_ids_and_states():
    plan = AgentPlan("目标", AgentRoute.DATA, ())
    board = TaskBoard.from_task_plan(
        plan,
        {
            "board_goal": "逐项完成结果",
            "final_deliverable": "最终报告",
            "tasks": [
                {
                    "task_title": "第一项",
                    "task_goal": "确认输入",
                    "deliverable": "输入摘要",
                    "done_when": "明确输入",
                },
                {
                    "task_title": "第二项",
                    "task_goal": "完成分析",
                    "deliverable": "分析结果",
                    "done_when": "有结果",
                },
            ],
        },
    )
    assert list(board.tasks) == ["T1", "T2"]
    assert board.tasks["T1"].status is TaskStatus.READY
    assert board.tasks["T2"].depends_on == ("T1",)


def test_verifier_consumes_model_passed_and_feedback():
    task = TaskItem("T1", "写摘要", "写摘要", done_when="包含结论")
    verifier = TaskVerifier()
    passed = verifier.verify(task, {"passed": True, "feedback": "包含结论"}, "结果")
    failed = verifier.verify(task, {"passed": False, "feedback": "缺少结论"}, "结果")
    assert passed.passed is True
    assert failed.passed is False
    assert failed.feedback == "缺少结论"


def test_semantic_capability_routes_are_resolved_through_local_registry():
    planner = TaskPlanner()
    request = AgentRequest(
        [{"role": "user", "content": "帮我分析一下"}],
        "u",
        "academic-intent-test",
        chat_only=False,
    )
    refined = planner.create_plan(request, capability_routes=(AgentRoute.DATA,))
    assert refined.route is AgentRoute.DATA
    assert "text_clustering" in refined.available_tools


def test_lightweight_chat_intent_is_detected_without_matching_task_requests():
    planner = TaskPlanner()
    assert planner.recognize_intent(
        AgentRequest([{"role": "user", "content": "你是谁啊？"}], "u", "greeting")
    ) == "lightweight_chat"
    assert planner.recognize_intent(
        AgentRequest([{"role": "user", "content": "你好，帮我分析这份数据"}], "u", "task")
    ) == "task"


def test_lightweight_chat_skips_semantic_planning_and_tools():
    semantic_calls = []
    model_calls = []

    def semantic_runner(messages):
        semantic_calls.append(messages)
        raise AssertionError("简单问候不应调用语义规划模型")

    def model_runner(messages):
        model_calls.append(messages)
        return response("我是 Academic Agent，可以帮你做学术数据分析。")

    runtime = AgentRuntime(memory=MemoryStub())
    events = []
    result = runtime.run(
        AgentRequest([{"role": "user", "content": "你是谁啊？"}], "u", "greeting-runtime"),
        model_runner,
        semantic_runner=semantic_runner,
        event_callback=events.append,
    )

    assert "Academic Agent" in result[-1]["content"]
    assert len(model_calls) == 1
    assert semantic_calls == []
    assert any(event["type"] == "intent_recognized" for event in events)
    assert not any(event["type"] == "plan_review_required" for event in events)


def test_simple_document_read_uses_document_route_without_academic_clarification():
    planner = TaskPlanner()
    request = AgentRequest(
        [{"role": "user", "content": "只需要读一下这个文件"}],
        "u",
        "simple-document-read",
    )

    assert planner.is_simple_document_read(request.query) is True
    assert planner.recognize_intent(request) == "simple_document_read"
    assert planner.create_plan(request).route is AgentRoute.DOCUMENT
    assert planner.requires_plan_review(request) is False


def test_simple_document_read_skips_plan_review_callback():
    semantic_calls = []
    review_calls = []

    def semantic_runner(messages):
        semantic_calls.append(messages)
        raise AssertionError("简单文件读取不应进入语义 Plan 阶段")

    runtime = AgentRuntime(memory=MemoryStub())
    events = []
    plan, board, messages, auto_mode = runtime._prepare_workflow(
        AgentRequest([{"role": "user", "content": "只需要读一下这个文件"}], "u", "simple-read-runtime"),
        semantic_runner=semantic_runner,
        event_callback=events.append,
        clarification_callback=lambda _info: review_calls.append("clarification"),
        plan_review_callback=lambda plan, board: review_calls.append((plan, board)),
        plan_confirmation_callback=None,
    )

    assert plan.route is AgentRoute.DOCUMENT
    assert len(board.tasks) == 1
    assert board.tasks["T1"].suggested_tools == ("document_rag",)
    assert messages[-1]["content"] == "只需要读一下这个文件"
    assert auto_mode is False
    assert semantic_calls == []
    assert review_calls == []
    assert any(event["type"] == "plan_review_skipped" for event in events)


def test_uploaded_file_analysis_routes_to_data_not_project():
    planner = TaskPlanner()
    for query in (
        "请查看当前文件的数据规模、字段和缺失值",
        "请对当前文件的评论列做文本聚类",
    ):
        request = AgentRequest(
            [{"role": "user", "content": query}],
            "u",
            "uploaded-file-routing-test",
        )
        plan = planner.create_plan(request)
        assert plan.route is AgentRoute.DATA
        assert "data_statistics" in plan.available_tools
        assert "text_clustering" in plan.available_tools
        assert "list_project_files" not in plan.available_tools


def test_cluster_structured_evidence_survives_truncated_draft():
    task = TaskItem(
        "T1",
        "KMeans 主题聚类分布",
        "整理每个主题簇的分布和特征描述",
        task_goal="整理每个主题簇的分布和特征描述",
        deliverable="3 个主题簇的完整分布数据与特征描述",
        done_when="完整提供 3 个主题簇的分布数据与特征描述",
    )
    observation = {
        "tool": "text_clustering",
        "result": {
            "success": True,
            "algorithm_name": "kmeans",
            "n_clusters": 3,
            "cluster_distribution": [
                {"cluster_id": 0, "segment_count": 50, "percentage": 0.34},
                {"cluster_id": 1, "segment_count": 45, "percentage": 0.31},
                {"cluster_id": 2, "segment_count": 50, "percentage": 0.35},
            ],
            "cluster_profiles": [
                {"cluster_id": i, "representative_texts": [f"簇 {i} 的代表文本"]}
                for i in range(3)
            ],
        },
    }
    result = TaskVerifier().verify(
        task,
        {"passed": False, "feedback": "草稿表格被截断"},
        "(3) KMeans 主题聚类分布章节发生截断",
        [observation],
    )
    assert result.passed is True
    assert "结构化验收" in result.feedback


def test_runtime_runs_one_task_at_a_time_and_assembles_results():
    semantic_outputs = iter([
        response('{"intent":"task","route":"data","focus":["结果质量"],"reason":"用户关注结果"}'),
        response('{"plan_goal":"完成摘要","final_deliverable":"摘要","readiness_summary":"信息足够","information_frame":[],"plan_ready":true}'),
        response("# 执行 Plan\n逐项生成摘要。"),
        response('{"board_goal":"完成摘要","final_deliverable":"摘要","tasks":[{"task_title":"写摘要","task_goal":"写出摘要","deliverable":"摘要正文","done_when":"结果非空"},{"task_title":"补充结论","task_goal":"补充结论","deliverable":"结论正文","done_when":"结果非空"}]}'),
        response('{"passed":true,"feedback":"结果非空"}'),
        response('{"action":"continue","reason":"继续"}'),
        response('{"passed":true,"feedback":"结果非空"}'),
        response('{"action":"continue","reason":"完成"}'),
    ])
    executed = []

    def semantic_runner(messages):
        return next(semantic_outputs)

    def model_runner(messages):
        current = messages[-1]["content"]
        executed.append(current)
        return response("当前任务结果")

    runtime = AgentRuntime(memory=MemoryStub())
    result = runtime.run(
        AgentRequest([{"role": "user", "content": "请对数据做统计摘要"}], "u", "s", "."),
        model_runner,
        semantic_runner=semantic_runner,
    )
    assert "当前任务结果" in result[-1]["content"]
    assert len(executed) == 2
    assert runtime.last_task_board is not None
    assert all(task.status is TaskStatus.COMPLETED for task in runtime.last_task_board.tasks.values())


def test_context_collapses_system_messages_to_one_leading_message():
    request = AgentRequest(
        [
            {"role": "system", "content": "旧系统约束"},
            {"role": "user", "content": "做数据摘要"},
            {"role": "system", "content": "新增系统约束"},
        ],
        "u",
        "system-message-test",
    )
    plan = AgentPlan("做数据摘要", AgentRoute.DATA, ())
    messages = AgentContextBuilder().build(request, plan)
    assert sum(message.get("role") == "system" for message in messages) == 1
    assert messages[0]["role"] == "system"
    assert "旧系统约束" in messages[0]["content"]
    assert "新增系统约束" in messages[0]["content"]


def test_plan_review_repairs_invalid_structure_without_exposing_internal_question():
    semantic_outputs = iter([
        response('{"plan_ready":true,"information_frame":[{"topic":"多余缺口","why_needed":"不应存在","question":"不应被用户看到"}]}'),
        response('{"plan_goal":"完成摘要","final_deliverable":"摘要","readiness_summary":"信息足够","information_frame":[],"plan_ready":true}'),
    ])
    events = []

    def semantic_runner(messages):
        return next(semantic_outputs)

    runtime = AgentRuntime(memory=MemoryStub())
    plan = AgentPlan("完成数据摘要", AgentRoute.DATA, ())
    review = runtime._plan_loop(
        AgentRequest([{"role": "user", "content": "请完成数据摘要"}], "u", "plan-review-repair", "."),
        plan,
        [],
        semantic_runner,
        None,
        [],
        events.append,
    )
    assert review["plan_ready"] is True
    assert not any(event["type"] == "clarification_required" for event in events)


def test_plan_review_rejects_internal_output_format_as_information_gap():
    semantic_outputs = iter([
        response('{"plan_ready":false,"information_frame":[{"topic":"Plan Review 输出结构","why_needed":"系统需要确认格式","question":"请确认 JSON 输出结构","options":[{"label":"确认","value":"确认","description":"确认格式"}]}]}'),
        response('{"plan_goal":"完成摘要","final_deliverable":"摘要","readiness_summary":"信息足够","information_frame":[],"plan_ready":true}'),
    ])

    def semantic_runner(messages):
        return next(semantic_outputs)

    runtime = AgentRuntime(memory=MemoryStub())
    plan = AgentPlan("完成数据摘要", AgentRoute.DATA, ())
    review = runtime._plan_loop(
        AgentRequest([{"role": "user", "content": "请完成数据摘要"}], "u", "internal-gap-test", "."),
        plan,
        [],
        semantic_runner,
        lambda _item: (_ for _ in ()).throw(AssertionError("不应向用户展示内部格式问题")),
        [],
        None,
    )
    assert review["plan_ready"] is True


def test_plan_review_uses_local_fallback_after_two_invalid_outputs():
    semantic_outputs = iter([
        response("不是 JSON"),
        response('{"plan_ready":"unknown","information_frame":[]}'),
    ])

    def semantic_runner(messages):
        return next(semantic_outputs)

    runtime = AgentRuntime(memory=MemoryStub())
    plan = AgentPlan("完成数据摘要", AgentRoute.DATA, ())
    events = []
    review = runtime._plan_loop(
        AgentRequest([{"role": "user", "content": "请完成数据摘要"}], "u", "fallback-test", "."),
        plan,
        [],
        semantic_runner,
        lambda _item: (_ for _ in ()).throw(AssertionError("兜底计划不应伪造用户问题")),
        [],
        events.append,
    )
    assert review["fallback_used"] is True
    assert review["plan_ready"] is True
    assert any(event["type"] == "plan_review_fallback" for event in events)


def test_runtime_replans_once_after_task_is_blocked():
    semantic_outputs = iter([
        response('{"intent":"task","route":"data","focus":["结果质量"],"reason":"用户关注结果"}'),
        response('{"plan_goal":"完成摘要","final_deliverable":"摘要","readiness_summary":"信息足够","information_frame":[],"plan_ready":true}'),
        response("# 初始 Plan\n先完成摘要。"),
        response('{"board_goal":"完成摘要","final_deliverable":"摘要","tasks":[{"task_title":"写摘要","task_goal":"写出摘要","deliverable":"摘要正文","done_when":"结果合格"}]}'),
        response('{"passed":false,"feedback":"结果缺少关键字段"}'),
        response('{"action":"continue","reason":"先重试当前任务"}'),
        response('{"passed":false,"feedback":"重试后仍缺少关键字段"}'),
        response('{"action":"continue","reason":"需要重新规划"}'),
        response('{"plan_goal":"改用补充资料完成摘要","final_deliverable":"修订摘要","readiness_summary":"可按新路径执行","information_frame":[],"plan_ready":true}'),
        response("# 修订 Plan\n补充资料后重新生成摘要。"),
        response('{"board_goal":"完成修订摘要","final_deliverable":"修订摘要","tasks":[{"task_title":"补充并重写摘要","task_goal":"补齐关键字段","deliverable":"修订摘要正文","done_when":"包含关键字段"}]}'),
        response('{"passed":true,"feedback":"已包含关键字段"}'),
        response('{"action":"continue","reason":"完成"}'),
    ])
    events = []
    executed = []

    def semantic_runner(messages):
        return next(semantic_outputs)

    def model_runner(messages):
        executed.append(messages[-1]["content"])
        return response("初始失败结果" if len(executed) < 3 else "修订后的结果")

    runtime = AgentRuntime(memory=MemoryStub())
    result = runtime.run(
        AgentRequest([{"role": "user", "content": "请完成数据摘要"}], "u", "replan-test", "."),
        model_runner,
        semantic_runner=semantic_runner,
        event_callback=events.append,
        plan_review_callback=lambda plan, board: {"approved": True},
    )
    assert "修订后的结果" in result[-1]["content"]
    assert len(executed) == 3
    assert any(event["type"] == "replan_started" for event in events)
    assert any(event["type"] == "replan_completed" for event in events)
    assert any(event["type"] == "convergence_completed" for event in events)
