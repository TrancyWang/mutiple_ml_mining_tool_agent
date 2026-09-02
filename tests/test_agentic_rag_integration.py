from pathlib import Path

from academic_agent.rag.application.service import DocumentRAGService
from academic_agent.agent.planning.planner import TaskPlanner
from academic_agent.agent.tooling.builtins import tool_registry
from academic_agent.agent.types import AgentRequest, AgentRoute


def test_document_rag_indexes_structure_and_citations(tmp_path: Path):
    manual = tmp_path / "manual.md"
    manual.write_text(
        "# 服务配置\n\n端口通过环境变量 PORT 配置。\n\n"
        "| 参数 | 默认值 |\n| --- | --- |\n| PORT | 8000 |\n",
        encoding="utf-8",
    )
    service = DocumentRAGService(tmp_path / "rag-data")
    result = service.answer(str(tmp_path), "端口如何配置", [manual])

    assert result["success"] is True
    assert result["hits"]
    assert result["hits"][0]["citation"] == "【1】"
    assert "PORT" in result["answer"]


def test_document_rag_is_registered_and_planned():
    request = AgentRequest(
        messages=[{"role": "user", "content": "请根据 PDF 手册回答配置步骤"}],
        user_id="test-user",
        session_id="test-session",
        workspace_path=".",
    )
    plan = TaskPlanner().create_plan(request)
    assert plan.route is AgentRoute.DOCUMENT
    assert "document_rag" in plan.available_tools
    assert tool_registry.get("document_rag") is not None
