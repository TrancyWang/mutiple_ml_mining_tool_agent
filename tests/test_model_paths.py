from __future__ import annotations

from academic_agent.infrastructure.model_paths import model_components, normalize_model_root


def test_model_components_distinguish_model_library_and_model_types(tmp_path):
    (tmp_path / "bge-cn").mkdir()
    (tmp_path / "multilingual-sentiment-analysis").mkdir()

    status = model_components(tmp_path)

    assert status["root"] == tmp_path.resolve()
    assert status["has_bge"] is True
    assert status["has_general_sentiment"] is True
    assert status["has_chinese_sentiment"] is False


def test_model_root_normalizes_bge_child_to_parent(tmp_path):
    bge_path = tmp_path / "bge-cn"
    bge_path.mkdir()

    assert normalize_model_root(bge_path) == tmp_path.resolve()
