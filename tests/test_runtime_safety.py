from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from academic_agent.integrations.video_text_adapter import SourceVideoTextAdapter


def test_clustering_embeddings_are_stably_normalized_without_overflow():
    values = np.array(
        [[1.0e300, -2.0e300], [3.0, 4.0], [0.0, 0.0]],
        dtype=np.float64,
    )

    normalized = SourceVideoTextAdapter._prepare_embeddings(values, expected_rows=3)

    assert normalized.shape == values.shape
    assert np.isfinite(normalized).all()
    assert np.allclose(np.linalg.norm(normalized[:2], axis=1), 1.0)
    assert np.allclose(normalized[2], 0.0)


def test_clustering_embeddings_reject_non_finite_values():
    values = np.array([[1.0, np.inf], [2.0, 3.0]])

    with pytest.raises(RuntimeError, match="非有限值"):
        SourceVideoTextAdapter._prepare_embeddings(values, expected_rows=2)


def test_clustering_embeddings_validate_row_count_and_rank():
    with pytest.raises(RuntimeError, match="二维矩阵"):
        SourceVideoTextAdapter._prepare_embeddings([1.0, 2.0], expected_rows=2)
    with pytest.raises(RuntimeError, match="向量条数"):
        SourceVideoTextAdapter._prepare_embeddings(np.ones((1, 2)), expected_rows=2)


def test_isolated_clustering_uses_spawn_and_returns_labels(tmp_path):
    """Exercise the process boundary without loading the real BGE model."""

    package = tmp_path / "src_codes" / "text_processor_subagent_stage_3" / "common"
    package.mkdir(parents=True)
    for init_path in (
        tmp_path / "src_codes" / "text_processor_subagent_stage_3" / "__init__.py",
        package / "__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")
    (package / "textEmbedding.py").write_text(
        """
import numpy as np

class _FakeEmbedding:
    def get_query_vector(self, texts):
        return np.asarray([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ], dtype=np.float64)

txt = _FakeEmbedding()
""".strip(),
        encoding="utf-8",
    )

    adapter = SourceVideoTextAdapter(tmp_path)
    result = adapter.clustering(
        pd.DataFrame({"content": ["a", "b", "c", "d"]}),
        text_column="content",
        n_clusters=2,
        algorithm="kmeans",
    )

    assert list(result.columns) == ["row_id", "cluster"]
    assert len(result) == 4
    assert set(result["cluster"]) == {0, 1}
