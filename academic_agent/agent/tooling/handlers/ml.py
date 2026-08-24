"""机器学习工具处理器。"""

from __future__ import annotations

from typing import Any


def _prepare_data():
    from academic_agent.application.services.machine_learning_service import machine_learning_service

    return machine_learning_service


def regression(**kwargs: Any) -> dict[str, Any]:
    return _prepare_data().regression_analysis(
        kwargs["target_var"], kwargs.get("feature_vars", []),
        kwargs.get("model_type", "linear"), float(kwargs.get("test_size", 0.2)),
        output_path=kwargs.get("output_path"),
    )


def classification(**kwargs: Any) -> dict[str, Any]:
    return _prepare_data().classification_analysis(
        kwargs["target_var"], kwargs.get("feature_vars", []),
        kwargs.get("model_type", "svm"), float(kwargs.get("test_size", 0.2)),
        output_path=kwargs.get("output_path"),
    )


def causal(**kwargs: Any) -> dict[str, Any]:
    return _prepare_data().causal_inference(
        kwargs["treatment_var"], kwargs["outcome_var"],
        kwargs.get("control_vars", []), kwargs.get("method", "ols"),
        output_path=kwargs.get("output_path"),
    )
