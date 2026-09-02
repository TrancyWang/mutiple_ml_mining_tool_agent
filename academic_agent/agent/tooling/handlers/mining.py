"""文本挖掘工具处理器。"""

from __future__ import annotations

from typing import Any


def preprocess(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.text_mining_service import text_mining_service

    return text_mining_service.preprocess(kwargs.get("text_column"))


def sentiment(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.text_mining_service import text_mining_service

    return text_mining_service.sentiment(
        kwargs.get("text_column"), kwargs.get("mode", "chinese")
    )


def clustering(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.text_mining_service import text_mining_service

    return text_mining_service.clustering(
        kwargs.get("text_column"), int(kwargs.get("n_clusters", 5)),
        kwargs.get("algorithm", "kmeans"), float(kwargs.get("eps", 0.5)),
        int(kwargs.get("min_samples", 5)),
    )


def repair_clustering(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.text_mining_service import text_mining_service

    return text_mining_service.repair_clustering(kwargs.get("text_column"))


def repair_sentiment(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.text_mining_service import text_mining_service

    return text_mining_service.repair_sentiment(kwargs.get("text_column"))


def keywords(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.text_mining_service import text_mining_service

    return text_mining_service.keywords(
        kwargs.get("text_column"), int(kwargs.get("top_n", 10))
    )
