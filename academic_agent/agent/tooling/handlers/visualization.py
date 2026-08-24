"""可视化工具处理器。"""

from __future__ import annotations

from typing import Any


def _viz_tools():
    from academic_agent.application.services.visualization_service import visualization_service

    return visualization_service


def wordcloud(**kwargs: Any) -> dict[str, Any]:
    return _viz_tools().wordcloud(kwargs.get("text_column"), kwargs.get("title", "词云图"))


def cluster_plot(**kwargs: Any) -> dict[str, Any]:
    return _viz_tools().cluster_plot(kwargs.get("cluster_column"), kwargs.get("title", "聚类分布"))


def sentiment_plot(**kwargs: Any) -> dict[str, Any]:
    return _viz_tools().sentiment_plot(kwargs.get("sentiment_column"), kwargs.get("title", "情感分布"))


def line_chart(**kwargs: Any) -> dict[str, Any]:
    return _viz_tools().line_chart(kwargs["x_column"], kwargs["y_column"], kwargs.get("title"))
