"""数据理解、统计和预处理工具处理器。"""

from __future__ import annotations

from typing import Any


def profile(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.infrastructure.data_profiler import profile_file

    return profile_file(kwargs["file_path"], kwargs.get("encoding", "utf-8"))


def load_data(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.data_service import data_service

    return data_service.load(
        kwargs["file_path"], kwargs.get("encoding", "UTF-8"),
        source_scope=kwargs.get("_source_scope"),
    )


def get_data_info(**_: Any) -> dict[str, Any]:
    from academic_agent.application.services.data_service import data_service

    return data_service.info()


def data_statistics(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.data_service import data_service

    return data_service.statistics(int(kwargs.get("top_n", 10)))


def data_preprocess(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.data_service import data_service

    return data_service.preprocess(
        kwargs.get("missing_strategy", "auto"),
        bool(kwargs.get("drop_duplicates", True)),
    )


def feature_processing(**kwargs: Any) -> dict[str, Any]:
    from academic_agent.application.services.data_service import data_service

    return data_service.process_features(
        kwargs.get("feature_columns"), bool(kwargs.get("scale_numeric", True))
    )
