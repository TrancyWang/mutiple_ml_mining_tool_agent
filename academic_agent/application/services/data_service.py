"""数据加载、统计、预处理和特征处理应用服务。"""

from __future__ import annotations

from typing import Any


class DataService:
    @property
    def tool(self):
        from academic_agent.tools.text_mining_tools import text_mining_tools

        return text_mining_tools

    @property
    def data(self):
        return self.tool.current_data

    def load(self, file_path: str, encoding: str = "UTF-8", source_scope: str | None = None) -> dict[str, Any]:
        return self.tool.load_data(file_path, encoding, source_scope=source_scope)

    def info(self) -> dict[str, Any]:
        return self.tool.get_data_info()

    def statistics(self, top_n: int = 10) -> dict[str, Any]:
        from academic_agent.algorithms.preprocessing.data_analysis import data_analysis_tools

        return data_analysis_tools.statistical_analysis(top_n)

    def preprocess(self, missing_strategy: str = "auto", drop_duplicates: bool = True) -> dict[str, Any]:
        from academic_agent.algorithms.preprocessing.data_analysis import data_analysis_tools

        return data_analysis_tools.preprocess_data(missing_strategy, drop_duplicates)

    def process_features(self, feature_columns=None, scale_numeric: bool = True) -> dict[str, Any]:
        from academic_agent.algorithms.preprocessing.data_analysis import data_analysis_tools

        return data_analysis_tools.process_features(feature_columns, scale_numeric)


data_service = DataService()

