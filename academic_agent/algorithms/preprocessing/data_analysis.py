"""数据统计、数据预处理和特征处理工具。

这些工具面向已经由 ``text_mining_tools.load_data`` 加载的数据集，输出
结构化工作簿，不直接覆盖原始文件。机器学习建模工具可以继续使用处理后的
结果文件作为输入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from academic_agent.infrastructure.analysis_artifacts import (
    artifact_result,
    create_run_dir,
    write_workbook,
)


class DataAnalysisTools:
    def _data(self) -> pd.DataFrame:
        from academic_agent.tools.text_mining_tools import text_mining_tools

        if text_mining_tools.current_data is None:
            raise ValueError("请先上传并加载数据文件")
        return text_mining_tools.current_data.copy()

    def _run_dir(self, task: str) -> Path:
        from academic_agent.tools.text_mining_tools import text_mining_tools

        return create_run_dir(
            text_mining_tools.current_file_path,
            task,
            text_mining_tools.current_source_scope,
        )

    @staticmethod
    def _log(message: str) -> None:
        from academic_agent.agent.executor import emit_tool_log

        emit_tool_log("data_analysis", message)

    @staticmethod
    def _column_kind(series: pd.Series, column_name: object = "") -> str:
        normalized_name = str(column_name).strip().lower()
        if normalized_name in {"text", "content", "comment", "review", "文本", "内容", "评论"}:
            return "text"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        values = series.dropna().astype(str)
        average_length = float(values.str.len().mean()) if not values.empty else 0.0
        unique_ratio = float(series.nunique(dropna=True) / max(len(series), 1))
        if average_length >= 30 or unique_ratio >= 0.8:
            return "text"
        return "categorical"

    def statistical_analysis(self, top_n: int = 10) -> dict[str, Any]:
        """生成数据质量、描述统计、类别分布和相关性报告。"""
        try:
            data = self._data()
            self._log(f"开始数据统计；样本={len(data)} 行；字段={len(data.columns)} 列")
            schema_rows = []
            category_rows = []
            for column in data.columns:
                series = data[column]
                kind = self._column_kind(series, column)
                schema_rows.append({
                    "column": str(column),
                    "dtype": str(series.dtype),
                    "feature_type": kind,
                    "rows": len(series),
                    "non_null": int(series.notna().sum()),
                    "missing": int(series.isna().sum()),
                    "missing_pct": round(float(series.isna().mean() * 100), 2),
                    "unique": int(series.nunique(dropna=True)),
                })
                if kind == "categorical":
                    for value, count in series.astype("string").fillna("<NA>").value_counts().head(top_n).items():
                        category_rows.append({
                            "column": str(column), "value": str(value), "count": int(count),
                            "percentage": round(float(count / max(len(series), 1) * 100), 2),
                        })

            numeric = data.select_dtypes(include=[np.number])
            numeric_summary = numeric.describe().T.reset_index().rename(columns={"index": "column"})
            if numeric_summary.empty:
                numeric_summary = pd.DataFrame(columns=["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
            correlations = numeric.corr(numeric_only=True).stack().reset_index()
            correlations.columns = ["feature_a", "feature_b", "correlation"]
            correlations = correlations[correlations["feature_a"] < correlations["feature_b"]]
            correlations["correlation"] = correlations["correlation"].round(4)
            quality = pd.DataFrame([
                {"metric": "rows", "value": len(data)},
                {"metric": "columns", "value": len(data.columns)},
                {"metric": "duplicate_rows", "value": int(data.duplicated().sum())},
                {"metric": "numeric_columns", "value": len(numeric.columns)},
                {"metric": "missing_cells", "value": int(data.isna().sum().sum())},
            ])
            run_dir = self._run_dir("data_statistics")
            output = run_dir / "data_statistics.xlsx"
            write_workbook(output, {
                "quality": quality,
                "schema": pd.DataFrame(schema_rows),
                "numeric_summary": numeric_summary,
                "category_distribution": pd.DataFrame(category_rows),
                "correlations": correlations,
            })
            self._log(f"数据统计完成；报告已生成：{output.name}")
            return {
                "success": True,
                "message": "数据统计分析完成，已生成质量与描述统计工作簿。",
                "rows": len(data), "columns": len(data.columns),
                "missing_cells": int(data.isna().sum().sum()),
                "duplicate_rows": int(data.duplicated().sum()),
                "numeric_columns": list(map(str, numeric.columns)),
                "text_columns": [row["column"] for row in schema_rows if row["feature_type"] == "text"],
                "output_file": str(output), "artifacts": artifact_result([output]),
                "implementation": "academic_agent.algorithms.preprocessing.data_analysis.DataAnalysisTools.statistical_analysis",
            }
        except Exception as exc:
            return {"success": False, "error": f"数据统计分析失败: {type(exc).__name__}: {exc}"}

    def preprocess_data(self, missing_strategy: str = "auto", drop_duplicates: bool = True) -> dict[str, Any]:
        """执行通用表格预处理，原始数据不覆盖，输出清洗后的文件。"""
        try:
            data = self._data()
            strategy = str(missing_strategy).lower()
            if strategy not in {"auto", "median", "mode", "drop", "none"}:
                return {"success": False, "error": "missing_strategy 仅支持 auto、median、mode、drop、none"}
            before_rows = len(data)
            duplicate_rows = int(data.duplicated().sum())
            if drop_duplicates:
                data = data.drop_duplicates().reset_index(drop=True)
            numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
            categorical_columns = [column for column in data.columns if column not in numeric_columns]
            filled = {"numeric": 0, "categorical": 0}
            if strategy in {"auto", "median", "mode"}:
                for column in numeric_columns:
                    if data[column].isna().any():
                        if strategy == "mode":
                            value = data[column].mode(dropna=True)
                            value = value.iloc[0] if not value.empty else 0
                        else:
                            value = data[column].median()
                            value = 0 if pd.isna(value) else value
                        data[column] = data[column].fillna(value)
                        filled["numeric"] += 1
                if strategy in {"auto", "mode"}:
                    for column in categorical_columns:
                        if data[column].isna().any():
                            value = data[column].mode(dropna=True)
                            data[column] = data[column].fillna(value.iloc[0] if not value.empty else "未知")
                            filled["categorical"] += 1
            if strategy == "drop":
                data = data.dropna().reset_index(drop=True)
            run_dir = self._run_dir("data_preprocessed")
            output = run_dir / "data_preprocessed.xlsx"
            write_workbook(output, {"processed_data": data})
            self._log(f"数据预处理完成；行数 {before_rows} → {len(data)}；输出={output.name}")
            return {
                "success": True, "message": "数据预处理完成，原始数据未覆盖。",
                "rows_before": before_rows, "rows_after": len(data),
                "duplicate_rows": duplicate_rows, "filled_columns": filled,
                "output_file": str(output), "artifacts": artifact_result([output]),
                "implementation": "academic_agent.algorithms.preprocessing.data_analysis.DataAnalysisTools.preprocess_data",
            }
        except Exception as exc:
            return {"success": False, "error": f"数据预处理失败: {type(exc).__name__}: {exc}"}

    def process_features(self, feature_columns: Iterable[str] | None = None, scale_numeric: bool = True) -> dict[str, Any]:
        """识别数值/类别特征并使用 sklearn 完成缺失填充、编码和标准化。"""
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            data = self._data()
            columns = list(feature_columns) if feature_columns else list(data.columns)
            missing = [column for column in columns if column not in data.columns]
            if missing:
                return {"success": False, "error": f"特征字段不存在: {missing}"}
            data = data[columns].copy()
            numeric = [column for column in columns if pd.api.types.is_numeric_dtype(data[column])]
            categorical = [column for column in columns if column not in numeric and self._column_kind(data[column], column) == "categorical"]
            ignored = [column for column in columns if column not in numeric and column not in categorical]
            if not numeric and not categorical:
                return {"success": False, "error": "没有识别到可用于建模的数值或低基数类别特征"}
            numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers = []
            if numeric:
                transformers.append(("numeric", Pipeline(numeric_steps), numeric))
            if categorical:
                try:
                    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                except TypeError:
                    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
                transformers.append(("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
                ]), categorical))
            transformer = ColumnTransformer(transformers=transformers, remainder="drop")
            matrix = transformer.fit_transform(data)
            names = list(transformer.get_feature_names_out())
            result_data = pd.DataFrame(matrix, columns=names, index=data.index)
            run_dir = self._run_dir("feature_processing")
            output = run_dir / "processed_features.xlsx"
            write_workbook(output, {"features": result_data, "feature_map": pd.DataFrame([
                {"source_column": column, "feature_type": "numeric" if column in numeric else "categorical"}
                for column in columns if column not in ignored
            ])})
            self._log(f"特征处理完成；输入={len(columns)} 个字段；输出特征={len(names)} 个")
            return {
                "success": True, "message": "特征处理完成，已完成缺失填充、类别编码和数值标准化。",
                "rows": len(result_data), "columns": len(names),
                "numeric_features": list(map(str, numeric)),
                "categorical_features": list(map(str, categorical)),
                "ignored_text_features": list(map(str, ignored)),
                "feature_names": names,
                "output_file": str(output), "artifacts": artifact_result([output]),
                "implementation": "sklearn ColumnTransformer + Pipeline",
            }
        except Exception as exc:
            return {"success": False, "error": f"特征处理失败: {type(exc).__name__}: {exc}"}


data_analysis_tools = DataAnalysisTools()
