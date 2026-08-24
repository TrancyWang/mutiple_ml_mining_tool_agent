"""Thin Agent adapters over the project's existing machine-learning tools."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from academic_agent.infrastructure.analysis_artifacts import artifact_result, create_run_dir
from academic_agent.infrastructure.runtime_paths import app_data_root


REGRESSION_TOOLS = {
    "linear": ("academic_agent.algorithms.machine_learning.regression.linear_regression_tool", "LinearRegressionTool"),
    "ridge": ("academic_agent.algorithms.machine_learning.regression.naive_bayes_tool", "RidgeRegressionTool"),
    "random_forest": ("academic_agent.algorithms.machine_learning.regression.random_forest_tool", "RandomForestRegressionTool"),
    "gbdt": ("academic_agent.algorithms.machine_learning.regression.gbdt_tool", "GBDTRegressionTool"),
    "adaboost": ("academic_agent.algorithms.machine_learning.regression.adaboost_tool", "AdaBoostRegressionTool"),
    "xgboost": ("academic_agent.algorithms.machine_learning.regression.xgboost_tool", "XGBoostRegressionTool"),
    "lightgbm": ("academic_agent.algorithms.machine_learning.regression.lightgbm_tool", "LightGBMRegressionTool"),
    "catboost": ("academic_agent.algorithms.machine_learning.regression.catboost_tool", "CatBoostRegressionTool"),
}

CAUSAL_TOOLS = {
    "ols": ("academic_agent.algorithms.machine_learning.causing.ols_logistic_tool", "OLSLogisticTool"),
    "logistic": ("academic_agent.algorithms.machine_learning.causing.ols_logistic_tool", "OLSLogisticTool"),
    "linear_dml": ("academic_agent.algorithms.machine_learning.causing.linear_dml_tool", "LinearDMLTool"),
    "double_ml": ("academic_agent.algorithms.machine_learning.causing.linear_dml_tool", "LinearDMLTool"),
    "psm": ("academic_agent.algorithms.machine_learning.causing.psm_tool", "PSMTool"),
    "matching": ("academic_agent.algorithms.machine_learning.causing.psm_tool", "PSMTool"),
    "causal_forest": ("academic_agent.algorithms.machine_learning.causing.causal_forest_dml_tool", "CausalForestDMLTool"),
}


def _load_class(reference: tuple[str, str]):
    module_name, class_name = reference
    return getattr(importlib.import_module(module_name), class_name)


def _prepare_plot_cache() -> None:
    cache = app_data_root() / ".academic_agent" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    font_cache = app_data_root() / ".academic_agent" / "fontconfig"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))


def _new_files(directory: Path, before: set[Path]) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path not in before)


class MachineLearningTools:
    """Validate Agent parameters and invoke existing project algorithm classes."""

    def __init__(self) -> None:
        self.current_data: pd.DataFrame | None = None

    def set_data(self, dataframe: pd.DataFrame) -> None:
        self.current_data = dataframe

    def _selected_data(self, columns: List[str]) -> pd.DataFrame:
        if self.current_data is None:
            raise ValueError("请先加载数据")
        missing = [column for column in columns if column not in self.current_data.columns]
        if missing:
            raise ValueError(f"以下变量不存在: {missing}")
        data = self.current_data[columns].dropna().copy()
        if data.empty:
            raise ValueError("删除缺失值后没有数据")
        return data

    @staticmethod
    def _create_run_dir(task: str) -> Path:
        from academic_agent.tools.text_mining_tools import text_mining_tools

        return create_run_dir(
            text_mining_tools.current_file_path,
            task,
            text_mining_tools.current_source_scope,
        )

    @staticmethod
    def _runtime_log(tool: str, message: str) -> None:
        from academic_agent.agent.executor import emit_tool_log

        emit_tool_log(tool, message)

    @staticmethod
    def _temporary_excel(data: pd.DataFrame):
        temporary = tempfile.TemporaryDirectory(prefix="academic_ml_")
        path = Path(temporary.name) / "selected_data.xlsx"
        data.to_excel(path, index=False)
        return temporary, path

    def regression_analysis(
        self, target_var: str, feature_vars: List[str], model_type: str = "linear",
        test_size: float = 0.2, save_result: bool = True,
        output_path: Optional[str] = None, multiple_folds: int = 5,
    ) -> Dict[str, Any]:
        """Invoke an existing regression class and preserve its native files."""
        try:
            model_type = str(model_type).lower()
            if model_type not in REGRESSION_TOOLS:
                return {"success": False, "error": f"当前项目不包含回归模型 {model_type}；可用模型: {sorted(REGRESSION_TOOLS)}"}
            data = self._selected_data([*feature_vars, target_var])
            run_dir = self._create_run_dir(f"regression_{model_type}")
            run_dir.mkdir(parents=True, exist_ok=True)
            before = set(run_dir.iterdir())
            temporary, input_path = self._temporary_excel(data)
            try:
                _prepare_plot_cache()
                tool_class = _load_class(REGRESSION_TOOLS[model_type])
                self._runtime_log("regression", f"加载 {tool_class.__name__}；样本={len(data)}；交叉训练={multiple_folds} 次")
                tool = tool_class(str(input_path), multiple_folder=int(multiple_folds), save_dir=str(run_dir), shap_save_dir=str(run_dir))
                tool.run_multiple_training()
                self._runtime_log("regression", "模型训练完成，正在生成指标、SHAP 和诊断图")
                workbook_name = Path(output_path).name if output_path else f"{model_type}_shap_importance.xlsx"
                tool.save_shap_importance(output_file=workbook_name, save_dir=str(run_dir))
                warnings = []
                for method_name, filename in (
                    ("plot_actual_vs_predicted", f"{model_type}_actual_vs_predicted.png"),
                    ("plot_residual_analysis", f"{model_type}_residual_analysis.png"),
                    ("plot_shap_summary", f"{model_type}_shap_summary.png"),
                ):
                    try:
                        getattr(tool, method_name)(output_file=filename, save_dir=str(run_dir))
                    except Exception as exc:
                        warnings.append(f"{method_name}: {exc}")
            finally:
                temporary.cleanup()
            files = _new_files(run_dir, before)
            workbook = next((path for path in files if path.suffix.lower() == ".xlsx"), None)
            images = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            return {
                "success": True,
                "message": f"已调用项目内 {tool.__class__.__name__} 完成回归分析。",
                "model_type": model_type,
                "metrics": {"r2": float(tool.r2_mean), "rmse": float(tool.rmse_mean), "mae": float(tool.mae_mean)},
                "trainings": int(multiple_folds), "sample_size": len(data),
                "output_file": str(workbook) if workbook else None,
                "image_path": str(images[0]) if images else None,
                "artifacts": artifact_result(files), "warnings": warnings,
                "implementation": f"{tool.__class__.__module__}.{tool.__class__.__name__}",
            }
        except Exception as exc:
            return {"success": False, "error": f"回归分析失败: {type(exc).__name__}: {exc}"}

    def classification_analysis(
        self, target_var: str, feature_vars: List[str], model_type: str = "svm",
        test_size: float = 0.2, save_result: bool = True,
        output_path: Optional[str] = None, multiple_folds: int = 5,
    ) -> Dict[str, Any]:
        """Invoke the project's SVMClassificationTool."""
        try:
            if str(model_type).lower() != "svm":
                return {"success": False, "error": "当前项目分类算法仅实现 SVMClassificationTool；请使用 model_type=svm。"}
            data = self._selected_data([*feature_vars, target_var])
            run_dir = self._create_run_dir("classification_svm")
            run_dir.mkdir(parents=True, exist_ok=True)
            before = set(run_dir.iterdir())
            temporary, input_path = self._temporary_excel(data)
            try:
                _prepare_plot_cache()
                tool_class = _load_class(("academic_agent.algorithms.machine_learning.regression.svm_tool", "SVMClassificationTool"))
                self._runtime_log("classification", f"加载 SVMClassificationTool；样本={len(data)}；交叉训练={multiple_folds} 次")
                tool = tool_class(str(input_path), multiple_folder=int(multiple_folds), save_dir=str(run_dir), shap_save_dir=str(run_dir))
                tool.run_multiple_training()
                self._runtime_log("classification", "模型训练完成，正在生成分类指标和评估图")
                workbook_name = Path(output_path).name if output_path else "svm_shap_importance.xlsx"
                tool.save_shap_importance(output_file=workbook_name, save_dir=str(run_dir))
                warnings = []
                for method_name, filename in (
                    ("plot_confusion_matrix", "svm_confusion_matrix.png"),
                    ("plot_classification_report", "svm_classification_report.png"),
                    ("plot_feature_importance", "svm_feature_importance.png"),
                ):
                    try:
                        getattr(tool, method_name)(output_file=filename, save_dir=str(run_dir))
                    except Exception as exc:
                        warnings.append(f"{method_name}: {exc}")
            finally:
                temporary.cleanup()
            files = _new_files(run_dir, before)
            workbook = next((path for path in files if path.suffix.lower() == ".xlsx"), None)
            images = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            return {
                "success": True, "message": "已调用项目内 SVMClassificationTool 完成分类分析。",
                "model_type": "svm",
                "metrics": {"accuracy": float(tool.accuracy_mean), "precision": float(tool.precision_mean), "recall": float(tool.recall_mean), "f1": float(tool.f1_mean)},
                "trainings": int(multiple_folds), "sample_size": len(data),
                "output_file": str(workbook) if workbook else None,
                "image_path": str(images[0]) if images else None,
                "artifacts": artifact_result(files), "warnings": warnings,
                "implementation": f"{tool.__class__.__module__}.{tool.__class__.__name__}",
            }
        except Exception as exc:
            return {"success": False, "error": f"分类分析失败: {type(exc).__name__}: {exc}"}

    def causal_inference(
        self, treatment_var: str, outcome_var: str, control_vars: List[str],
        method: str = "ols", save_result: bool = True, output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke an existing causal-analysis class and collect its native outputs."""
        try:
            method = "ols" if str(method).lower() == "linear_regression" else str(method).lower()
            if method not in CAUSAL_TOOLS:
                return {"success": False, "error": f"当前项目不包含因果方法 {method}；可用方法: {sorted(CAUSAL_TOOLS)}"}
            data = self._selected_data([treatment_var, outcome_var, *control_vars])
            run_dir = self._create_run_dir(f"causal_{method}")
            run_dir.mkdir(parents=True, exist_ok=True)
            before = set(run_dir.iterdir())
            temporary, input_path = self._temporary_excel(data)
            try:
                _prepare_plot_cache()
                tool_class = _load_class(CAUSAL_TOOLS[method])
                self._runtime_log("causal_inference", f"加载 {tool_class.__name__}；样本={len(data)}；方法={method}")
                kwargs = {"save_dir": str(run_dir)}
                if method in {"ols", "logistic"}:
                    kwargs["model_type"] = method
                tool = tool_class(str(input_path), treatment_var, outcome_var, control_vars, **kwargs)
                tool.run_full_analysis(save_dir=str(run_dir))
                self._runtime_log("causal_inference", "因果分析完成，正在收集估计结果和图表")
            finally:
                temporary.cleanup()
            files = _new_files(run_dir, before)
            workbook = next((path for path in files if path.suffix.lower() == ".xlsx"), None)
            images = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            effect = getattr(tool, "ate", getattr(tool, "att", None))
            if effect is None and getattr(tool, "coefficients", None) is not None:
                effect = float(tool.coefficients[1])
            return {
                "success": True, "message": f"已调用项目内 {tool.__class__.__name__} 完成因果分析。",
                "method": method, "treatment_effect": float(effect) if effect is not None else None,
                "sample_size": len(data), "output_file": str(workbook) if workbook else None,
                "image_path": str(images[0]) if images else None,
                "artifacts": artifact_result(files),
                "implementation": f"{tool.__class__.__module__}.{tool.__class__.__name__}",
            }
        except Exception as exc:
            return {"success": False, "error": f"因果推断失败: {type(exc).__name__}: {exc}"}


ml_tools = MachineLearningTools()
