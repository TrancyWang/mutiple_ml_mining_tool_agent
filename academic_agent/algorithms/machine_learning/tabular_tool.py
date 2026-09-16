"""通用表格机器学习工具。

这个模块补充了分类算法，并为新增的回归算法提供统一的数据预处理：
数值列填充、类别列填充与独热编码都放在 Pipeline 内，避免把测试集信息
泄漏到训练集。工具接口与项目现有算法类保持一致，便于 Agent 统一调用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def _one_hot_encoder() -> OneHotEncoder:
    """兼容 scikit-learn 1.2 之前和之后的参数名称。"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - 旧版 sklearn 兼容分支
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor(frame: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _one_hot_encoder()),
    ])

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("没有可用于建模的特征列")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _safe_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool = False,
):
    stratify_values = None
    if stratify:
        counts = y.value_counts(dropna=False)
        # 只有每个类别至少有两个样本、且测试集足够容纳类别时才分层抽样。
        estimated_test_count = int(np.ceil(len(y) * test_size))
        if len(counts) > 1 and counts.min() >= 2 and estimated_test_count >= len(counts):
            stratify_values = y
    try:
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=stratify_values
        )
    except ValueError:
        # 小样本数据可能无法满足分层切分，退回普通随机切分并让模型给出结果。
        return train_test_split(x, y, test_size=test_size, random_state=random_state)


class _TabularBase:
    main_color = "#4C72B0"
    line_color = "#8A8A8A"

    def __init__(
        self,
        file_path: str,
        multiple_folder: int = 5,
        test_size: float = 0.2,
        save_dir: str | None = None,
        shap_save_dir: str | None = None,
    ) -> None:
        self.file_path = file_path
        self.multiple_folder = max(1, int(multiple_folder))
        self.test_size = min(max(float(test_size), 0.05), 0.8)
        self.save_dir = save_dir or str(Path(file_path).parent)
        self.shap_save_dir = shap_save_dir or self.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.shap_save_dir, exist_ok=True)
        self.df = pd.read_excel(file_path)
        if self.df.shape[1] < 2:
            raise ValueError("至少需要一列特征和一列目标变量")
        self.X = self.df.iloc[:, :-1].copy()
        self.y = self.df.iloc[:, -1].copy()
        self.feature_names = [str(column) for column in self.X.columns]
        self.model = None
        self.pipeline = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.importance_mean = None

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "figure.dpi": 150,
        })
        sns.set_style("white")

    @staticmethod
    def _permutation_importance(
        pipeline: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str,
        random_state: int,
    ) -> np.ndarray:
        try:
            result = permutation_importance(
                pipeline,
                x_test,
                y_test,
                scoring=scoring,
                n_repeats=3,
                random_state=random_state,
                n_jobs=-1,
            )
            return np.asarray(result.importances_mean, dtype=float)
        except Exception:
            # 评估集过小或只有单一类别时，仍保留可用的零重要性表，而不是让
            # 模型主结果因为解释性分析失败而整体失败。
            return np.zeros(len(x_test.columns), dtype=float)

    def _save_importance_workbook(self, output_file: str, metrics: dict[str, float]) -> pd.DataFrame:
        path = Path(self.shap_save_dir) / output_file
        importance = pd.DataFrame({
            "Feature": self.feature_names,
            "PermutationImportance": np.asarray(self.importance_mean, dtype=float),
        }).sort_values("PermutationImportance", ascending=False)
        metrics_data = pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())})
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
            metrics_data.to_excel(writer, sheet_name="Model_Metrics", index=False)
        return importance

    def plot_feature_importance(self, output_file: str | None = None, max_display: int = 15, save_dir: str | None = None):
        if self.importance_mean is None:
            return None
        path = Path(save_dir or self.save_dir) / (output_file or f"{self.model_type}_feature_importance.png")
        importance = pd.DataFrame({"Feature": self.feature_names, "Importance": self.importance_mean})
        importance = importance.sort_values("Importance", ascending=True).tail(max_display)
        plt.figure(figsize=(9, max(4, len(importance) * 0.35)))
        plt.barh(importance["Feature"], importance["Importance"], color=self.main_color)
        plt.xlabel("Permutation importance")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)


class TabularClassificationTool(_TabularBase):
    """支持数值/类别特征的多算法分类工具。"""

    SUPPORTED_MODELS = {
        "svm": SVC,
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "knn": KNeighborsClassifier,
        "naive_bayes": GaussianNB,
    }

    def __init__(self, *args, model_type: str = "svm", **kwargs) -> None:
        self.model_type = str(model_type).lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的分类模型: {self.model_type}")
        super().__init__(*args, **kwargs)
        self.class_names = self.y.dropna().drop_duplicates().tolist()
        self.accuracy_list: list[float] = []
        self.precision_list: list[float] = []
        self.recall_list: list[float] = []
        self.f1_list: list[float] = []

    def _build_model(self, random_state: int):
        options: dict[str, Any] = {"random_state": random_state}
        if self.model_type in {"svm", "logistic", "random_forest", "extra_trees"}:
            options["class_weight"] = "balanced"
        if self.model_type == "svm":
            options.update({"probability": True, "C": 1.0})
        elif self.model_type == "logistic":
            options.update({"max_iter": 2000})
        elif self.model_type == "random_forest":
            options.update({"n_estimators": 300, "n_jobs": -1})
        elif self.model_type == "extra_trees":
            options.update({"n_estimators": 300, "n_jobs": -1})
        elif self.model_type == "gradient_boosting":
            options = {"random_state": random_state, "n_estimators": 200, "learning_rate": 0.05}
        elif self.model_type == "knn":
            options = {"n_neighbors": min(5, max(1, len(self.X) // 10))}
        elif self.model_type == "naive_bayes":
            options = {}
        return self.SUPPORTED_MODELS[self.model_type](**options)

    def run_multiple_training(self) -> None:
        for index in range(self.multiple_folder):
            x_train, x_test, y_train, y_test = _safe_train_test_split(
                self.X, self.y, self.test_size, 42 + index, stratify=True
            )
            pipeline = Pipeline([
                ("preprocessor", _preprocessor(self.X, scale_numeric=True)),
                ("model", self._build_model(42 + index)),
            ])
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            self.accuracy_list.append(float(accuracy_score(y_test, y_pred)))
            self.precision_list.append(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)))
            self.recall_list.append(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)))
            self.f1_list.append(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)))
            self.importance_mean = self._permutation_importance(
                pipeline, x_test, y_test, "f1_weighted", 42 + index
            )
            if index == self.multiple_folder - 1:
                self.pipeline = pipeline
                self.model = pipeline.named_steps["model"]
                self.X_test, self.y_test, self.y_pred = x_test, y_test, y_pred

        self.accuracy_mean = float(np.mean(self.accuracy_list))
        self.precision_mean = float(np.mean(self.precision_list))
        self.recall_mean = float(np.mean(self.recall_list))
        self.f1_mean = float(np.mean(self.f1_list))

    def save_shap_importance(self, output_file: str | None = None, save_dir: str | None = None):
        if save_dir:
            self.shap_save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        return self._save_importance_workbook(
            output_file or f"{self.model_type}_feature_importance.xlsx",
            {
                "Accuracy": self.accuracy_mean,
                "Precision": self.precision_mean,
                "Recall": self.recall_mean,
                "F1-Score": self.f1_mean,
            },
        )

    def plot_confusion_matrix(self, output_file: str | None = None, save_dir: str | None = None):
        path = Path(save_dir or self.save_dir) / (output_file or f"{self.model_type}_confusion_matrix.png")
        labels = list(dict.fromkeys([*self.class_names, *pd.Series(self.y_pred).dropna().tolist()]))
        matrix = confusion_matrix(self.y_test, self.y_pred, labels=labels)
        plt.figure(figsize=(max(5, len(labels) * 0.8), 4.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_classification_report(self, output_file: str | None = None, save_dir: str | None = None):
        path = Path(save_dir or self.save_dir) / (output_file or f"{self.model_type}_classification_report.png")
        report = classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)
        rows = []
        for label, values in report.items():
            if isinstance(values, dict):
                for metric in ("precision", "recall", "f1-score"):
                    rows.append({"Class": str(label), "Metric": metric, "Value": values.get(metric, 0.0)})
        frame = pd.DataFrame(rows)
        plt.figure(figsize=(10, 5))
        if not frame.empty:
            sns.barplot(data=frame, x="Class", y="Value", hue="Metric")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)


class TabularRegressionTool(_TabularBase):
    """补充 ElasticNet、Huber、ExtraTrees 和 KNN 回归。"""

    SUPPORTED_MODELS = {
        "elastic_net": ElasticNet,
        "huber": HuberRegressor,
        "extra_trees": ExtraTreesRegressor,
        "knn": KNeighborsRegressor,
    }

    def __init__(self, *args, model_type: str = "elastic_net", **kwargs) -> None:
        self.model_type = str(model_type).lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的回归模型: {self.model_type}")
        super().__init__(*args, **kwargs)
        self.y = pd.to_numeric(self.y, errors="coerce")
        valid = self.y.notna()
        self.X, self.y = self.X.loc[valid].reset_index(drop=True), self.y.loc[valid].reset_index(drop=True)
        self.r2_list: list[float] = []
        self.rmse_list: list[float] = []
        self.mae_list: list[float] = []

    def _build_model(self, random_state: int):
        if self.model_type == "elastic_net":
            return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=random_state)
        if self.model_type == "huber":
            return HuberRegressor(max_iter=2000)
        if self.model_type == "extra_trees":
            return ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        return KNeighborsRegressor(n_neighbors=min(5, max(1, len(self.X) // 10)))

    def run_multiple_training(self) -> None:
        for index in range(self.multiple_folder):
            x_train, x_test, y_train, y_test = _safe_train_test_split(
                self.X, self.y, self.test_size, 42 + index
            )
            pipeline = Pipeline([
                ("preprocessor", _preprocessor(self.X, scale_numeric=True)),
                ("model", self._build_model(42 + index)),
            ])
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            self.r2_list.append(float(r2_score(y_test, y_pred)))
            self.rmse_list.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
            self.mae_list.append(float(mean_absolute_error(y_test, y_pred)))
            self.importance_mean = self._permutation_importance(
                pipeline, x_test, y_test, "r2", 42 + index
            )
            if index == self.multiple_folder - 1:
                self.pipeline = pipeline
                self.model = pipeline.named_steps["model"]
                self.X_test, self.y_test, self.y_pred = x_test, y_test, y_pred
        self.r2_mean = float(np.mean(self.r2_list))
        self.rmse_mean = float(np.mean(self.rmse_list))
        self.mae_mean = float(np.mean(self.mae_list))

    def save_shap_importance(self, output_file: str | None = None, save_dir: str | None = None):
        if save_dir:
            self.shap_save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        return self._save_importance_workbook(
            output_file or f"{self.model_type}_feature_importance.xlsx",
            {"R2": self.r2_mean, "RMSE": self.rmse_mean, "MAE": self.mae_mean},
        )

    def plot_actual_vs_predicted(self, output_file: str | None = None, save_dir: str | None = None):
        path = Path(save_dir or self.save_dir) / (output_file or f"{self.model_type}_actual_vs_predicted.png")
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, self.y_pred, color=self.main_color, alpha=0.65)
        low, high = float(np.min(self.y_test)), float(np.max(self.y_test))
        plt.plot([low, high], [low, high], "--", color=self.line_color)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_residual_analysis(self, output_file: str | None = None, save_dir: str | None = None):
        path = Path(save_dir or self.save_dir) / (output_file or f"{self.model_type}_residual_analysis.png")
        residuals = np.asarray(self.y_test) - np.asarray(self.y_pred)
        plt.figure(figsize=(7, 5))
        plt.scatter(self.y_pred, residuals, color=self.main_color, alpha=0.65)
        plt.axhline(0, linestyle="--", color=self.line_color)
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_shap_summary(self, output_file: str | None = None, save_dir: str | None = None):
        return self.plot_feature_importance(output_file, save_dir=save_dir)

