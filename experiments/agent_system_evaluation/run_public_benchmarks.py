"""Public-dataset benchmark: Gemini-only versus local ML components.

The comparison is intentionally controlled:

* both conditions receive the same deterministic training subset and test set;
* Gemini-only may reason over the supplied examples but cannot call Python or
  a local model;
* the hybrid condition uses CatBoost for tabular data and TF-IDF + LinearSVC
  for text classification;
* hidden CLUE TNEWS test labels are never used.

This is a pilot runner. Increase --train-size, --test-size and --repeats after
the first run is stable. Gemini calls are made in small test batches so a
single malformed structured response does not invalidate the whole dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "benchmark_data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from academic_agent.agent.providers.config import get_llm_config_priority
from academic_agent.controllers.auth import AuthController


SEED = 20260825


def prediction_tool(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "emit_predictions",
            "description": "只回传测试行的预测值，不执行代码、工具或外部检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row": {"type": "integer"},
                                "value": value_schema,
                            },
                            "required": ["row", "value"],
                        },
                    }
                },
                "required": ["predictions"],
            },
        },
    }


DATASETS = {
    "bank_marketing": {
        "kind": "tabular_classification",
        "file": "bank_marketing.csv",
        "target": "y",
        "train_size": 300,
        "test_size": 50,
    },
    "wine_quality": {
        "kind": "tabular_regression",
        "file": "wine_quality.csv",
        "target": "quality",
        "train_size": 300,
        "test_size": 50,
    },
    "chnsenticorp": {
        "kind": "text_classification",
        "train_file": "chnsenticorp_train.parquet",
        "test_file": "chnsenticorp_validation.parquet",
        "text_column": "text",
        "target": "label",
        "train_size": 300,
        "test_size": 50,
    },
    "clue_tnews": {
        "kind": "text_classification",
        "train_file": "clue_tnews_train.parquet",
        "test_file": "clue_tnews_validation.parquet",
        "text_column": "sentence",
        "target": "label",
        "train_size": 500,
        "test_size": 50,
    },
}


def load_dataset(name: str, repeat: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = DATASETS[name]
    rng_seed = SEED + repeat
    if config["kind"].startswith("tabular"):
        frame = pd.read_csv(DATA_DIR / config["file"])
        frame = frame.dropna(subset=[config["target"]]).reset_index(drop=True)
        stratify = frame[config["target"]] if config["kind"] == "tabular_classification" else None
        train, test = train_test_split(frame, test_size=0.2, random_state=rng_seed, stratify=stratify)
    else:
        train = pd.read_parquet(DATA_DIR / config["train_file"])
        test = pd.read_parquet(DATA_DIR / config["test_file"])
    train_n = min(int(config["train_size"]), len(train))
    test_n = min(int(config["test_size"]), len(test))
    if config["kind"].endswith("classification") or config["kind"] == "text_classification":
        train = train.groupby(config["target"], group_keys=False).apply(
            lambda group: group.sample(
                n=max(1, round(train_n * len(group) / len(train))),
                random_state=rng_seed,
            )
        ).reset_index(drop=True)
        # Correct rounding drift while preserving a deterministic order.
        if len(train) > train_n:
            train = train.sample(n=train_n, random_state=rng_seed)
        elif len(train) < train_n:
            missing = train_n - len(train)
            pool = pd.read_parquet(DATA_DIR / config["train_file"]) if "train_file" in config else frame
            extra = pool.drop(train.index, errors="ignore").sample(n=missing, random_state=rng_seed)
            train = pd.concat([train, extra], ignore_index=True)
        test = test.sample(n=test_n, random_state=rng_seed).reset_index(drop=True)
    else:
        train = train.sample(n=train_n, random_state=rng_seed).reset_index(drop=True)
        test = test.sample(n=test_n, random_state=rng_seed).reset_index(drop=True)
    return train.reset_index(drop=True), test.reset_index(drop=True), config


def compact_records(frame: pd.DataFrame, target: str | None = None) -> list[dict[str, Any]]:
    records = []
    for row_id, row in frame.reset_index(drop=True).iterrows():
        record = {str(k): (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        if target is not None:
            record["row"] = int(row_id)
        records.append(record)
    return records


def gemini_predict(
    client: OpenAI,
    config: dict[str, Any],
    train: pd.DataFrame,
    test_batch: pd.DataFrame,
    target: str,
    kind: str,
    text_column: str | None = None,
) -> tuple[dict[int, Any], str, float]:
    started = time.perf_counter()
    if text_column:
        train_view = train[[text_column, target]].rename(columns={text_column: "text"})
        test_view = test_batch[[text_column]].rename(columns={text_column: "text"})
        task_description = "中文文本分类"
    else:
        train_view = train
        test_view = test_batch.drop(columns=[target], errors="ignore")
        task_description = "表格分类" if kind == "tabular_classification" else "表格回归"
    train_data = json.dumps(compact_records(train_view), ensure_ascii=False, separators=(",", ":"))
    test_data = json.dumps(compact_records(test_view, target=None), ensure_ascii=False, separators=(",", ":"))
    if kind in {"tabular_classification", "text_classification"}:
        if kind == "text_classification" and "clue_tnews" in target:
            value_schema = {"type": "string"}
        else:
            value_schema = {"type": "string"}
        label_instruction = "分类标签必须原样使用训练数据中的 label 值。"
    else:
        value_schema = {"type": "number"}
        label_instruction = "预测值必须是数字，保留合理的小数位。"
    system = (
        "你是一个严格的离线预测器。禁止调用代码、Python、本地模型、工具和外部知识。"
        "只能根据给定训练样本中的规律预测测试样本。"
    )
    user = (
        f"任务类型：{task_description}。训练数据含有目标列 {target}，测试数据没有目标列。"
        f"{label_instruction} 保持测试行顺序，并对每个测试行都返回一个预测。"
        "必须通过 emit_predictions 返回 JSON。测试行编号从 0 开始。\n"
        f"训练数据：{train_data}\n测试数据：{test_data}"
    )
    last_error = ""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=5000,
                tools=[prediction_tool(value_schema)],
                tool_choice={"type": "function", "function": {"name": "emit_predictions"}},
            )
            message = response.choices[0].message
            if not message.tool_calls:
                raise ValueError("Gemini did not return emit_predictions tool call")
            payload = json.loads(message.tool_calls[0].function.arguments or "{}")
            items = payload.get("predictions", [])
            predictions = {int(item["row"]): item["value"] for item in items}
            expected = set(range(len(test_batch)))
            if set(predictions) != expected:
                raise ValueError(f"prediction row mismatch: expected {len(expected)}, got {len(predictions)}")
            if kind == "tabular_regression":
                predictions = {row: float(value) for row, value in predictions.items()}
            else:
                predictions = {row: str(value).strip() for row, value in predictions.items()}
            return predictions, "", time.perf_counter() - started
        except Exception as exc:
            last_error = f"attempt_{attempt + 1}: {type(exc).__name__}: {exc}"
    return {}, last_error, time.perf_counter() - started


def fit_local(train: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, float, str]:
    started = time.perf_counter()
    kind = config["kind"]
    target = config["target"]
    if kind == "tabular_classification":
        x_train = train.drop(columns=[target]).copy()
        x_test = test.drop(columns=[target]).copy()
        categories = [column for column in x_train.columns if x_train[column].dtype == "object"]
        for column in categories:
            x_train[column] = x_train[column].fillna("<missing>").astype(str)
            x_test[column] = x_test[column].fillna("<missing>").astype(str)
        model = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=SEED,
        )
        model.fit(x_train, train[target].astype(str), cat_features=categories)
        return model.predict(x_test).reshape(-1).astype(str), time.perf_counter() - started, "catboost_classifier"
    if kind == "tabular_regression":
        x_train = train.drop(columns=[target, "wine_type"], errors="ignore")
        x_test = test.drop(columns=[target, "wine_type"], errors="ignore")
        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            random_seed=SEED,
        )
        model.fit(x_train, train[target].astype(float))
        return model.predict(x_test), time.perf_counter() - started, "catboost_regressor"
    text_column = config["text_column"]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True, max_features=100000)
    x_train = vectorizer.fit_transform(train[text_column].astype(str))
    x_test = vectorizer.transform(test[text_column].astype(str))
    model = LinearSVC(C=2.0)
    model.fit(x_train, train[target].astype(str))
    return model.predict(x_test).astype(str), time.perf_counter() - started, "tfidf_linear_svc"


def evaluate(kind: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    if kind == "tabular_regression":
        y_true_num = y_true.astype(float).to_numpy()
        y_pred_num = np.asarray(y_pred, dtype=float)
        return {
            "r2": float(r2_score(y_true_num, y_pred_num)),
            "rmse": float(mean_squared_error(y_true_num, y_pred_num) ** 0.5),
            "mae": float(mean_absolute_error(y_true_num, y_pred_num)),
        }
    true = y_true.astype(str).to_numpy()
    pred = np.asarray(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
    }


def run_one_dataset(client: OpenAI, llm_config: dict[str, Any], name: str, repeat: int, batch_size: int) -> list[dict[str, Any]]:
    train, test, config = load_dataset(name, repeat)
    target = config["target"]
    local_pred, local_elapsed, local_model = fit_local(train, test, config)
    local_metrics = evaluate(config["kind"], test[target], local_pred)
    rows = [{
        "dataset": name,
        "repeat": repeat,
        "method": "hybrid_local_ml",
        "model": local_model,
        "n_train": len(train),
        "n_test": len(test),
        "elapsed_seconds": round(local_elapsed, 4),
        "valid": True,
        "error": "",
        **local_metrics,
    }]
    all_predictions: dict[int, Any] = {}
    errors = []
    gemini_elapsed = 0.0
    for start in range(0, len(test), batch_size):
        batch = test.iloc[start:start + batch_size].reset_index(drop=True)
        predictions, error, elapsed = gemini_predict(
            client,
            llm_config,
            train,
            batch,
            target,
            config["kind"],
            config.get("text_column"),
        )
        gemini_elapsed += elapsed
        if error:
            errors.append(error)
        all_predictions.update({start + row: value for row, value in predictions.items()})
    if len(all_predictions) == len(test) and not errors:
        gemini_pred = [all_predictions[index] for index in range(len(test))]
        metrics = evaluate(config["kind"], test[target], gemini_pred)
        valid = True
        error = ""
    else:
        metrics = {"r2": None, "rmse": None, "mae": None} if config["kind"] == "tabular_regression" else {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
        valid = False
        error = "; ".join(errors) or f"missing predictions: {len(all_predictions)}/{len(test)}"
    rows.append({
        "dataset": name,
        "repeat": repeat,
        "method": "gemini_only",
        "model": llm_config["model"],
        "n_train": len(train),
        "n_test": len(test),
        "elapsed_seconds": round(gemini_elapsed, 4),
        "valid": valid,
        "error": error,
        **metrics,
    })
    print(json.dumps({"dataset": name, "repeat": repeat, "local": local_metrics, "gemini": metrics, "gemini_valid": valid, "error": error}, ensure_ascii=False))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_names = ["accuracy", "macro_f1", "balanced_accuracy", "r2", "rmse", "mae", "elapsed_seconds"]
    for dataset in sorted({row["dataset"] for row in rows}):
        output[dataset] = {}
        for method in ("hybrid_local_ml", "gemini_only"):
            items = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            output[dataset][method] = {
                metric: float(np.mean([item[metric] for item in items if item.get(metric) is not None]))
                for metric in metric_names
                if any(item.get(metric) is not None for item in items)
            }
            output[dataset][method]["valid_runs"] = sum(bool(item.get("valid")) for item in items)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run public dataset Gemini-only vs local ML benchmark")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--train-size", type=int, default=None, help="override all dataset training subset sizes")
    parser.add_argument("--test-size", type=int, default=None, help="override all dataset test sizes")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results_public_benchmarks"))
    args = parser.parse_args()
    if args.train_size is not None or args.test_size is not None:
        for item in DATASETS.values():
            if args.train_size is not None:
                item["train_size"] = args.train_size
            if args.test_size is not None:
                item["test_size"] = args.test_size
    AuthController().load_root_env()
    llm_config = get_llm_config_priority(model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"), provider="gemini")
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["model_server"])
    rows: list[dict[str, Any]] = []
    for repeat in range(1, max(1, args.repeats) + 1):
        for name in args.datasets:
            rows.extend(run_one_dataset(client, llm_config, name, repeat, max(1, args.batch_size)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": "gemini",
        "model": llm_config["model"],
        "datasets": args.datasets,
        "repeats": max(1, args.repeats),
        "batch_size": args.batch_size,
        "summary": summarize(rows),
        "results": rows,
    }
    output_path = output_dir / "public_benchmark_results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
