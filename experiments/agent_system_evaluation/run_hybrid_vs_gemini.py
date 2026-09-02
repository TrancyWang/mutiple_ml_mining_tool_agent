"""Effectiveness benchmark: Gemini-only versus local models/ML code.

The Gemini-only condition receives the same labeled/training data but cannot
call tools, Python, BERT, or sklearn. The hybrid condition uses local
transformer/sklearn implementations on the identical fixed test split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC

from academic_agent.agent.providers.config import get_llm_config_priority
from academic_agent.controllers.auth import AuthController
from academic_agent.integrations.video_text_adapter import source_video_text_adapter


SENTIMENT_PATH = REPO_ROOT.parent / "text_mining_tools_agent_client_cpu" / "test_sentiment_data.xlsx"

def prediction_tool(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "emit_predictions",
            "description": "仅结构化回传模型预测，不执行任何计算、代码或外部工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"row": {"type": "integer"}, "value": value_schema},
                            "required": ["row", "value"],
                        },
                    },
                },
                "required": ["predictions"],
            },
        },
    }


def json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Gemini response is not JSON: {cleaned[:300]}")
    return json.loads(cleaned[start:end + 1])


def normalize_sentiment(value: Any) -> str:
    text = str(value).strip().lower()
    if any(token in text for token in ("positive", "正面", "积极", "好")):
        return "Positive"
    if any(token in text for token in ("negative", "负面", "消极", "差", "坏")):
        return "Negative"
    if any(token in text for token in ("neutral", "中性", "一般")):
        return "Neutral"
    raise ValueError(f"unknown sentiment label: {value}")


def gemini_json(client: OpenAI, config: dict[str, Any], system: str, user: str, value_schema: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=2400,
        tools=[prediction_tool(value_schema)],
        tool_choice={"type": "function", "function": {"name": "emit_predictions"}},
    )
    message = response.choices[0].message
    if message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments or "{}")
    return json_from_text(message.content or "")


def sentiment_benchmark(client: OpenAI, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not SENTIMENT_PATH.is_file():
        raise FileNotFoundError(f"sentiment dataset not found: {SENTIMENT_PATH}")
    frame = pd.read_excel(SENTIMENT_PATH).dropna(subset=["text", "sentiment"]).reset_index(drop=True)
    allowed = {"Positive", "Negative", "Neutral"}
    frame["sentiment"] = frame["sentiment"].map(normalize_sentiment)
    frame = frame[frame["sentiment"].isin(allowed)].copy()
    rows = [{"id": int(index), "text": str(row.text)} for index, row in frame.iterrows()]
    prompt_data = json.dumps(rows, ensure_ascii=False)
    system = "你是一个纯文本分类器。禁止调用工具、代码、本地模型或外部知识。只根据输入文本分类。"
    user = (
        "请将每条文本分类为 Positive、Negative 或 Neutral。"
        "请通过 emit_predictions 回传，value 填写 Positive、Negative 或 Neutral。\n"
        f"数据：{prompt_data}"
    )
    results: list[dict[str, Any]] = []
    for method in ("gemini_only", "hybrid_bert"):
        started = time.perf_counter()
        error = ""
        try:
            if method == "gemini_only":
                payload = gemini_json(client, config, system, user, {"type": "string"})
                predictions = {int(item.get("row", item.get("id"))): normalize_sentiment(item.get("value", item.get("label"))) for item in payload["predictions"]}
            else:
                raw = source_video_text_adapter.general_sentiment([row["text"] for row in rows])
                mapping = {"Very Negative": "Negative", "Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive", "Very Positive": "Positive"}
                predictions = {row["id"]: mapping[str(item["sentiment"])] for row, item in zip(rows, raw)}
            y_true = [frame.iloc[item_id]["sentiment"] for item_id in range(len(frame))]
            y_pred = [predictions[item_id] for item_id in range(len(frame))]
            results.append({"task": "sentiment", "method": method, "accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro"), "elapsed_seconds": time.perf_counter() - started, "n": len(y_true), "error": error})
        except Exception as exc:
            results.append({"task": "sentiment", "method": method, "accuracy": None, "macro_f1": None, "elapsed_seconds": time.perf_counter() - started, "n": len(frame), "error": f"{type(exc).__name__}: {exc}"})
    return results


def make_ml_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260825)
    features = pd.DataFrame(rng.normal(size=(70, 3)), columns=["f1", "f2", "f3"])
    classification_score = 1.3 * features.f1 - 1.1 * features.f2 + 0.7 * features.f3 + rng.normal(0, 0.25, len(features))
    classification = features.copy()
    classification["label"] = (classification_score > 0).astype(int)
    regression = features.copy()
    regression["target"] = 2.0 * features.f1 - 0.8 * features.f2 + 0.4 * features.f3 + rng.normal(0, 0.2, len(features))
    return classification.iloc[:60].reset_index(drop=True), classification.iloc[60:].reset_index(drop=True), regression.iloc[:60].reset_index(drop=True), regression.iloc[60:].reset_index(drop=True)


def ml_prompt(train: pd.DataFrame, test: pd.DataFrame, target: str, task_name: str) -> tuple[str, str]:
    test_features = test.drop(columns=[target]).to_csv(index=False)
    train_data = train.to_csv(index=False)
    system = "你是一个纯数据预测器。禁止调用 Python、sklearn、工具或外部模型。只能根据给定训练表学习规律并预测测试表。只返回 JSON。"
    user = (
        f"这是一个{task_name}任务。训练数据包含真实目标列 {target}，测试数据不包含目标列。"
        f"请预测测试集的 {target}，保持测试行顺序。只返回 JSON："
        f"{{\"predictions\":[{{\"row\":0,\"value\":0}}]}}。\n"
        f"训练数据：\n{train_data}\n测试特征：\n{test_features}"
    )
    return system, user


def ml_benchmark(client: OpenAI, config: dict[str, Any], task_name: str, train: pd.DataFrame, test: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    system, user = ml_prompt(train, test, target, task_name)
    for method in ("gemini_only", "hybrid_sklearn"):
        started = time.perf_counter()
        try:
            x_train = train[["f1", "f2", "f3"]]
            x_test = test[["f1", "f2", "f3"]]
            if target == "label":
                if method == "hybrid_sklearn":
                    model = SVC(kernel="rbf", random_state=42)
                    model.fit(x_train, train[target])
                    pred = model.predict(x_test)
                else:
                    payload = gemini_json(client, config, system, user, {"type": "integer"})
                    pred = np.array([int(round(float(item["value"]))) for item in payload["predictions"]])
                results.append({"task": "classification", "method": method, "accuracy": accuracy_score(test[target], pred), "macro_f1": f1_score(test[target], pred, average="macro"), "elapsed_seconds": time.perf_counter() - started, "n": len(test), "error": ""})
            else:
                if method == "hybrid_sklearn":
                    model = LinearRegression().fit(x_train, train[target])
                    pred = model.predict(x_test)
                else:
                    # Gemini's OpenAI-compatible endpoint is more reliable
                    # when numerical predictions are returned as strings;
                    # conversion to float happens below.
                    payload = gemini_json(client, config, system, user, {"type": "string"})
                    pred = np.array([float(item["value"]) for item in payload["predictions"]])
                results.append({"task": "regression", "method": method, "r2": r2_score(test[target], pred), "rmse": mean_squared_error(test[target], pred) ** 0.5, "mae": mean_absolute_error(test[target], pred), "elapsed_seconds": time.perf_counter() - started, "n": len(test), "error": ""})
        except Exception as exc:
            results.append({"task": "classification" if target == "label" else "regression", "method": method, "accuracy": None if target == "label" else None, "macro_f1": None if target == "label" else None, "r2": None if target != "label" else None, "rmse": None if target != "label" else None, "mae": None if target != "label" else None, "elapsed_seconds": time.perf_counter() - started, "n": len(test), "error": f"{type(exc).__name__}: {exc}"})
    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in sorted({row["task"] for row in rows}):
        summary[task] = {}
        for method in ("gemini_only", "hybrid_bert", "hybrid_sklearn"):
            items = [row for row in rows if row["task"] == task and row["method"] == method]
            if not items:
                continue
            summary[task][method] = {
                key: float(np.mean([item[key] for item in items if item.get(key) is not None]))
                for key in ("accuracy", "macro_f1", "r2", "rmse", "mae", "elapsed_seconds")
                if any(item.get(key) is not None for item in items)
            }
            summary[task][method]["valid_runs"] = sum(not item.get("error") for item in items)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Gemini-only with local BERT/sklearn analysis")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results_hybrid_vs_gemini"))
    args = parser.parse_args()
    AuthController().load_root_env()
    config = get_llm_config_priority(model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"), provider="gemini")
    client = OpenAI(api_key=config["api_key"], base_url=config["model_server"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    classification_train, classification_test, regression_train, regression_test = make_ml_data()
    for repeat in range(max(1, args.repeats)):
        for row in sentiment_benchmark(client, config):
            row["repeat"] = repeat + 1
            rows.append(row)
        for row in ml_benchmark(client, config, "分类", classification_train, classification_test, "label"):
            row["repeat"] = repeat + 1
            rows.append(row)
        for row in ml_benchmark(client, config, "回归", regression_train, regression_test, "target"):
            row["repeat"] = repeat + 1
            rows.append(row)
        print(f"completed repeat {repeat + 1}/{max(1, args.repeats)}")
    payload = {"provider": "gemini", "model": config["model"], "repeats": max(1, args.repeats), "summary": summarize(rows), "results": rows}
    (output_dir / "hybrid_vs_gemini_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "output": str(output_dir / "hybrid_vs_gemini_results.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
