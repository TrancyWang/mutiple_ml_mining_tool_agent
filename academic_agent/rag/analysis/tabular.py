from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class TabularAnalyzer:
    """安全的 CSV/JSON 分析能力，替代旧示例中的任意 Python 执行工具。

    只执行白名单统计操作，不执行用户生成的 Python 或 SQL，适合直接暴露给
    API 和 Agent 工具。复杂分析仍可由上层接入隔离的数据分析沙箱。
    """

    EXTENSIONS = {"csv", "json", "jsonl"}

    def __init__(self, storage: Any | None = None):
        self.storage = storage

    def is_tabular(self, path: Path) -> bool:
        return path.suffix.lower().lstrip(".") in self.EXTENSIONS

    def load_rows(self, path: Path) -> list[dict[str, Any]]:
        ext = path.suffix.lower().lstrip(".")
        if ext == "csv":
            with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]

        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return []
        if ext == "jsonl":
            value = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            value = json.loads(raw)
        if isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                return [dict(item) for item in value]
            return [{"value": item} for item in value]
        if isinstance(value, dict):
            # Common API shape: {"data": [{...}, ...]}.
            for key in ("data", "rows", "items", "records"):
                if isinstance(value.get(key), list) and all(isinstance(item, dict) for item in value[key]):
                    return [dict(item) for item in value[key]]
            return [{"key": key, "value": item} for key, item in value.items()]
        return [{"value": value}]

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return float(value)
        try:
            number = float(str(value).replace(",", ""))
            return number if math.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _display(number: float) -> int | float:
        return int(number) if number.is_integer() else round(number, 6)

    def profile(self, rows: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(str(key))
        details: list[dict[str, Any]] = []
        for column in columns:
            values = [row.get(column) for row in rows]
            non_null = [value for value in values if value not in (None, "")]
            numbers = [number for value in non_null if (number := self._number(value)) is not None]
            if numbers and len(numbers) >= max(2, int(len(non_null) * 0.8)):
                kind = "number"
            elif all(isinstance(value, bool) for value in non_null):
                kind = "boolean"
            else:
                kind = "text"
            item: dict[str, Any] = {
                "name": column,
                "type": kind,
                "non_null": len(non_null),
                "null_count": len(values) - len(non_null),
                "unique_count": len({str(value) for value in non_null}),
                "sample": [str(value) for value in non_null[:3]],
            }
            if numbers:
                item.update({
                    "min": self._display(min(numbers)),
                    "max": self._display(max(numbers)),
                    "mean": self._display(sum(numbers) / len(numbers)),
                })
            details.append(item)
        return {"row_count": len(rows), "column_count": len(columns), "columns": details, "preview": rows[:limit]}

    def aggregate(
        self,
        rows: list[dict[str, Any]],
        column: str | None = None,
        group_by: str | None = None,
        aggregation: str = "count",
        limit: int = 20,
    ) -> dict[str, Any]:
        if group_by and any(group_by not in row for row in rows):
            raise ValueError(f"unknown group_by column: {group_by}")
        if column and any(column not in row for row in rows):
            raise ValueError(f"unknown column: {column}")
        if aggregation == "unique":
            values = sorted({str(row.get(column)) for row in rows if row.get(column) not in (None, "")})
            return {"aggregation": aggregation, "column": column, "values": values[:limit], "count": len(values)}

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if group_by:
            for row in rows:
                groups[str(row.get(group_by, ""))].append(row)
        else:
            groups["all"] = rows

        output: list[dict[str, Any]] = []
        for name, group in groups.items():
            if aggregation == "count":
                value: int | float = len(group)
            else:
                if not column:
                    raise ValueError(f"column is required for aggregation: {aggregation}")
                numbers = [number for row in group if (number := self._number(row.get(column))) is not None]
                if not numbers:
                    value = None
                elif aggregation == "sum":
                    value = self._display(sum(numbers))
                elif aggregation == "mean":
                    value = self._display(sum(numbers) / len(numbers))
                elif aggregation == "min":
                    value = self._display(min(numbers))
                elif aggregation == "max":
                    value = self._display(max(numbers))
                else:
                    raise ValueError(f"unsupported aggregation: {aggregation}")
            output.append({"group": name, "value": value})
        output.sort(key=lambda item: (item["value"] is None, -(item["value"] or 0), item["group"]))
        return {"aggregation": aggregation, "column": column, "group_by": group_by, "results": output[:limit]}

    def chart(
        self,
        rows: list[dict[str, Any]],
        output_dir: Path,
        column: str | None = None,
        group_by: str | None = None,
        chart_type: str = "bar",
        limit: int = 20,
    ) -> dict[str, Any]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("install matplotlib to generate charts") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        if chart_type == "hist":
            if not column:
                raise ValueError("column is required for hist chart")
            values = [number for row in rows if (number := self._number(row.get(column))) is not None]
            if not values:
                raise ValueError(f"column has no numeric values: {column}")
            plt.hist(values, bins=min(20, max(5, int(len(values) ** 0.5))))
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.title(f"Distribution of {column}")
        else:
            if group_by:
                counts = Counter(str(row.get(group_by, "")) for row in rows)
                pairs = counts.most_common(limit)
                labels, values = zip(*pairs) if pairs else ([], [])
            elif column:
                pairs = [(str(row.get(column, "")), 1) for row in rows[:limit]]
                labels, values = zip(*pairs) if pairs else ([], [])
            else:
                raise ValueError("column or group_by is required for bar/line chart")
            if chart_type == "line":
                plt.plot(list(labels), list(values), marker="o")
            else:
                plt.bar(list(labels), list(values))
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Count")
            plt.title(f"{group_by or column}")
        figure = plt.gcf()
        figure.tight_layout()
        filename = "chart.png"
        path = output_dir / filename
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close("all")
        return {"path": str(path), "filename": filename, "chart_type": chart_type}

    def analyze_file(self, path: Path, operation: str = "profile", **kwargs: Any) -> dict[str, Any]:
        if not self.is_tabular(path):
            raise ValueError("analysis is currently supported for CSV, JSON and JSONL documents")
        rows = self.load_rows(path)
        limit = int(kwargs.get("limit", 20))
        if operation in {"profile", "preview"}:
            result = self.profile(rows, limit)
            if operation == "preview":
                return {"row_count": result["row_count"], "preview": result["preview"]}
            return result
        if operation == "aggregate":
            return self.aggregate(rows, kwargs.get("column"), kwargs.get("group_by"), kwargs.get("aggregation", "count"), limit)
        if operation == "chart":
            return self.chart(rows, path.parent / "analysis", kwargs.get("column"), kwargs.get("group_by"), kwargs.get("chart_type", "bar"), limit)
        if operation == "ask":
            return self.answer(rows, kwargs.get("query", ""), limit)
        raise ValueError(f"unsupported analysis operation: {operation}")

    def answer(self, rows: list[dict[str, Any]], query: str, limit: int = 20) -> dict[str, Any]:
        """Small deterministic router for no-LLM usage and Agent tool calls."""
        normalized = query.lower()
        profile = self.profile(rows, limit)
        if any(word in normalized for word in ("多少行", "几行", "行数", "多少条", "记录数", "row count")):
            return {"answer": f"数据共有 {len(rows)} 行。", "operation": "count", "row_count": len(rows)}
        if any(word in normalized for word in ("列名", "字段", "columns", "column")):
            return {"answer": "列包括：" + "、".join(item["name"] for item in profile["columns"]), "operation": "columns", "columns": profile["columns"]}
        for phrase, operation in (("平均", "mean"), ("均值", "mean"), ("总和", "sum"), ("求和", "sum"), ("最大", "max"), ("最小", "min")):
            if phrase in normalized:
                numeric = next((item["name"] for item in profile["columns"] if item["type"] == "number"), None)
                if numeric:
                    result = self.aggregate(rows, numeric, None, operation, limit)
                    return {"answer": json.dumps(result, ensure_ascii=False), "operation": operation, **result}
        return {"answer": f"数据共有 {len(rows)} 行、{len(profile['columns'])} 列。", "operation": "profile", **profile}

    def answer_dataset(self, dataset_id: str, query: str) -> dict[str, Any]:
        if not self.storage:
            return {"answer": "未配置数据存储。"}
        results = []
        for document in self.storage.list_documents(dataset_id):
            path = self.storage.find_document_file(dataset_id, document["id"])
            if path and self.is_tabular(path):
                try:
                    results.append({"document_id": document["id"], "document_name": document["name"], **self.analyze_file(path, "ask", query=query)})
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        if not results:
            return {"answer": "当前知识库中没有可分析的 CSV/JSON 文档。", "documents": []}
        return {"answer": "\n".join(f"{item['document_name']}：{item['answer']}" for item in results), "documents": results}
