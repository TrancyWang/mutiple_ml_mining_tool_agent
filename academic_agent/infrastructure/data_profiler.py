"""Lightweight file profiling used immediately after upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pandas as pd


def _load_table(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding=encoding)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding=encoding))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.json_normalize(payload)
    if suffix in {".txt", ".md"}:
        return pd.DataFrame({"text": path.read_text(encoding=encoding, errors="ignore").splitlines()})
    raise ValueError(f"暂不支持表格分析文件格式：{suffix}")


def profile_file(file_path: str, encoding: str = "utf-8") -> dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        return {"success": False, "error": f"文件不存在：{file_path}"}
    try:
        df = _load_table(path, encoding)
        columns = []
        for name in df.columns:
            series = df[name]
            sample = series.dropna().astype(str)
            avg_len = float(sample.str.len().mean()) if not sample.empty else 0.0
            lower_name = str(name).strip().lower()
            text_name_hint = lower_name in {
                "text", "content", "comment", "review", "remark", "remarks",
                "description", "semantic_unit", "semantic_units", "文本", "内容", "评论", "备注",
            }
            columns.append({
                "name": str(name),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "avg_text_length": round(avg_len, 2),
                "role": "text" if series.dtype == "object" and (avg_len >= 8 or text_name_hint) else "feature",
            })
        text_candidates = [c["name"] for c in columns if c["role"] == "text"]
        return {
            "success": True,
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "text_candidates": text_candidates,
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_cells": int(df.isna().sum().sum()),
            "preview": df.head(5).fillna("").to_dict("records"),
            "column_profile": columns,
        }
    except Exception as exc:
        return {"success": False, "error": f"文件分析失败：{exc}"}
