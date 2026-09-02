"""任务结果的语义检查和程序侧安全兜底。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from academic_agent.agent.response import assistant_content
from academic_agent.agent.types import TaskItem, VerificationResult


class TaskVerifier:
    """消费模型返回的 passed/feedback，不让模型直接修改任务状态。"""

    _ARTIFACT_KEYS = (
        "output_file", "image_path", "image_url", "visualization",
        "output_files", "artifacts",
    )

    def verify(
        self,
        task: TaskItem,
        review: dict[str, Any] | None,
        draft: Any,
        observations: list[dict[str, Any]] | None = None,
    ) -> VerificationResult:
        observations = observations or []
        evidence = [
            {
                "tool": item.get("tool"),
                "success": bool((item.get("result") or {}).get("success")),
                "summary": self._summary(item.get("result") or {}),
            }
            for item in observations
        ]
        artifacts = self._artifacts(observations)
        text = self._draft_text(draft)

        if not text and not observations:
            return VerificationResult(
                passed=False,
                feedback="当前任务没有生成结果，也没有观察到工具结果。",
                evidence=evidence,
                artifacts=artifacts,
            )
        repair_failures = [
            item for item in observations
            if str(item.get("tool", "")).startswith("repair_")
            and not bool((item.get("result") or {}).get("success"))
        ]
        if repair_failures:
            result = repair_failures[-1].get("result") or {}
            return VerificationResult(
                passed=False,
                feedback=str(result.get("error") or "规则修复没有返回可用结果。"),
                evidence=evidence,
                artifacts=artifacts,
            )
        # 对聚类这类已经产生结构化统计结果的任务，优先使用工具证据验收。
        # 主模型的自然语言草稿可能在长表格中途停止，但这不等于算法结果
        # 不完整；只要每个簇都有分布和代表文本，就不应被错误打回重做。
        structured = self._structured_completion(task, observations)
        if structured is not None:
            passed, feedback = structured
            return VerificationResult(
                passed=passed,
                feedback=feedback,
                evidence=evidence,
                artifacts=artifacts,
            )

        if not isinstance(review, dict):
            return VerificationResult(
                passed=False,
                feedback="任务检查没有返回合法的 passed/feedback 结构。",
                evidence=evidence,
                artifacts=artifacts,
            )

        passed = review.get("passed") is True
        feedback = str(review.get("feedback") or ("任务满足 done_when" if passed else "任务未满足 done_when"))
        return VerificationResult(
            passed=passed,
            feedback=feedback,
            evidence=evidence,
            artifacts=artifacts,
        )

    @staticmethod
    def _structured_completion(
        task: TaskItem,
        observations: list[dict[str, Any]],
    ) -> tuple[bool, str] | None:
        """检查工具是否已经提供足够的聚类结构化证据。"""
        task_text = " ".join(
            str(value or "")
            for value in (task.title, task.task_goal, task.deliverable, task.done_when)
        ).lower()
        sentiment_repairs = [
            item.get("result") or {}
            for item in observations
            if item.get("tool") == "repair_sentiment_analysis"
            and isinstance(item.get("result"), dict)
        ]
        if sentiment_repairs:
            result = sentiment_repairs[-1]
            distribution = result.get("sentiment_distribution")
            if result.get("success") and isinstance(distribution, dict) and distribution:
                return True, "情感分析规则修复已复用已有标签并重建分布。"
            return False, "情感分析规则修复未返回可用的情感分布。"
        if "聚类" not in task_text and "kmeans" not in task_text:
            return None

        clustering_results = [
            item.get("result") or {}
            for item in observations
            if item.get("tool") in {"text_clustering", "repair_text_clustering"}
            and isinstance(item.get("result"), dict)
            and bool((item.get("result") or {}).get("success"))
        ]
        if not clustering_results:
            return None
        result = clustering_results[-1]
        distribution = result.get("cluster_distribution")
        profiles = result.get("cluster_profiles")
        if not isinstance(distribution, list) or not distribution:
            return False, "文本聚类工具未返回完整的主题簇分布结构。"
        if not isinstance(profiles, list) or len(profiles) < len(distribution):
            return False, "文本聚类工具未返回每个主题簇的特征描述证据。"

        distribution_ids = {
            str(item.get("cluster_id"))
            for item in distribution
            if isinstance(item, dict) and item.get("cluster_id") is not None
        }
        profile_ids = {
            str(item.get("cluster_id"))
            for item in profiles
            if isinstance(item, dict) and item.get("cluster_id") is not None
        }
        if distribution_ids != profile_ids:
            return False, "主题簇分布和特征描述的簇编号不完整或不一致。"
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("representative_texts"), list)
            or not any(str(text).strip() for text in item.get("representative_texts", []))
            for item in profiles
        ):
            return False, "部分主题簇缺少代表文本，无法形成可靠的特征描述。"

        expected = result.get("n_clusters")
        algorithm = str(
            result.get("algorithm_name") or result.get("algorithm") or ""
        ).lower()
        if algorithm in {"kmeans", "agglomerative"}:
            try:
                if int(expected) != len(distribution):
                    return False, f"聚类工具声明应有 {expected} 个主题簇，但实际只返回 {len(distribution)} 个。"
            except (TypeError, ValueError):
                return False, "聚类工具未返回可核对的主题簇数量。"
        return True, "文本聚类工具已返回完整簇分布和每簇代表文本，已通过结构化验收。"

    @staticmethod
    def _draft_text(draft: Any) -> str:
        if not isinstance(draft, list):
            return str(draft or "").strip()
        # Keep compatibility with adapters that wrap a response list once.
        while draft and isinstance(draft[0], list):
            draft = draft[-1]
        return assistant_content(draft).strip()

    @classmethod
    def _artifacts(cls, observations: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        for item in observations:
            result = item.get("result") or {}
            for key in cls._ARTIFACT_KEYS:
                value = result.get(key)
                values = value if isinstance(value, list) else [value]
                for raw in values:
                    if not raw or str(raw).startswith(("http://", "https://")):
                        continue
                    candidate = Path(str(raw)).expanduser()
                    if candidate.is_file() and str(candidate.resolve()) not in paths:
                        paths.append(str(candidate.resolve()))
        return paths

    @staticmethod
    def _summary(result: dict[str, Any]) -> str:
        if result.get("error"):
            return f"错误：{result['error']}"
        for key in ("answer", "message", "operation", "rows", "artifacts", "output_file"):
            value = result.get(key)
            if value not in (None, "", [], {}):
                text = str(value)
                return text if len(text) <= 240 else text[:237] + "..."
        return "工具返回成功" if result.get("success") else "工具未成功返回"
