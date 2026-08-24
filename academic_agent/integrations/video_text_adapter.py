"""Adapters for reusable text-processing implementations from the source project.

The source project is intentionally not edited. Imports are lazy so the main UI
can start even when large source models are not installed or configured.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from academic_agent.infrastructure.model_paths import apply_model_root, default_model_root, normalize_model_root
from academic_agent.infrastructure.runtime_paths import is_frozen, resource_root


def _default_source_root() -> Path:
    configured = os.getenv("VIDEO_AGENT_SOURCE_ROOT", "").strip()
    if configured and Path(configured).is_dir():
        return Path(configured).resolve()
    if is_frozen():
        return (resource_root() / "video_text_mutiplemodal_agent").resolve()
    return (resource_root().parent / "video_text_mutiplemodal_agent").resolve()


class SourceVideoTextAdapter:
    """Expose the source project's preprocessor, embedding clusterer and BERT sentiment."""

    def __init__(self, source_root: Path | None = None) -> None:
        self.source_root = Path(source_root or _default_source_root()).resolve()
        self.model_root: Path | None = normalize_model_root(os.getenv("PRETRAINED_MODELS_DIR"))
        self.sentiment_model_path: Path | None = self._optional_path("SENTIMENT_MODEL_PATH")
        self.custom_dictionary_path: Path | None = self._optional_path("CUSTOM_DICTIONARY_PATH")

    @staticmethod
    def _optional_path(name: str) -> Path | None:
        value = os.getenv(name, "").strip()
        return Path(value).expanduser().resolve() if value else None

    def set_model_root(self, model_root: str | Path) -> None:
        self.model_root = normalize_model_root(model_root)
        apply_model_root(self.model_root)

    def reset_model_root(self) -> None:
        self.model_root = None
        apply_model_root(None)

    def set_sentiment_model_path(self, model_path: str | Path) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"情感分析模型目录不存在：{path}")
        self.sentiment_model_path = path
        os.environ["SENTIMENT_MODEL_PATH"] = str(path)

    def reset_sentiment_model_path(self) -> None:
        self.sentiment_model_path = None
        os.environ.pop("SENTIMENT_MODEL_PATH", None)

    def set_custom_dictionary(self, dictionary_path: str | Path) -> None:
        path = Path(dictionary_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"自定义词典不存在：{path}")
        self.custom_dictionary_path = path
        os.environ["CUSTOM_DICTIONARY_PATH"] = str(path)

    def reset_custom_dictionary(self) -> None:
        self.custom_dictionary_path = None
        os.environ.pop("CUSTOM_DICTIONARY_PATH", None)

    def _load_custom_dictionary(self) -> None:
        if not self.custom_dictionary_path:
            return
        import jieba
        jieba.load_userdict(str(self.custom_dictionary_path))

    def _prepare_imports(self) -> None:
        # UMAP is imported by the source clusterer even when dimensionality
        # reduction is disabled. Prevent numba from writing to a restricted
        # system cache during local execution.
        os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        src_root = self.source_root / "src_codes"
        main_root = src_root / "main_process_agent"
        for path in (src_root, main_root):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        model_dir = self.model_root or default_model_root()
        os.environ.setdefault("PRETRAINED_MODELS_DIR", str(model_dir))

    def _load(self, module_name: str) -> Any:
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"源项目不存在：{self.source_root}")
        self._prepare_imports()
        if module_name.endswith(".clustering") and "umap" not in sys.modules:
            # The source clusterer imports UMAP unconditionally, although the
            # standard path below does not reduce dimensions. Keep the source
            # implementation intact while avoiding an optional numba cache
            # write during normal clustering.
            umap_stub = types.ModuleType("umap")
            umap_stub.UMAP = type(
                "UMAP",
                (),
                {"__init__": lambda self, *args, **kwargs: None,
                 "fit_transform": lambda self, values: values},
            )
            sys.modules["umap"] = umap_stub
        return importlib.import_module(module_name)

    def preprocess(self, texts: list[str]) -> list[str]:
        module = self._load("text_processor_subagent_stage_3.common.data_preprocess")
        processor = module.Data_PreProcessor()
        return processor.preprocess_data([str(text) for text in texts])

    def sentiment(self, texts: list[str], top_k: int = 1, batch_size: int = 8) -> list[dict[str, Any]]:
        module = self._load("text_processor_subagent_stage_3.sentiment.chinese_sentiment_bert_analysis")
        # 源实现保留不动；这里仅在适配层替换其已加载模型，支持用户选择独立模型目录。
        if self.sentiment_model_path:
            transformers = __import__("transformers", fromlist=["AutoTokenizer", "AutoModelForSequenceClassification"])
            module.tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.sentiment_model_path), local_files_only=True
            )
            module.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self.sentiment_model_path), local_files_only=True
            ).to(module.device).eval()
            module.id2label = module.model.config.id2label
            module.label2id = module.model.config.label2id
        return module.predict_emotion([str(text) for text in texts], top_k=top_k, batch_size=batch_size)

    def general_sentiment(self, texts: list[str]) -> list[dict[str, Any]]:
        module = self._load("text_processor_subagent_stage_3.sentiment.sentiment_general_analysis")
        rows = []
        for text in texts:
            probabilities, prediction = module.predict_sentiment([str(text)])
            predicted_index = int(prediction)
            rows.append({
                "text": str(text),
                "sentiment": module.sentiment_map[predicted_index],
                "probability": float(probabilities[0][predicted_index].item()),
            })
        return rows

    def clustering(
        self,
        data: pd.DataFrame,
        text_column: str,
        n_clusters: int = 5,
        output_file: str | None = None,
    ) -> pd.DataFrame:
        module = self._load("text_processor_subagent_stage_3.clustering.clustering")
        work = data[[text_column]].copy()
        work["_agent_row_id"] = list(range(len(work)))
        work = work.dropna(subset=[text_column])
        with tempfile.TemporaryDirectory(prefix="academic_cluster_") as temp_dir:
            temp_input = Path(temp_dir) / "source_cluster_input.csv"
            work.rename(columns={text_column: "text_for_bert"}).to_csv(temp_input, index=False, encoding="utf-8-sig")
            clusterer = module.VideoTextClustering()
            clustered = clusterer.cluster_video_texts(
                str(temp_input),
                n_clusters=n_clusters,
                is_silhouette=False,
                is_reduce_dimension=False,
                output_file=output_file,
                text_column="text_for_bert",
                id_column="_agent_row_id",
            )
            return clustered.rename(columns={"_agent_row_id": "row_id", "cluster_id": "cluster"})

    def keywords_by_group(
        self,
        data: pd.DataFrame,
        text_column: str,
        group_column: str | None = None,
        id_column: str | None = None,
        top_n: int = 35,
    ) -> list[dict[str, Any]]:
        """Run the source project's KeyBERT fusion model by cluster or corpus."""
        self._load_custom_dictionary()
        module = self._load(
            "text_processor_subagent_stage_3.clustering.textKeyBert_cluster_sentence_fuse_agent"
        )
        model = module.TextKeyBert()
        groups = data.groupby(group_column, dropna=False) if group_column else [("all", data)]
        rows: list[dict[str, Any]] = []
        for group_name, group in groups:
            docs = []
            for row_index, row in group.iterrows():
                text = row.get(text_column)
                if pd.isna(text) or not str(text).strip():
                    continue
                comment_id = row.get(id_column) if id_column else row_index
                like = row.get("like", row.get("digg_count", 0))
                reply = row.get("reply", row.get("reply_count", 0))
                docs.append({
                    "comment_id": str(comment_id),
                    "text": str(text),
                    "like": 0 if pd.isna(like) else like,
                    "reply": 0 if pd.isna(reply) else reply,
                })
            if not docs:
                continue
            keywords = model.get_key_words(docs, res_top_n=top_n)
            rows.append({
                "source_file": Path(self.source_root).name,
                "cluster": str(group_name),
                "segment_count": len(docs),
                "comment_count": len({doc["comment_id"] for doc in docs}),
                "keywords": str(keywords),
            })
        return rows


source_video_text_adapter = SourceVideoTextAdapter()

