"""Adapters for reusable text-processing implementations from the source project.

The source project is intentionally not edited. Imports are lazy so the main UI
can start even when large source models are not installed or configured.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

from academic_agent.infrastructure.runtime_safety import (
    configure_numeric_runtime,
    configure_torch_threads,
)

configure_numeric_runtime()

import numpy as np
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
        self._general_model_path: Path | None = None
        self._general_tokenizer: Any | None = None
        self._general_model: Any | None = None

    @staticmethod
    def _optional_path(name: str) -> Path | None:
        value = os.getenv(name, "").strip()
        return Path(value).expanduser().resolve() if value else None

    def set_model_root(self, model_root: str | Path) -> None:
        self.model_root = normalize_model_root(model_root)
        apply_model_root(self.model_root)
        self._general_model_path = None
        self._general_tokenizer = None
        self._general_model = None

    def reset_model_root(self) -> None:
        self.model_root = None
        apply_model_root(None)
        self._general_model_path = None
        self._general_tokenizer = None
        self._general_model = None

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
        configure_numeric_runtime()
        configure_torch_threads()
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

    @staticmethod
    def _prepare_embeddings(values: Any, expected_rows: int) -> np.ndarray:
        """Validate and safely normalize model vectors before BLAS/KMeans.

        Scaling by the largest absolute component first avoids overflow in
        L2 normalization even when a broken model returns unusually large
        finite values. Non-finite vectors are rejected explicitly so they do
        not reach native matrix multiplication.
        """

        embeddings = np.asarray(values, dtype=np.float64)
        if embeddings.ndim != 2:
            raise RuntimeError(f"文本向量维度异常：期望二维矩阵，实际为 {embeddings.shape}")
        if embeddings.shape[0] != expected_rows:
            raise RuntimeError(
                f"文本向量条数异常：期望 {expected_rows} 条，实际为 {embeddings.shape[0]} 条"
            )
        if embeddings.shape[1] == 0:
            raise RuntimeError("文本向量没有特征维度，无法进行聚类")
        if not np.isfinite(embeddings).all():
            bad_rows = int((~np.isfinite(embeddings).all(axis=1)).sum())
            raise RuntimeError(f"文本向量包含 {bad_rows} 个非有限值向量，已阻止进入 KMeans")

        max_abs = np.max(np.abs(embeddings), axis=1, keepdims=True)
        scaled = np.divide(
            embeddings,
            max_abs,
            out=np.zeros_like(embeddings),
            where=max_abs > 0,
        )
        norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
        return np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms > 1e-12)

    @staticmethod
    def _fit_cluster_labels(model: Any, embeddings: np.ndarray) -> np.ndarray:
        """Run sklearn under a one-thread native pool when available."""

        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return model.fit_predict(embeddings)
        with threadpool_limits(limits=1):
            return model.fit_predict(embeddings)

    def _clustering_in_process(
        self,
        data: pd.DataFrame,
        text_column: str,
        n_clusters: int,
        algorithm: str,
        eps: float,
        min_samples: int,
        output_file: str | None,
    ) -> pd.DataFrame:
        """Perform clustering without entering the source KMeans loop.

        The source adapter still supplies the BGE/SentenceTransformer vectors,
        but KMeans is run here after validation. This removes the source
        implementation's unsafe native path while retaining its model.
        """

        work = data[[text_column]].copy()
        work["_agent_row_id"] = list(range(len(work)))
        work = work.dropna(subset=[text_column])
        if work.empty:
            raise RuntimeError("没有可用于聚类的有效文本")

        # Import only the source embedding component. The source clustering
        # module eagerly constructs a second global model and then enters its
        # own KMeans implementation; both are unnecessary for this adapter.
        embedding_module = self._load(
            "text_processor_subagent_stage_3.common.textEmbedding"
        )
        embedder = getattr(embedding_module, "txt", None)
        if embedder is None:
            embedder = embedding_module.TextEmbedding()
        texts = work[text_column].astype(str).tolist()
        embeddings = embedder.get_query_vector(texts)
        if embeddings is None:
            raise RuntimeError("文本向量生成失败，无法进行聚类")
        embeddings = self._prepare_embeddings(embeddings, len(work))

        if algorithm in {"kmeans", "agglomerative"}:
            n_clusters = int(n_clusters)
            if n_clusters < 2:
                raise ValueError("聚类数至少为 2")
            if n_clusters > len(work):
                raise ValueError(f"聚类数 {n_clusters} 不能大于有效文本数 {len(work)}")

        if algorithm == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(
                n_clusters=int(n_clusters),
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=42,
            )
        elif algorithm == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(n_clusters=int(n_clusters))
        else:
            from sklearn.cluster import DBSCAN

            model = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=1)

        labels = self._fit_cluster_labels(model, embeddings)
        clustered = pd.DataFrame({
            "_agent_row_id": work["_agent_row_id"].to_numpy(),
            "cluster_id": labels,
        })
        if output_file:
            clustered.to_csv(output_file, index=False, encoding="utf-8-sig")
        return clustered

    def _clustering_isolated(self, **kwargs: Any) -> pd.DataFrame:
        """Run model-backed clustering outside the Qt process.

        A native SIGSEGV cannot be caught by Python. Using ``spawn`` means a
        failed model/BLAS worker can be reported as a normal tool error while
        the GUI and conversation remain alive.
        """

        import multiprocessing as mp
        import queue

        timeout = float(os.getenv("ACADEMIC_AGENT_CLUSTER_TIMEOUT", "1800"))
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        payload = {
            "source_root": str(self.source_root),
            "model_root": str(self.model_root) if self.model_root else None,
            **kwargs,
        }
        process = context.Process(
            target=_cluster_worker,
            args=(payload, result_queue),
            name="academic-agent-clustering",
        )
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise RuntimeError(f"聚类子进程超过 {int(timeout)} 秒未完成，已安全停止")

        try:
            message = result_queue.get(timeout=5)
        except queue.Empty:
            exit_code = process.exitcode
            if exit_code and exit_code < 0:
                raise RuntimeError(
                    f"聚类子进程被信号 {-exit_code} 中断；已隔离该崩溃，主程序未退出"
                )
            raise RuntimeError(f"聚类子进程异常退出（退出码 {exit_code}）")
        finally:
            result_queue.close()
            result_queue.join_thread()

        if not message.get("success"):
            detail = message.get("error") or "未知错误"
            if process.exitcode and process.exitcode < 0:
                detail = f"聚类子进程被信号 {-process.exitcode} 中断；{detail}"
            raise RuntimeError(detail)
        return pd.DataFrame(message.get("rows") or [])

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

    def _load_general_sentiment_model(self) -> tuple[Any, Any, dict[int, str]]:
        """Load the general model from the configured model-library root.

        The source implementation builds its path relative to its own source
        file, which is not reliable after PyInstaller packaging. The client
        already has a model-library setting, so resolve this model through the
        same setting instead.
        """

        model_root = self.model_root or default_model_root()
        model_path = model_root / "multilingual-sentiment-analysis"
        if not model_path.is_dir():
            raise FileNotFoundError(
                "通用五分类情感模型目录不存在："
                f"{model_path}\n"
                "请在“文本挖掘模型库根目录”中选择包含 "
                "multilingual-sentiment-analysis 子目录的 pretrain_models 文件夹。"
            )

        if (
            self._general_model_path == model_path
            and self._general_tokenizer is not None
            and self._general_model is not None
        ):
            return (
                self._general_tokenizer,
                self._general_model,
                {
                    0: "Very Negative",
                    1: "Negative",
                    2: "Neutral",
                    3: "Positive",
                    4: "Very Positive",
                },
            )

        self._prepare_imports()
        transformers = __import__(
            "transformers",
            fromlist=["AutoTokenizer", "AutoModelForSequenceClassification"],
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        model.eval()
        self._general_model_path = model_path
        self._general_tokenizer = tokenizer
        self._general_model = model
        return (
            tokenizer,
            model,
            {
                0: "Very Negative",
                1: "Negative",
                2: "Neutral",
                3: "Positive",
                4: "Very Positive",
            },
        )

    def general_sentiment(self, texts: list[str]) -> list[dict[str, Any]]:
        tokenizer, model, sentiment_map = self._load_general_sentiment_model()
        import torch

        normalized_texts = [str(text) for text in texts]
        rows: list[dict[str, Any]] = []
        batch_size = 16
        for start in range(0, len(normalized_texts), batch_size):
            batch_texts = normalized_texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=1).cpu().tolist()
            for index, predicted_index in enumerate(predictions):
                rows.append({
                    "text": batch_texts[index],
                    "sentiment": sentiment_map[int(predicted_index)],
                    "probability": float(
                        probabilities[index][int(predicted_index)].item()
                    ),
                })
        return rows

    def clustering(
        self,
        data: pd.DataFrame,
        text_column: str,
        n_clusters: int = 5,
        algorithm: str = "kmeans",
        eps: float = 0.5,
        min_samples: int = 5,
        output_file: str | None = None,
    ) -> pd.DataFrame:
        algorithm = str(algorithm or "kmeans").lower()
        if algorithm not in {"kmeans", "agglomerative", "dbscan"}:
            raise ValueError("聚类算法仅支持 kmeans、agglomerative 或 dbscan")
        kwargs = {
            "data": data.copy(),
            "text_column": text_column,
            "n_clusters": int(n_clusters),
            "algorithm": algorithm,
            "eps": float(eps),
            "min_samples": int(min_samples),
            "output_file": output_file,
        }
        if os.getenv("ACADEMIC_AGENT_DISABLE_CLUSTER_ISOLATION", "0") == "1":
            clustered = self._clustering_in_process(**kwargs)
        else:
            clustered = self._clustering_isolated(**kwargs)
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


def _cluster_worker(payload: dict[str, Any], result_queue: Any) -> None:
    """Multiprocessing entry point kept at module level for ``spawn``."""

    try:
        os.environ["ACADEMIC_AGENT_CLUSTER_WORKER"] = "1"
        adapter = SourceVideoTextAdapter(Path(payload["source_root"]))
        if payload.get("model_root"):
            adapter.set_model_root(payload["model_root"])
        result = adapter._clustering_in_process(
            data=payload["data"],
            text_column=payload["text_column"],
            n_clusters=payload["n_clusters"],
            algorithm=payload["algorithm"],
            eps=payload["eps"],
            min_samples=payload["min_samples"],
            output_file=payload.get("output_file"),
        )
        result_queue.put({"success": True, "rows": result.to_dict("records")})
    except BaseException as exc:
        import traceback

        result_queue.put({
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        })
