"""Adapters for reusable text-processing implementations from the source project.

The source project is intentionally not edited. Imports are lazy so the main UI
can start even when large source models are not installed or configured.
"""

from __future__ import annotations

import importlib
import os
import sys
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
    configured = os.getenv("TEXT_MINING_SOURCE_ROOT", "").strip()
    if configured and Path(configured).is_dir():
        return Path(configured).resolve()
    sibling_root = resource_root().parent
    if is_frozen():
        return (resource_root() / "text_mining_tools_agent_client_cpu").resolve()
    return (sibling_root / "text_mining_tools_agent_client_cpu").resolve()


class SourceVideoTextAdapter:
    """Expose the source project's preprocessor, embedding clusterer and BERT sentiment."""

    def __init__(self, source_root: Path | None = None) -> None:
        self.source_root = Path(source_root or _default_source_root()).resolve()
        configured_model_root = normalize_model_root(os.getenv("PRETRAINED_MODELS_DIR"))
        source_model_root = self.source_root / "pretrain_models"
        self.model_root: Path | None = configured_model_root or (
            source_model_root.resolve() if source_model_root.is_dir() else None
        )
        self.sentiment_model_path: Path | None = self._optional_path("SENTIMENT_MODEL_PATH")
        self.custom_dictionary_path: Path | None = self._optional_path("CUSTOM_DICTIONARY_PATH")

    @staticmethod
    def _optional_path(name: str) -> Path | None:
        value = os.getenv(name, "").strip()
        return Path(value).expanduser().resolve() if value else None

    @staticmethod
    def _runtime_log(tool: str, message: str) -> None:
        """Forward optional source-model progress into the active Agent run."""

        try:
            from academic_agent.agent.executor import emit_tool_log
            emit_tool_log(tool, message)
        except Exception:
            # Progress reporting must never make a model call fail, especially
            # when the adapter is used directly from a script or test.
            return

    def set_model_root(self, model_root: str | Path) -> None:
        self.model_root = normalize_model_root(model_root)
        apply_model_root(self.model_root)
        if self.model_root is not None:
            os.environ["TEXT_MINING_MODEL_DIR"] = str(self.model_root)

    def reset_model_root(self) -> None:
        self.model_root = None
        apply_model_root(None)
        os.environ.pop("TEXT_MINING_MODEL_DIR", None)

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
        package_root = self.source_root / "src_code"
        for path in (self.source_root, package_root):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        model_dir = self.model_root or default_model_root()
        os.environ.setdefault("PRETRAINED_MODELS_DIR", str(model_dir))
        # The reference CPU client resolves models through this variable.
        # Setting it only when absent preserves an explicitly configured
        # source-project value.
        os.environ.setdefault("TEXT_MINING_MODEL_DIR", str(model_dir))

    def _source_module(self, component: str) -> str:
        """Resolve a component from the polished CPU source project."""

        modules = {
            "preprocess": "src_code.preprocessing.data_preprocess",
            "embedding": "src_code.text_mining.embeddings_models.textEmbedding",
            "sentiment": "src_code.text_mining.sentiment.chinese_sentiment_bert_analysis",
            "general_sentiment": "src_code.text_mining.sentiment.sentiment_general_analysis",
            "keywords": "src_code.text_mining.keyword_extractor.textKeyBert_cluster_sentence_fuse_pro",
        }
        try:
            return modules[component]
        except KeyError as exc:
            raise ValueError(f"未知源项目组件：{component}") from exc

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

    @staticmethod
    def _normalize_cluster_algorithm(algorithm: str | None) -> str:
        value = str(algorithm or "kmeans").strip().lower()
        aliases = {
            "k-means": "kmeans",
            "k均值": "kmeans",
            "minibatch": "minibatch_kmeans",
            "mini_batch_kmeans": "minibatch_kmeans",
            "hierarchical": "agglomerative",
            "层次聚类": "agglomerative",
            "gaussian_mixture": "gmm",
            "gaussian mixture": "gmm",
            "高斯混合": "gmm",
            "谱聚类": "spectral",
            "亲和传播": "affinity_propagation",
            "affinity": "affinity_propagation",
            "mean_shift": "meanshift",
            "均值漂移": "meanshift",
        }
        return aliases.get(value, value)

    def _clustering_in_process(
        self,
        data: pd.DataFrame,
        text_column: str,
        n_clusters: int,
        algorithm: str,
        eps: float,
        min_samples: int,
        output_file: str | None,
        linkage: str = "ward",
        covariance_type: str = "full",
        damping: float = 0.5,
        bandwidth: float | None = None,
        max_iter: int = 300,
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
        embedding_module = self._load(self._source_module("embedding"))
        embedder = embedding_module.TextEmbedding(
            log_callback=lambda message: self._runtime_log(
                "text_clustering", str(message)
            ),
            progress_callback=lambda value: self._runtime_log(
                "text_clustering", f"模型加载/向量提取进度：{value}%"
            ),
        )
        texts = work[text_column].astype(str).tolist()
        try:
            embeddings = embedder.get_query_vector(texts, batch_size=64)
        except TypeError:
            embeddings = embedder.get_query_vector(texts)
        if embeddings is None:
            raise RuntimeError("文本向量生成失败，无法进行聚类")
        embeddings = self._prepare_embeddings(embeddings, len(work))

        algorithm = self._normalize_cluster_algorithm(algorithm)
        requires_clusters = {
            "kmeans", "minibatch_kmeans", "agglomerative", "gmm", "spectral", "birch"
        }
        if algorithm in requires_clusters:
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
                max_iter=int(max_iter),
                random_state=42,
            )
        elif algorithm == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans

            model = MiniBatchKMeans(
                n_clusters=int(n_clusters),
                max_iter=int(max_iter),
                random_state=42,
                n_init=3,
            )
        elif algorithm == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            if linkage == "ward" and not np.issubdtype(embeddings.dtype, np.number):
                linkage = "average"
            model = AgglomerativeClustering(
                n_clusters=int(n_clusters), linkage=str(linkage)
            )
        elif algorithm == "gmm":
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(
                n_components=int(n_clusters),
                covariance_type=str(covariance_type),
                max_iter=int(max_iter),
                random_state=42,
            )
        elif algorithm == "spectral":
            from sklearn.cluster import SpectralClustering

            neighbors = min(10, max(2, len(work) - 1)) if len(work) > 2 else None
            params = {
                "n_clusters": int(n_clusters),
                "random_state": 42,
                "n_init": 10,
            }
            if neighbors is not None:
                params.update({"affinity": "nearest_neighbors", "n_neighbors": neighbors})
            else:
                params["affinity"] = "rbf"
            model = SpectralClustering(**params)
        elif algorithm == "birch":
            from sklearn.cluster import Birch

            model = Birch(n_clusters=int(n_clusters))
        elif algorithm == "affinity_propagation":
            from sklearn.cluster import AffinityPropagation

            model = AffinityPropagation(
                damping=float(damping), max_iter=int(max_iter), random_state=42
            )
        elif algorithm == "meanshift":
            from sklearn.cluster import MeanShift

            model = MeanShift(bandwidth=bandwidth)
        else:
            from sklearn.cluster import DBSCAN

            if algorithm != "dbscan":
                raise ValueError(f"不支持的聚类算法：{algorithm}")
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
        return importlib.import_module(module_name)

    def preprocess(self, texts: list[str]) -> list[str]:
        module = self._load(self._source_module("preprocess"))
        processor = module.Data_PreProcessor()
        return processor.preprocess_data([str(text) for text in texts])

    def sentiment(self, texts: list[str], top_k: int = 1, batch_size: int = 8) -> list[dict[str, Any]]:
        module = self._load(self._source_module("sentiment"))
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
        kwargs = {"top_k": top_k, "batch_size": batch_size}
        kwargs["progress_callback"] = lambda message: self._runtime_log(
            "sentiment_analysis", str(message)
        )
        return module.predict_emotion([str(text) for text in texts], **kwargs)

    def general_sentiment(self, texts: list[str]) -> list[dict[str, Any]]:
        module = self._load(self._source_module("general_sentiment"))
        results = module.predict_sentiment(
            [str(text) for text in texts],
            batch_size=16,
            progress_callback=lambda message: self._runtime_log(
                "sentiment_analysis", str(message)
            ),
        )
        if not isinstance(results, list):
            raise RuntimeError("新版通用情感分析模块返回结果格式异常")
        return [
            {
                "text": item.get("text", ""),
                "sentiment": item.get("predicted_sentiment", "Neutral"),
                "probability": float(item.get("probability", 0.0)),
            }
            for item in results
        ]

    def clustering(
        self,
        data: pd.DataFrame,
        text_column: str,
        n_clusters: int = 5,
        algorithm: str = "kmeans",
        eps: float = 0.5,
        min_samples: int = 5,
        linkage: str = "ward",
        covariance_type: str = "full",
        damping: float = 0.5,
        bandwidth: float | None = None,
        max_iter: int = 300,
        output_file: str | None = None,
    ) -> pd.DataFrame:
        algorithm = self._normalize_cluster_algorithm(algorithm)
        supported = {
            "kmeans", "minibatch_kmeans", "dbscan", "agglomerative", "gmm",
            "spectral", "birch", "affinity_propagation", "meanshift",
        }
        if algorithm not in supported:
            raise ValueError(
                "聚类算法仅支持：" + "、".join(sorted(supported))
            )
        kwargs = {
            "data": data.copy(),
            "text_column": text_column,
            "n_clusters": int(n_clusters),
            "algorithm": algorithm,
            "eps": float(eps),
            "min_samples": int(min_samples),
            "linkage": str(linkage),
            "covariance_type": str(covariance_type),
            "damping": float(damping),
            "bandwidth": None if bandwidth in (None, "") else float(bandwidth),
            "max_iter": int(max_iter),
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
        module = self._load(self._source_module("keywords"))
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
            linkage=payload.get("linkage", "ward"),
            covariance_type=payload.get("covariance_type", "full"),
            damping=payload.get("damping", 0.5),
            bandwidth=payload.get("bandwidth"),
            max_iter=payload.get("max_iter", 300),
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
