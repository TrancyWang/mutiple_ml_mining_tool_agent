"""Elasticsearch + Milvus Lite 混合记忆后端。

Elasticsearch 保存可按会话和关键词查询的完整消息；Milvus 保存 BGE 向量，
用于语义召回。两个后端都采用可选连接，未启动外部服务时不会阻断 Agent。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover - optional dependency
    Elasticsearch = None

try:
    from pymilvus import DataType, MilvusClient
except ImportError:  # pragma: no cover - optional dependency
    DataType = None
    MilvusClient = None

# pymilvus 对本地 .db 使用动态导入；显式保留这个可选入口，确保 PyInstaller
# 不会把 Milvus Lite 从发布包中裁掉。
try:  # pragma: no cover - optional dependency
    import milvus_lite.server_manager as _milvus_lite_server_manager  # noqa: F401
except ImportError:
    _milvus_lite_server_manager = None

try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery
    from llama_index.vector_stores.milvus import MilvusVectorStore
    HAS_LLAMAINDEX = True
except ImportError:  # pragma: no cover - optional dependency
    BaseEmbedding = None
    TextNode = None
    VectorStoreQuery = None
    MilvusVectorStore = None
    HAS_LLAMAINDEX = False

from academic_agent.infrastructure.persistence.chroma_store import BGEEmbeddingFunction
from academic_agent.infrastructure.runtime_paths import app_data_root


if HAS_LLAMAINDEX:
    class BGEIndexEmbedding(BaseEmbedding):
        """将现有 bge-cn 模型适配为 LlamaIndex embedding。"""

        model_name: str = "bge-cn"

        def __init__(self, embedding_function: BGEEmbeddingFunction, **kwargs: Any):
            super().__init__(**kwargs)
            self._embedding_function = embedding_function

        def _get_text_embedding(self, text: str) -> List[float]:
            return self._embedding_function([text])[0]

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._embedding_function([query])[0]

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)


class ElasticsearchMemory:
    def __init__(self) -> None:
        self.enabled = os.getenv("ES_ENABLED", "false").lower() == "true"
        self.host = os.getenv("ES_HOST", "http://localhost:9200")
        self.index = os.getenv("ES_INDEX", "academic_agent_conversations")
        self.client = None
        if not self.enabled or Elasticsearch is None:
            return
        try:
            self.client = Elasticsearch(
                self.host,
                basic_auth=(os.getenv("ES_USER", "elastic"), os.getenv("ES_PASSWORD"))
                if os.getenv("ES_PASSWORD") else None,
                verify_certs=os.getenv("ES_VERIFY_CERTS", "false").lower() == "true",
                request_timeout=10,
            )
            if self.client.ping():
                self._ensure_index()
                logger.info("✅ Elasticsearch 记忆已连接: %s", self.host)
            else:
                logger.warning("⚠️ Elasticsearch ping 失败，暂时禁用")
                self.client = None
        except Exception as exc:
            logger.warning("⚠️ Elasticsearch 初始化失败: %s", exc)
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(index=self.index, mappings={
            "properties": {
                "user_id": {"type": "keyword"},
                "session_id": {"type": "keyword"},
                "conversation_id": {"type": "keyword"},
                "role": {"type": "keyword"},
                "content": {"type": "text"},
                "model_name": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "metadata": {"type": "object", "enabled": True},
            }
        })

    def save_message(self, user_id: str, role: str, content: str,
                     session_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.available:
            return False
        try:
            conversation_id = session_id or user_id
            self.client.index(index=self.index, document={
                "user_id": str(user_id),
                "session_id": str(session_id or user_id),
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "model_name": os.getenv("DEFAULT_MODEL", "academic-agent"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metadata": metadata or {},
            })
            return True
        except Exception as exc:
            logger.warning("⚠️ Elasticsearch 保存消息失败: %s", exc)
            return False

    def get_messages(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            result = self.client.search(
                index=self.index,
                query={"term": {"user_id": str(user_id)}},
                size=limit,
                sort=[{"timestamp": "asc"}],
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as exc:
            logger.warning("⚠️ Elasticsearch 查询消息失败: %s", exc)
            return []

    def search_messages(self, user_id: str, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            result = self.client.search(
                index=self.index,
                query={"bool": {"must": [
                    {"term": {"user_id": str(user_id)}},
                    {"match": {"content": keyword}},
                ]}},
                size=limit,
                sort=[{"timestamp": "desc"}],
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as exc:
            logger.warning("⚠️ Elasticsearch 关键词检索失败: %s", exc)
            return []

    def delete_user(self, user_id: str) -> None:
        if self.available:
            try:
                self.client.delete_by_query(
                    index=self.index,
                    query={"term": {"user_id": str(user_id)}},
                    conflicts="proceed",
                )
            except Exception as exc:
                logger.warning("⚠️ Elasticsearch 删除记忆失败: %s", exc)

    def delete_session(self, user_id: str, session_id: str) -> None:
        if self.available:
            try:
                self.client.delete_by_query(
                    index=self.index,
                    query={"bool": {"filter": [
                        {"term": {"user_id": str(user_id)}},
                        {"term": {"session_id": str(session_id)}},
                    ]}},
                    conflicts="proceed",
                    refresh=True,
                )
            except Exception as exc:
                logger.warning("⚠️ Elasticsearch 删除会话失败: %s", exc)


class MilvusSemanticMemory:
    def __init__(self) -> None:
        self.enabled = os.getenv("MILVUS_ENABLED", "false").lower() == "true"
        configured_uri = os.getenv("MILVUS_URI", "")
        self.uri = configured_uri or str(
            app_data_root() / "milvus_data" / "academic_memories.db"
        )
        if not Path(self.uri).is_absolute():
            self.uri = str(app_data_root() / self.uri)
        # LlamaIndex 维护的集合使用独立名称，避免与旧版手写 schema 冲突。
        self.collection_name = os.getenv("MILVUS_COLLECTION", "academic_agent_memories_llamaindex")
        self.embedding = BGEEmbeddingFunction()
        self.client = None
        self.vector_store = None
        self.dim = int(os.getenv("MILVUS_EMBEDDING_DIM", "768"))
        if not self.enabled or MilvusVectorStore is None:
            return
        try:
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
            self.vector_store = MilvusVectorStore(
                uri=self.uri,
                collection_name=self.collection_name,
                dim=self.dim,
                similarity_metric="COSINE",
                overwrite=False,
                stores_text=True,
                is_embedding_query=True,
                use_async_client=False,
                stores_node=False,
            )
            # Milvus Lite 新建/加载后默认可能处于 released 状态。
            self.vector_store.client.load_collection(self.collection_name)
            logger.info("✅ LlamaIndex + Milvus Lite 记忆已启用: %s", self.uri)
        except Exception as exc:
            logger.warning("⚠️ LlamaIndex/Milvus Lite 初始化失败，语义记忆将使用 Chroma 兜底: %s", exc)
            self.vector_store = None

    @property
    def available(self) -> bool:
        return self.vector_store is not None

    def insert(self, user_id: str, text: str, memory_type: str = "conversation",
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.available:
            return False
        try:
            node = TextNode(
                id_=f"{user_id}_{time.time_ns()}",
                text=text,
                metadata={
                    "user_id": str(user_id),
                    "memory_type": memory_type,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **(metadata or {}),
                },
            )
            node.embedding = self.embedding([text])[0]
            self.vector_store.add([node])
            return True
        except Exception as exc:
            logger.warning("⚠️ Milvus 写入失败: %s", exc)
            return False

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        workspace_path: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            index_embedding = BGEIndexEmbedding(self.embedding)
            query_obj = VectorStoreQuery(
                query_embedding=index_embedding.get_query_embedding(query),
                similarity_top_k=max(top_k * 5, top_k),
            )
            result = self.vector_store.query(query_obj)
            memories = []
            for node, score in zip(result.nodes or [], result.similarities or []):
                metadata = node.metadata or {}
                if str(metadata.get("user_id")) != str(user_id):
                    continue
                if str(metadata.get("workspace_path") or "") != str(workspace_path or ""):
                    continue
                memories.append({
                    "text": node.get_content(),
                    "metadata": {
                        "user_id": metadata.get("user_id", user_id),
                        "type": metadata.get("memory_type", "conversation"),
                        "timestamp": metadata.get("timestamp", ""),
                        "session_id": metadata.get("session_id", ""),
                        "workspace_path": metadata.get("workspace_path", ""),
                    },
                    "distance": score,
                })
            return memories[:top_k]
        except Exception as exc:
            logger.warning("⚠️ Milvus 语义检索失败: %s", exc)
            return []

    def delete_user(self, user_id: str) -> None:
        # MilvusVectorStore 当前版本没有统一的 metadata delete API；
        # 用户级清理由 collection 重建/后续管理任务处理，不影响跨轮次召回。
        return None

    def delete_session(self, user_id: str, session_id: str) -> None:
        if not self.available:
            return
        try:
            safe_user = str(user_id).replace('"', '\\"')
            safe_session = str(session_id).replace('"', '\\"')
            self.vector_store.client.delete(
                collection_name=self.collection_name,
                filter=f'user_id == "{safe_user}" and session_id == "{safe_session}"',
            )
        except Exception as exc:
            logger.warning("⚠️ Milvus 删除会话记忆失败: %s", exc)


class HybridMemoryService:
    def __init__(self) -> None:
        self.elasticsearch = ElasticsearchMemory()
        self.milvus = MilvusSemanticMemory()

    def reload(self) -> None:
        """按最新设置重新加载本地 ES/Milvus 连接。"""
        self.elasticsearch = ElasticsearchMemory()
        self.milvus = MilvusSemanticMemory()

    def save_conversation(self, user_id: str, user_text: str, assistant_response: str,
                          session_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        self.elasticsearch.save_message(
            user_id, "user", user_text, session_id=session_id, metadata=metadata
        )
        if assistant_response:
            self.elasticsearch.save_message(
                user_id, "assistant", assistant_response,
                session_id=session_id, metadata=metadata,
            )

    def save_semantic_memory(self, user_id: str, text: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        return self.milvus.insert(user_id, text, metadata=metadata)

    def search_semantic(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        workspace_path: str | None = None,
    ) -> List[Dict[str, Any]]:
        return self.milvus.search(
            user_id,
            query,
            top_k=top_k,
            workspace_path=workspace_path,
        )

    def clear_user(self, user_id: str) -> None:
        self.elasticsearch.delete_user(user_id)
        self.milvus.delete_user(user_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        self.elasticsearch.delete_session(user_id, session_id)
        self.milvus.delete_session(user_id, session_id)


hybrid_memory = HybridMemoryService()
