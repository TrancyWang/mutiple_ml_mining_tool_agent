"""向量存储服务 - 使用 ChromaDB 实现轻量级向量数据库"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from academic_agent.infrastructure.model_paths import default_model_root, normalize_model_root
from academic_agent.infrastructure.runtime_paths import app_data_root
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    Settings = None
    HAS_CHROMADB = False

logger = logging.getLogger(__name__)


def _where_equals(**conditions: str) -> Dict[str, Any]:
    """构造兼容 Chroma 新版校验器的 metadata 过滤条件。"""
    items = [{key: {"$eq": value}} for key, value in conditions.items() if value is not None]
    if not items:
        return {}
    if len(items) == 1:
        return items[0]
    return {"$and": items}


class BGEEmbeddingFunction:
    """Lazy local embedding function backed by the source project's bge-cn."""

    def __init__(self, model_path: str | None = None):
        self.model_path = (normalize_model_root(model_path) or default_model_root()) / "bge-cn"
        self.model = None

    def set_model_root(self, model_root: str | Path) -> None:
        self.model_path = (normalize_model_root(model_root) or default_model_root()) / "bge-cn"
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return self.model
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"BGE 模型目录不存在：{self.model_path}")
        from sentence_transformers import SentenceTransformer
        import torch

        device_name = os.getenv("BGE_DEVICE", "")
        if not device_name:
            device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(str(self.model_path), device=device_name)
        logger.info("✅ 已加载 Chroma BGE embedding：%s (%s)", self.model_path, device_name)
        return self.model

    def __call__(self, input):
        return self._load_model().encode(
            [str(document) for document in input],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


class VectorStoreService:
    """ChromaDB 向量存储服务
    
    特点：
    - 轻量级，无需 Docker
    - 嵌入式数据库，数据存储在本地
    - 支持语义搜索
    - 适合中小规模应用
    """
    
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(app_data_root() / "chroma_db")
        self.db_path = str(db_path)
        self.collection_name = "text_mining_memories_bge_cn"
        self.embedding_function = BGEEmbeddingFunction()
        self.client = None
        self.collection = None
        self._initialized = False
        
    def initialize(self):
        """初始化 ChromaDB 客户端和集合"""
        if self._initialized:
            return

        if not HAS_CHROMADB:
            logger.warning("⚠️ 未安装 ChromaDB，记忆功能降级为短期记忆；安装 requirements.txt 后可启用长期记忆")
            return
        
        try:
            # 确保数据库目录存在
            if not os.path.exists(self.db_path):
                os.makedirs(self.db_path, exist_ok=True)
                logger.info(f"📁 创建向量数据库目录: {self.db_path}")
            
            # 初始化 ChromaDB 客户端（持久化模式）
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            # Chroma's default ONNX embedding uses ~/.cache/chroma. Keep the
            # model cache beside the project's persistent database instead of
            # relying on a possibly read-only system cache directory.
            try:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
                ONNXMiniLM_L6_V2.DOWNLOAD_PATH = os.path.abspath(
                    os.path.join(self.db_path, "onnx_models")
                )
                os.makedirs(ONNXMiniLM_L6_V2.DOWNLOAD_PATH, exist_ok=True)
            except Exception as embedding_cache_error:
                logger.warning(f"⚠️ 无法设置 Chroma embedding 缓存目录：{embedding_cache_error}")
            logger.info(f"✅ 成功初始化 ChromaDB: {self.db_path}")
            
            # 创建或获取集合
            self._create_collection()
            self._initialized = True
            logger.info(f"✅ ChromaDB 集合 '{self.collection_name}' 已就绪")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB 初始化失败: {e}")
            logger.warning("⚠️  将继续运行，但记忆功能将不可用")
            self._initialized = False

    def set_model_root(self, model_root: str | Path) -> None:
        """切换 BGE 路径；已加载的 embedding 模型会在下次使用时重载。"""
        self.embedding_function.set_model_root(model_root)

    def reset_model_root(self) -> None:
        self.embedding_function.model_path = default_model_root() / "bge-cn"
        self.embedding_function.model = None
    
    def _create_collection(self):
        """创建或加载集合"""
        try:
            # 尝试获取现有集合
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"📦 加载现有集合: {self.collection_name}")
            except:
                # 集合不存在，创建新集合
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Text mining conversation memories",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"✨ 创建新集合: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"❌ 创建集合失败: {e}")
            raise
    
    def add_memory(self, user_id: str, text: str, memory_type: str = "conversation",
                  metadata: Optional[Dict] = None) -> bool:
        """添加一条记忆
        
        Args:
            user_id: 用户ID
            text: 文本内容
            memory_type: 记忆类型 (conversation/summary/profile)
            metadata: 额外元数据
            
        Returns:
            是否成功
        """
        if not self._initialized:
            logger.warning("ChromaDB 未初始化，跳过记忆存储")
            return False
        
        try:
            # 生成唯一ID
            memory_id = f"{user_id}_{datetime.now().timestamp()}"
            
            # 准备元数据
            memory_metadata = {
                "user_id": user_id,
                "text": text,
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
            }
            
            # 合并额外元数据
            if metadata:
                memory_metadata.update(metadata)
            
            # 添加到集合（ChromaDB 会自动生成 embedding）
            self.collection.add(
                ids=[memory_id],
                documents=[text],
                metadatas=[memory_metadata]
                , embeddings=self.embedding_function([text])
            )
            
            logger.debug(f"💾 记忆已存储: user_id={user_id}, type={memory_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 记忆存储失败: {e}")
            return False
    
    def search_memories(self, user_id: str, query: str, 
                       memory_type: Optional[str] = None,
                       top_k: int = 5) -> List[Dict]:
        """搜索相关记忆
        
        Args:
            user_id: 用户ID
            query: 查询文本
            memory_type: 可选的记忆类型过滤
            top_k: 返回结果数量
            
        Returns:
            匹配的记忆列表
        """
        if not self._initialized:
            logger.warning("ChromaDB 未初始化，无法检索记忆")
            return []
        
        try:
            # 构建过滤条件
            where_filter = _where_equals(user_id=user_id, type=memory_type)
            
            # 执行语义搜索
            results = self.collection.query(
                query_embeddings=self.embedding_function([query]),
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # 解析结果
            memories = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    memory = {
                        "id": doc_id,
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    }
                    memories.append(memory)
            
            logger.debug(f"🔍 检索到 {len(memories)} 条记忆")
            return memories
            
        except Exception as e:
            logger.error(f"❌ 记忆检索失败: {e}")
            return []
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户的最近对话历史
        
        Args:
            user_id: 用户ID
            limit: 返回的对话数量
            
        Returns:
            对话历史列表
        """
        if not self._initialized:
            return []
        
        try:
            # 获取所有对话类型的记忆
            results = self.collection.get(
                where=_where_equals(user_id=user_id, type="conversation"),
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            conversations = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    conversations.append({
                        "id": doc_id,
                        "text": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    })
            
            # 按时间戳排序（最新的在前）
            conversations.sort(
                key=lambda x: x['metadata'].get('timestamp', ''),
                reverse=True
            )
            
            return conversations[:limit]
            
        except Exception as e:
            logger.error(f"❌ 获取对话历史失败: {e}")
            return []
    
    def delete_user_memories(self, user_id: str, memory_type: Optional[str] = None):
        """删除用户的记忆
        
        Args:
            user_id: 用户ID
            memory_type: 可选的记忆类型，不指定则删除所有类型
        """
        if not self._initialized:
            return
        
        try:
            where_filter = _where_equals(user_id=user_id, type=memory_type)
            
            # 获取要删除的记录
            results = self.collection.get(where=where_filter)
            
            if results['ids']:
                # 删除记录
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️  已删除用户记忆: user_id={user_id}, count={len(results['ids'])}")
            else:
                logger.info(f"ℹ️  用户 {user_id} 没有记忆需要删除")
            
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")

    def delete_session_memories(self, user_id: str, session_id: str) -> None:
        """Best-effort permanent deletion of one session's semantic memories."""
        if not self._initialized:
            return
        try:
            results = self.collection.get(where=_where_equals(
                user_id=str(user_id), session_id=str(session_id)
            ))
            if results.get("ids"):
                self.collection.delete(ids=results["ids"])
        except Exception as exc:
            logger.warning("⚠️ 删除 Chroma 会话记忆失败: %s", exc)
    
    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict:
        """获取记忆统计信息
        
        Args:
            user_id: 可选的用户ID，不指定则返回全局统计
            
        Returns:
            统计信息字典
        """
        if not self._initialized:
            return {}
        
        try:
            stats = {"total": 0, "by_type": {}}
            
            # 获取所有记录
            if user_id:
                results = self.collection.get(where={"user_id": user_id})
            else:
                results = self.collection.get()
            
            total_count = len(results['ids']) if results['ids'] else 0
            stats["total"] = total_count
            
            # 按类型统计
            if results['metadatas']:
                type_counts = {}
                for metadata in results['metadatas']:
                    mem_type = metadata.get('type', 'unknown')
                    type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                
                stats["by_type"] = type_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭连接"""
        if self._initialized:
            try:
                logger.info("🔒 ChromaDB 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 ChromaDB 连接失败: {e}")


# 全局服务实例
vector_store = VectorStoreService()
