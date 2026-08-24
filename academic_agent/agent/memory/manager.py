"""记忆管理器 - 实现短期记忆和长期记忆管理"""

import logging
import os
import threading
from typing import List, Dict
from datetime import datetime
from academic_agent.infrastructure.persistence.chroma_store import vector_store
from academic_agent.infrastructure.persistence.hybrid_store import hybrid_memory
from academic_agent.repositories.sqlite_store import sqlite_memory

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器
    
    负责：
    1. 短期记忆管理（最近对话）
    2. 长期记忆管理（重要信息抽取和存储）
    3. 记忆检索和上下文构建
    """
    
    def __init__(self):
        # 短期记忆：每个用户的最近对话历史（内存中）
        self.short_term_memories: Dict[str, List[Dict]] = {}
        self._deleted_sessions: set[tuple[str, str]] = set()
        self._memory_write_lock = threading.RLock()
        
        # 最大短期记忆长度
        self.max_short_term_length = 20
        
        # SQLite 是默认且始终可用的完整历史存储。Chroma 不再在默认路径
        # 自动启动，避免用户未选择 ES/Milvus 时仍创建第三套记忆数据库。
        if os.getenv("CHROMA_ENABLED", "false").lower() == "true":
            try:
                vector_store.initialize()
                logger.info("✅ 可选 Chroma 语义存储已初始化")
            except Exception as e:
                logger.warning(f"⚠️ 可选 Chroma 初始化失败: {e}")
        logger.info(
            "🧠 混合记忆后端: SQLite=%s, ES=%s, Milvus=%s",
            sqlite_memory.db_path,
            hybrid_memory.elasticsearch.available,
            hybrid_memory.milvus.available,
        )
    
    def add_to_short_term(self, user_id: str, role: str, content: str,
                          session_id: str | None = None,
                          workspace_path: str | None = None):
        """添加到短期记忆
        
        Args:
            user_id: 用户ID
            role: 角色 (user/assistant)
            content: 内容
        """
        memory_key = f"{user_id}:{session_id or user_id}"
        if memory_key not in self.short_term_memories:
            self.short_term_memories[memory_key] = []
        
        # 添加新消息
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.short_term_memories[memory_key].append(message)

        # SQLite 保存完整历史，确保程序重启后仍能恢复最近对话。
        try:
            sqlite_memory.save_message(
                user_id,
                role,
                content,
                session_id=session_id,
                workspace_path=workspace_path,
            )
        except Exception as exc:
            logger.warning(f"⚠️ SQLite 历史对话保存失败: {exc}")
        
        # 限制短期记忆长度
        if len(self.short_term_memories[memory_key]) > self.max_short_term_length:
            # 保留最近的 N 条消息
            self.short_term_memories[memory_key] = self.short_term_memories[memory_key][-self.max_short_term_length:]
        
        logger.debug(f"📝 短期记忆已更新: key={memory_key}, length={len(self.short_term_memories[memory_key])}")
    
    def get_short_term_context(self, user_id: str, session_id: str | None = None) -> List[Dict]:
        """获取短期记忆上下文
        
        Args:
            user_id: 用户ID
            
        Returns:
            对话历史列表
        """
        return self.short_term_memories.get(f"{user_id}:{session_id or user_id}", [])
    
    def clear_short_term(self, user_id: str):
        """清除短期记忆
        
        Args:
            user_id: 用户ID
        """
        prefix = f"{user_id}:"
        for key in [item for item in self.short_term_memories if item.startswith(prefix)]:
            del self.short_term_memories[key]
        if not any(item.startswith(prefix) for item in self.short_term_memories):
            logger.info(f"🗑️  已清除用户短期记忆: user_id={user_id}")
    
    def process_and_store_memory(self, user_id: str, user_text: str,
                                assistant_response: str = "",
                                session_id: str | None = None,
                                workspace_path: str | None = None):
        """处理并存储到长期记忆
        
        Args:
            user_id: 用户ID
            user_text: 用户输入文本
            assistant_response: 助手回复（可选）
        """
        session_key = (str(user_id), str(session_id or user_id))
        if session_key in self._deleted_sessions:
            return
        try:
            # 判断是否值得存储到长期记忆
            importance_score = self._calculate_importance(user_text)
            
            combined_text = (
                f"用户: {user_text}\n助手: {assistant_response}"
                if assistant_response else user_text
            )
            with self._memory_write_lock:
                if session_key in self._deleted_sessions:
                    return
                if importance_score >= 0.5 and os.getenv("CHROMA_ENABLED", "false").lower() == "true":
                    vector_store.add_memory(
                        user_id=user_id,
                        text=combined_text,
                        memory_type="conversation",
                        metadata={
                            "importance_score": importance_score,
                            "user_text": user_text,
                            "assistant_response": assistant_response,
                            "session_id": session_id or "",
                            "workspace_path": workspace_path or "",
                        },
                    )
                elif importance_score < 0.5:
                    logger.debug(f"⚪ 低重要性对话，不存储到长期记忆: score={importance_score:.2f}")

                # ES 保存完整对话；Milvus 只保存达到阈值的语义记忆。
                hybrid_memory.save_conversation(
                    user_id=user_id,
                    user_text=user_text,
                    assistant_response=assistant_response,
                    session_id=session_id,
                    metadata={
                        "importance_score": importance_score,
                        "session_id": session_id,
                        "workspace_path": workspace_path or "",
                    },
                )
                if importance_score >= 0.5:
                    hybrid_memory.save_semantic_memory(
                        user_id=user_id,
                        text=combined_text,
                        metadata={
                            "importance_score": importance_score,
                            "session_id": session_id or "",
                            "workspace_path": workspace_path or "",
                        },
                    )
        
        except Exception as e:
            logger.error(f"❌ 处理长期记忆失败: {e}")
    
    def retrieve_relevant_memories(
        self,
        user_id: str,
        current_query: str,
        top_k: int = 3,
        workspace_path: str | None = None,
    ) -> str:
        """检索相关记忆并构建上下文
        
        Args:
            user_id: 用户ID
            current_query: 当前查询
            top_k: 返回的记忆数量
            
        Returns:
            构建好的上下文字符串
        """
        try:
            # 优先使用 Milvus 语义召回；不可用时回退到已有 Chroma。
            memories = hybrid_memory.search_semantic(
                user_id,
                current_query,
                top_k=top_k,
                workspace_path=workspace_path,
            )
            if not memories and os.getenv("CHROMA_ENABLED", "false").lower() == "true":
                candidates = vector_store.search_memories(
                    user_id=user_id,
                    query=current_query,
                    memory_type="conversation",
                    top_k=max(top_k * 5, top_k),
                )
                expected_workspace = str(workspace_path or "")
                memories = [
                    memory for memory in candidates
                    if str(memory.get("metadata", {}).get("workspace_path") or "") == expected_workspace
                ][:top_k]
            
            if not memories:
                return ""
            
            # 构建上下文
            context_parts = ["【相关历史记忆】"]
            
            for i, memory in enumerate(memories, 1):
                text = memory.get('text', '')
                timestamp = memory.get('metadata', {}).get('timestamp', '')
                
                # 格式化时间戳
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        time_str = "未知时间"
                else:
                    time_str = "未知时间"
                
                context_parts.append(f"{i}. [{time_str}] {text}")
            
            context = "\n".join(context_parts)
            logger.info(f"✅ 检索到 {len(memories)} 条相关记忆")
            return context
            
        except Exception as e:
            logger.error(f"❌ 检索记忆失败: {e}")
            return ""
    
    def build_context_with_memory(
        self,
        user_id: str,
        current_query: str,
        session_id: str | None = None,
        workspace_path: str | None = None,
        use_short_term: bool = True,
        use_long_term: bool = True,
    ) -> str:
        """构建包含记忆的完整上下文
        
        Args:
            user_id: 用户ID
            current_query: 当前查询
            use_short_term: 是否使用短期记忆
            use_long_term: 是否使用长期记忆
            
        Returns:
            完整的上下文字符串
        """
        context_parts = []
        
        # 1. 短期记忆（最近对话）
        if use_short_term:
            try:
                persistent_recent = sqlite_memory.get_recent_messages(
                    user_id,
                    limit=5,
                    session_id=session_id,
                    workspace_path=workspace_path if workspace_path is not None else "",
                )
            except Exception as exc:
                logger.warning(f"⚠️ SQLite 历史对话读取失败: {exc}")
                persistent_recent = []
            recent_messages = persistent_recent or self.get_short_term_context(user_id, session_id)
            if recent_messages:
                context_parts.append("【最近对话】")
                for msg in recent_messages[-5:]:
                    role_label = "用户" if msg['role'] == 'user' else "助手"
                    context_parts.append(f"{role_label}: {msg['content']}")
        
        # 2. 长期记忆（相关历史）
        if use_long_term:
            long_term_context = self.retrieve_relevant_memories(
                user_id=user_id,
                current_query=current_query,
                top_k=3,
                workspace_path=workspace_path,
            )
            if long_term_context:
                context_parts.append(long_term_context)
        
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return ""
    
    def _calculate_importance(self, text: str) -> float:
        """计算文本的重要性评分 (0~1)
        
        评分标准：
        - 包含技术关键词 → +0.3
        - 包含问题或请求 → +0.2
        - 长度较长（详细说明） → +0.2
        - 包含特定操作意图 → +0.3
        
        Args:
            text: 输入文本
            
        Returns:
            重要性评分 (0~1)
        """
        score = 0.0
        
        # 检查技术关键词
        tech_keywords = [
            "分析", "挖掘", "聚类", "分类", "回归", "情感", "关键词",
            "数据", "模型", "算法", "特征", "训练", "预测",
            "预处理", "清洗", "可视化", "图表", "统计"
        ]
        
        if any(keyword in text for keyword in tech_keywords):
            score += 0.3
        
        # 检查是否包含问题或请求
        question_indicators = ["如何", "怎么", "什么", "为什么", "请", "帮我", "能否"]
        if any(indicator in text for indicator in question_indicators):
            score += 0.2
        
        # 检查长度（详细说明通常更重要）
        if len(text) > 50:
            score += 0.2
        elif len(text) > 20:
            score += 0.1
        
        # 检查操作意图
        action_keywords = ["加载", "上传", "执行", "运行", "生成", "创建", "删除"]
        if any(keyword in text for keyword in action_keywords):
            score += 0.3
        
        # 限制在 0~1 范围内
        score = min(max(score, 0.0), 1.0)
        
        return round(score, 2)
    
    def get_user_memory_stats(self, user_id: str) -> Dict:
        """获取用户的记忆统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        stats = vector_store.get_memory_stats(user_id)
        stats["sqlite_path"] = str(sqlite_memory.db_path)
        stats["sqlite_message_count"] = sqlite_memory.count_messages(user_id)
        stats["elasticsearch_enabled"] = hybrid_memory.elasticsearch.available
        stats["milvus_enabled"] = hybrid_memory.milvus.available
        prefix = f"{user_id}:"
        stats["short_term_count"] = sum(
            len(messages)
            for key, messages in self.short_term_memories.items()
            if key.startswith(prefix)
        )
        return stats
    
    def clear_all_memories(self, user_id: str):
        """清除用户的所有记忆
        
        Args:
            user_id: 用户ID
        """
        # 清除短期记忆
        self.clear_short_term(user_id)
        
        # 清除长期记忆
        vector_store.delete_user_memories(user_id)
        hybrid_memory.clear_user(user_id)
        sqlite_memory.delete_user(user_id)
        
        logger.info(f"🗑️  已清除用户所有记忆: user_id={user_id}")

    def delete_session(self, user_id: str, session_id: str) -> None:
        """Permanently remove one session from every enabled memory backend."""
        with self._memory_write_lock:
            self._deleted_sessions.add((str(user_id), str(session_id)))
            self.short_term_memories.pop(f"{user_id}:{session_id}", None)
            vector_store.delete_session_memories(user_id, session_id)
            hybrid_memory.delete_session(user_id, session_id)


# 全局记忆管理器实例
memory_manager = MemoryManager()
