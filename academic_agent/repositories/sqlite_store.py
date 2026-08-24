"""SQLite 会话历史存储。

SQLite 保存完整消息和顺序，向量数据库只负责语义召回。该模块不依赖
Elasticsearch，也不需要启动任何服务。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from academic_agent.infrastructure.runtime_paths import app_data_root


BASE_DIR = app_data_root()


class SQLiteConversationStore:
    """持久化保存对话消息，并提供 FTS5 关键词检索。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path or os.getenv("SQLITE_MEMORY_PATH", "")
        configured_path = Path(configured).expanduser() if configured else BASE_DIR / "data" / "academic_agent.db"
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        self.db_path = configured_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_user_time
                    ON conversation_messages(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_conversation_session_time
                    ON conversation_messages(session_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS workspace_projects (
                    user_id TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, workspace_path)
                );
                CREATE INDEX IF NOT EXISTS idx_workspace_projects_recent
                    ON workspace_projects(user_id, last_used_at DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts USING fts5(
                    message_id UNINDEXED,
                    user_id UNINDEXED,
                    role,
                    content
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversation_messages)").fetchall()
            }
            if "workspace_path" not in columns:
                connection.execute(
                    "ALTER TABLE conversation_messages ADD COLUMN workspace_path TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_workspace_time "
                "ON conversation_messages(user_id, workspace_path, created_at DESC)"
            )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "workspace_path": row["workspace_path"] if "workspace_path" in row.keys() else "",
            "role": row["role"],
            "content": row["content"],
            "metadata": metadata,
            "timestamp": row["created_at"],
        }

    def save_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: str | None = None,
        workspace_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if not content:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        owner = str(user_id)
        conversation = str(session_id or user_id)
        project = str(Path(workspace_path).expanduser().resolve()) if workspace_path else ""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO conversation_messages
                    (user_id, session_id, workspace_path, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (owner, conversation, project, str(role), str(content),
                 json.dumps(metadata or {}, ensure_ascii=False, default=str), now),
            )
            message_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO conversation_messages_fts(message_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                (str(message_id), owner, str(role), str(content)),
            )
            return message_id

    def register_workspace(self, user_id: str, workspace_path: str | Path) -> None:
        """登记用户使用过的项目，即使项目暂时还没有产生对话。"""
        project = Path(workspace_path).expanduser().resolve()
        if not project.is_dir():
            return
        owner = str(user_id)
        name = project.name or str(project)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspace_projects
                    (user_id, workspace_path, name, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, workspace_path) DO UPDATE SET
                    name = excluded.name,
                    last_used_at = excluded.last_used_at
                """,
                (owner, str(project), name, now, now),
            )

    def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
        session_id: str | None = None,
        workspace_path: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM conversation_messages WHERE user_id = ?"
        params: list[Any] = [str(user_id)]
        if session_id:
            query += " AND session_id = ?"
            params.append(str(session_id))
        if workspace_path is not None:
            project = str(Path(workspace_path).expanduser().resolve()) if workspace_path else ""
            query += " AND workspace_path = ?"
            params.append(project)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    def get_recent_sessions(
        self,
        user_id: str,
        limit: int = 20,
        workspace_path: str | None = None,
        include_unbound: bool = True,
        bound_only: bool = False,
    ) -> list[dict[str, Any]]:
        """读取最近的完整会话摘要，而不是把每条用户消息当成一条历史。

        ``session_id`` 是 Qt 客户端在新建对话时生成的稳定 ID。旧版本没有
        session ID 的消息会落在 ``user_id`` 这个兼容会话中，仍然可以恢复。
        """
        amount = max(1, int(limit))
        project_filter = ""
        params: list[Any] = [str(user_id)]
        if workspace_path is not None:
            project = str(Path(workspace_path).expanduser().resolve()) if workspace_path else ""
            if include_unbound:
                project_filter = " AND (workspace_path = ? OR workspace_path = '')"
                params.append(project)
            else:
                project_filter = " AND workspace_path = ?"
                params.append(project)
        elif bound_only:
            project_filter = " AND workspace_path <> ''"
        params.append(amount)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    grouped.user_id,
                    grouped.session_id,
                    grouped.workspace_path,
                    grouped.first_id,
                    grouped.last_id,
                    grouped.message_count,
                    grouped.started_at,
                    grouped.last_at,
                    COALESCE((
                        SELECT content
                        FROM conversation_messages AS first_user
                        WHERE first_user.user_id = grouped.user_id
                          AND first_user.session_id = grouped.session_id
                          AND first_user.workspace_path = grouped.workspace_path
                          AND first_user.role = 'user'
                        ORDER BY first_user.id ASC
                        LIMIT 1
                    ), '新会话') AS title
                FROM (
                    SELECT
                        user_id,
                        session_id,
                        workspace_path,
                        MIN(id) AS first_id,
                        MAX(id) AS last_id,
                        COUNT(*) AS message_count,
                        MIN(created_at) AS started_at,
                        MAX(created_at) AS last_at
                    FROM conversation_messages
                    WHERE user_id = ?
                """ + project_filter + """
                    GROUP BY user_id, session_id, workspace_path
                ) AS grouped
                ORDER BY grouped.last_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "workspace_path": row["workspace_path"] or "",
                "is_unbound": not bool(row["workspace_path"]),
                "title": str(row["title"] or "新会话").strip(),
                "message_count": int(row["message_count"] or 0),
                "first_id": int(row["first_id"] or 0),
                "last_id": int(row["last_id"] or 0),
                "started_at": row["started_at"],
                "last_at": row["last_at"],
            }
            for row in rows
        ]

    def get_recent_projects(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """读取用户最近使用过的项目工作区，包括尚无对话的新项目。"""
        with self._connect() as connection:
            registered_rows = connection.execute(
                """
                SELECT workspace_path, name, last_used_at
                FROM workspace_projects
                WHERE user_id = ?
                """,
                (str(user_id),),
            ).fetchall()
            conversation_rows = connection.execute(
                """
                SELECT workspace_path, MAX(id) AS last_id, COUNT(DISTINCT session_id) AS session_count
                FROM conversation_messages
                WHERE user_id = ? AND workspace_path <> ''
                GROUP BY workspace_path
                ORDER BY last_id DESC
                LIMIT ?
                """,
                (str(user_id), max(1, int(limit))),
            ).fetchall()
        projects: dict[str, dict[str, Any]] = {}
        for row in registered_rows:
            path = str(row["workspace_path"])
            projects[path] = {
                "workspace_path": path,
                "name": str(row["name"] or Path(path).name or path),
                "session_count": 0,
                "last_id": 0,
                "last_used_at": str(row["last_used_at"] or ""),
            }
        for row in conversation_rows:
            path = str(row["workspace_path"])
            project = projects.setdefault(path, {
                "workspace_path": path,
                "name": Path(path).name or path,
                "session_count": 0,
                "last_id": 0,
                "last_used_at": "",
            })
            project["session_count"] = int(row["session_count"] or 0)
            project["last_id"] = int(row["last_id"] or 0)
        return sorted(
            projects.values(),
            key=lambda item: (int(item.get("last_id", 0)), str(item.get("last_used_at", ""))),
            reverse=True,
        )[: max(1, int(limit))]

    def search_messages(self, user_id: str, keyword: str, limit: int = 20) -> list[dict[str, Any]]:
        keyword = str(keyword).strip()
        if not keyword:
            return []
        # FTS5 对特殊查询字符较敏感，使用引号按短语检索，避免用户输入破坏查询。
        phrase = '"' + keyword.replace('"', '""') + '"'
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT messages.*
                FROM conversation_messages_fts AS fts
                JOIN conversation_messages AS messages ON messages.id = CAST(fts.message_id AS INTEGER)
                WHERE fts.user_id = ? AND conversation_messages_fts MATCH ?
                ORDER BY messages.id DESC LIMIT ?
                """,
                (str(user_id), phrase, max(1, int(limit))),
            ).fetchall()
            if not rows:
                # SQLite 默认 unicode61 tokenizer 对中文短语支持有限，使用 LIKE 兜底。
                rows = connection.execute(
                    """
                    SELECT * FROM conversation_messages
                    WHERE user_id = ? AND content LIKE ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (str(user_id), f"%{keyword}%", max(1, int(limit))),
                ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def count_messages(self, user_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM conversation_messages WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return int(row["count"] if row else 0)

    def delete_session(
        self,
        user_id: str,
        session_id: str,
        workspace_path: str | None = None,
    ) -> int:
        """Permanently delete one complete conversation and its FTS rows."""
        owner = str(user_id)
        conversation = str(session_id)
        project = str(Path(workspace_path).expanduser().resolve()) if workspace_path else ""
        with self._lock, self._connect() as connection:
            ids = connection.execute(
                "SELECT id FROM conversation_messages "
                "WHERE user_id = ? AND session_id = ? AND workspace_path = ?",
                (owner, conversation, project),
            ).fetchall()
            message_ids = [str(row["id"]) for row in ids]
            if message_ids:
                connection.executemany(
                    "DELETE FROM conversation_messages_fts WHERE message_id = ?",
                    [(message_id,) for message_id in message_ids],
                )
            connection.execute(
                "DELETE FROM conversation_messages "
                "WHERE user_id = ? AND session_id = ? AND workspace_path = ?",
                (owner, conversation, project),
            )
        return len(message_ids)

    def delete_workspace(self, user_id: str, workspace_path: str | Path) -> dict[str, Any]:
        """Delete only Agent metadata, conversations, and FTS rows for a project."""
        owner = str(user_id)
        project = str(Path(workspace_path).expanduser().resolve())
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id FROM conversation_messages "
                "WHERE user_id = ? AND workspace_path = ?",
                (owner, project),
            ).fetchall()
            message_ids = [str(row["id"]) for row in rows]
            session_ids = sorted({str(row["session_id"]) for row in rows})
            if message_ids:
                connection.executemany(
                    "DELETE FROM conversation_messages_fts WHERE message_id = ?",
                    [(message_id,) for message_id in message_ids],
                )
            connection.execute(
                "DELETE FROM conversation_messages WHERE user_id = ? AND workspace_path = ?",
                (owner, project),
            )
            connection.execute(
                "DELETE FROM workspace_projects WHERE user_id = ? AND workspace_path = ?",
                (owner, project),
            )
        return {"message_count": len(message_ids), "session_ids": session_ids}

    def delete_user(self, user_id: str) -> None:
        owner = str(user_id)
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM conversation_messages_fts WHERE user_id = ?", (owner,))
            connection.execute("DELETE FROM conversation_messages WHERE user_id = ?", (owner,))


sqlite_memory = SQLiteConversationStore()
