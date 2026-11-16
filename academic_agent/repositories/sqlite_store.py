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
        # journal_mode is a database-level setting. Reapplying WAL on every
        # read connection can itself wait for a writer and freeze the Qt
        # event loop while a workspace is being switched.
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    workspace_path TEXT NOT NULL DEFAULT '',
                    workspace_id TEXT NOT NULL DEFAULT '',
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    board_json TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    runtime_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runs_recovery
                    ON agent_runs(user_id, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_run_events_run
                    ON agent_run_events(run_id, seq ASC);
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    workspace_path TEXT NOT NULL,
                    output_path TEXT NOT NULL DEFAULT '',
                    permission_mode TEXT NOT NULL DEFAULT 'read_only',
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_workspaces_user_time
                    ON workspaces(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS workspace_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL,
                    artifact_type TEXT NOT NULL DEFAULT 'file',
                    sha256 TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(workspace_id, path, sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_workspace_artifacts_lookup
                    ON workspace_artifacts(workspace_id, updated_at DESC);
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
            run_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(agent_runs)").fetchall()
            }
            if "workspace_id" not in run_columns:
                connection.execute(
                    "ALTER TABLE agent_runs ADD COLUMN workspace_id TEXT NOT NULL DEFAULT ''"
                )
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

    @staticmethod
    def _decode_json(value: str | None, default: Any) -> Any:
        try:
            return json.loads(value or "")
        except (TypeError, json.JSONDecodeError):
            return default

    @classmethod
    def _row_to_agent_run(cls, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "user_id": str(row["user_id"]),
            "session_id": str(row["session_id"]),
            "workspace_path": str(row["workspace_path"] or ""),
            "workspace_id": str(row["workspace_id"] or "") if "workspace_id" in row.keys() else "",
            "request": cls._decode_json(row["request_json"], {}),
            "plan": cls._decode_json(row["plan_json"], {}),
            "board": cls._decode_json(row["board_json"], {}),
            "messages": cls._decode_json(row["messages_json"], []),
            "runtime": cls._decode_json(row["runtime_json"], {}),
            "status": str(row["status"]),
            "last_error": str(row["last_error"] or ""),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "completed_at": row["completed_at"],
        }

    def save_agent_run(
        self,
        run_id: str,
        user_id: str,
        session_id: str,
        workspace_path: str | None,
        request: dict[str, Any],
        plan: dict[str, Any],
        board: dict[str, Any],
        messages: list[dict[str, Any]],
        runtime: dict[str, Any] | None = None,
        status: str = "running",
        last_error: str = "",
    ) -> None:
        """保存一个可恢复的 Agent 检查点；同一 run_id 更新同一条记录。"""
        now = datetime.now(timezone.utc).isoformat()
        project = str(Path(workspace_path).expanduser().resolve()) if workspace_path else ""
        workspace_id = str(request.get("workspace_id") or "")
        payload = (
            str(run_id),
            str(user_id),
            str(session_id),
            project,
            workspace_id,
            json.dumps(request, ensure_ascii=False, default=str),
            json.dumps(plan, ensure_ascii=False, default=str),
            json.dumps(board, ensure_ascii=False, default=str),
            json.dumps(messages, ensure_ascii=False, default=str),
            json.dumps(runtime or {}, ensure_ascii=False, default=str),
            str(status),
            str(last_error or ""),
            now,
            now,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, user_id, session_id, workspace_path, workspace_id,
                    request_json, plan_json, board_json, messages_json,
                    runtime_json, status, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    session_id = excluded.session_id,
                    workspace_path = excluded.workspace_path,
                    workspace_id = excluded.workspace_id,
                    request_json = excluded.request_json,
                    plan_json = excluded.plan_json,
                    board_json = excluded.board_json,
                    messages_json = excluded.messages_json,
                    runtime_json = excluded.runtime_json,
                    status = excluded.status,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at,
                    completed_at = CASE
                        WHEN excluded.status = 'completed' THEN excluded.updated_at
                        ELSE NULL
                    END
                """,
                payload,
            )

    def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return self._row_to_agent_run(row) if row is not None else None

    def ensure_workspace(
        self,
        workspace_id: str,
        user_id: str,
        session_id: str,
        workspace_path: str | Path,
        output_path: str | Path | None = None,
        permission_mode: str = "read_only",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or refresh the durable identity for a task workspace."""
        now = datetime.now(timezone.utc).isoformat()
        root = str(Path(workspace_path).expanduser().resolve())
        output = str(Path(output_path).expanduser().resolve()) if output_path else str(Path(root) / "output")
        payload = (
            str(workspace_id), str(user_id), str(session_id), root, output,
            str(permission_mode), "active",
            json.dumps(metadata or {}, ensure_ascii=False, default=str), now, now,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspaces (
                    workspace_id, user_id, session_id, workspace_path, output_path,
                    permission_mode, status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    session_id = excluded.session_id,
                    workspace_path = excluded.workspace_path,
                    output_path = excluded.output_path,
                    permission_mode = excluded.permission_mode,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
        return self.get_workspace(workspace_id) or {
            "workspace_id": str(workspace_id),
            "user_id": str(user_id),
            "session_id": str(session_id),
            "workspace_path": root,
            "output_path": output,
            "permission_mode": str(permission_mode),
            "status": "active",
            "metadata": metadata or {},
        }

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?",
                (str(workspace_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return {
            "workspace_id": str(row["workspace_id"]),
            "user_id": str(row["user_id"]),
            "session_id": str(row["session_id"]),
            "workspace_path": str(row["workspace_path"]),
            "output_path": str(row["output_path"] or ""),
            "permission_mode": str(row["permission_mode"] or "read_only"),
            "status": str(row["status"] or "active"),
            "metadata": metadata if isinstance(metadata, dict) else {},
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def register_workspace_artifact(
        self,
        workspace_id: str,
        path: str | Path,
        *,
        run_id: str = "",
        task_id: str = "",
        artifact_type: str = "file",
        sha256: str = "",
        size_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a real output without copying its contents into SQLite."""
        import uuid

        now = datetime.now(timezone.utc).isoformat()
        artifact_id = uuid.uuid4().hex
        payload = (
            artifact_id, str(workspace_id), str(run_id), str(task_id), str(path),
            str(artifact_type), str(sha256), max(0, int(size_bytes)),
            json.dumps(metadata or {}, ensure_ascii=False, default=str), now, now,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspace_artifacts (
                    artifact_id, workspace_id, run_id, task_id, path, artifact_type,
                    sha256, size_bytes, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, path, sha256) DO UPDATE SET
                    run_id = excluded.run_id,
                    task_id = excluded.task_id,
                    artifact_type = excluded.artifact_type,
                    size_bytes = excluded.size_bytes,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
            row = connection.execute(
                """
                SELECT * FROM workspace_artifacts
                WHERE workspace_id = ? AND path = ? AND sha256 = ?
                """,
                (str(workspace_id), str(path), str(sha256)),
            ).fetchone()
        return {
            "artifact_id": str(row["artifact_id"]),
            "workspace_id": str(row["workspace_id"]),
            "run_id": str(row["run_id"]),
            "task_id": str(row["task_id"]),
            "path": str(row["path"]),
            "artifact_type": str(row["artifact_type"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"] or 0),
        }

    def list_workspace_artifacts(self, workspace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM workspace_artifacts
                WHERE workspace_id = ? ORDER BY updated_at DESC LIMIT ?
                """,
                (str(workspace_id), max(1, int(limit))),
            ).fetchall()
        return [
            {
                "artifact_id": str(row["artifact_id"]),
                "workspace_id": str(row["workspace_id"]),
                "run_id": str(row["run_id"]),
                "task_id": str(row["task_id"]),
                "path": str(row["path"]),
                "artifact_type": str(row["artifact_type"]),
                "sha256": str(row["sha256"]),
                "size_bytes": int(row["size_bytes"] or 0),
            }
            for row in rows
        ]

    def list_recoverable_agent_runs(
        self,
        user_id: str,
        workspace_path: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """列出被正常完成标记之外的运行，供重启后继续。"""
        statuses = ("prepared", "running", "interrupted")
        query = (
            "SELECT * FROM agent_runs WHERE user_id = ? "
            "AND status IN (?, ?, ?)"
        )
        params: list[Any] = [str(user_id), *statuses]
        if workspace_path is not None:
            project = str(Path(workspace_path).expanduser().resolve()) if workspace_path else ""
            query += " AND workspace_path = ?"
            params.append(project)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_agent_run(row) for row in rows]

    def finish_agent_run(self, run_id: str, status: str = "completed", last_error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status = ?, last_error = ?, updated_at = ?,
                    completed_at = CASE WHEN ? = 'completed' THEN ? ELSE completed_at END
                WHERE run_id = ?
                """,
                (str(status), str(last_error or ""), now, str(status), now, str(run_id)),
            )

    def append_agent_event(
        self,
        run_id: str,
        event_id: str,
        seq: int,
        event_type: str,
        event: dict[str, Any],
    ) -> None:
        """追加一条脱敏后的运行事件；同一序号不会重复写入。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO agent_run_events
                    (event_id, run_id, seq, event_type, event_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_id),
                    str(run_id),
                    int(seq),
                    str(event_type),
                    json.dumps(event, ensure_ascii=False, default=str),
                    now,
                ),
            )

    def list_agent_events(self, run_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        """按运行顺序读取事件，供 UI、诊断和 replay 使用。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_json FROM agent_run_events
                WHERE run_id = ? ORDER BY seq ASC LIMIT ?
                """,
                (str(run_id), max(1, int(limit))),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                value = json.loads(row["event_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def get_agent_event_cursor(self, run_id: str) -> int:
        """返回某次运行已落盘事件的最后序号，用于恢复时继续追加。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS cursor FROM agent_run_events WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return int(row["cursor"] or 0) if row is not None else 0

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
