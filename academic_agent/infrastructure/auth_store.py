"""本地登录信息存储。

登录信息与对话记忆共用应用数据目录下的 SQLite 文件，但使用独立的
``auth_accounts`` 表，避免依赖 Agent 服务或云端数据库。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from academic_agent.infrastructure.runtime_paths import app_data_root


class AuthStore:
    """保存最近一次登录的用户名和模型配置。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path or os.getenv("SQLITE_MEMORY_PATH", "")
        path = Path(configured).expanduser() if configured else app_data_root() / "data" / "academic_agent.db"
        if not path.is_absolute():
            path = app_data_root() / path
        self.db_path = path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_accounts (
                    username TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    api_key TEXT NOT NULL DEFAULT '',
                    remember_api_key INTEGER NOT NULL DEFAULT 0,
                    last_login TEXT NOT NULL
                )
                """
            )

    def last_account(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT username, provider, api_key, remember_api_key FROM auth_accounts "
                "ORDER BY last_login DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def save_account(
        self,
        username: str,
        provider: str,
        api_key: str = "",
        remember_api_key: bool = False,
    ) -> None:
        stored_key = api_key if remember_api_key else ""
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_accounts
                    (username, provider, api_key, remember_api_key, last_login)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    provider = excluded.provider,
                    api_key = excluded.api_key,
                    remember_api_key = excluded.remember_api_key,
                    last_login = excluded.last_login
                """,
                (
                    str(username).strip(),
                    str(provider).strip(),
                    stored_key,
                    int(bool(remember_api_key)),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def delete_account(self, username: str) -> None:
        """删除明确退出登录的本地账户记录。历史对话不会被删除。"""
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_accounts WHERE username = ?",
                (str(username).strip(),),
            )


auth_store = AuthStore()
