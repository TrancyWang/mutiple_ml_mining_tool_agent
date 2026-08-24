"""认证和模型凭据用例；不依赖 Qt。"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from academic_agent.models.auth import AuthSession
from academic_agent.infrastructure.auth_store import AuthStore, auth_store
from academic_agent.infrastructure.runtime_paths import env_candidates


ROOT_USERNAME = "trancy"
ROOT_PASSWORD = "handsomeman"

_CLOUD_KEY_NAMES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ALIYUN_API_KEY",
    "DASHSCOPE_API_KEY_BJ",
    "DASHSCOPE_API_KEY_SG",
    "QWEN_API_KEY_SG",
)


class AuthController:
    def __init__(self, store: AuthStore = auth_store) -> None:
        self.store = store

    @staticmethod
    def _read_env(path: Path) -> None:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            line = line.removeprefix("export ").strip()
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                os.environ[key] = value

    def load_root_env(self) -> Path | None:
        env_file = next((path for path in env_candidates() if path.is_file()), None)
        if env_file is None:
            return None
        try:
            from dotenv import load_dotenv

            load_dotenv(str(env_file), override=True)
        except ImportError:
            self._read_env(env_file)
        return env_file

    def last_account(self) -> dict | None:
        return self.store.last_account()

    def saved_session(self) -> AuthSession | None:
        account = self.last_account()
        if not account:
            return None
        username = str(account.get("username", "")).strip()
        provider = str(account.get("provider", "")).strip().lower()
        if username == ROOT_USERNAME and provider == "env":
            if any(path.is_file() for path in env_candidates()):
                return AuthSession(username=ROOT_USERNAME, is_root=True, provider="env")
            return None
        api_key = str(account.get("api_key", "")).strip()
        if provider in {"gemini", "qwen"} and account.get("remember_api_key") and api_key:
            return AuthSession(username=username, is_root=False, provider=provider, api_key=api_key)
        return None

    def authenticate(
        self,
        username: str,
        password: str,
        provider: str,
        api_key: str,
        remember_api_key: bool,
    ) -> AuthSession:
        username = username.strip()
        if not username:
            raise ValueError("请输入用户名。")
        is_root = hmac.compare_digest(username, ROOT_USERNAME) and hmac.compare_digest(
            password, ROOT_PASSWORD
        )
        if is_root:
            session = AuthSession(username=ROOT_USERNAME, is_root=True, provider="env")
            self.store.save_account(username, "env")
            return session
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("普通用户必须填写 Gemini 或千问 API Key。")
        provider = provider if provider in {"gemini", "qwen"} else "gemini"
        session = AuthSession(username=username, is_root=False, provider=provider, api_key=api_key)
        self.store.save_account(
            username,
            provider,
            api_key,
            remember_api_key=remember_api_key,
        )
        return session

    def apply(self, session: AuthSession) -> Path | None:
        if session.is_root:
            return self.load_root_env()
        self.clear_cloud_credentials()
        os.environ["LLM_PROVIDER"] = session.provider
        if session.provider == "gemini":
            os.environ["GEMINI_API_KEY"] = session.api_key
        else:
            os.environ["DASHSCOPE_API_KEY_SG"] = session.api_key
        return None

    @staticmethod
    def clear_cloud_credentials() -> None:
        """Remove every bundled cloud credential before a non-root/guest session."""
        for name in _CLOUD_KEY_NAMES:
            os.environ.pop(name, None)

    def logout(self, username: str) -> None:
        self.clear_cloud_credentials()
        if username and username != "guest_user":
            self.store.delete_account(username)
