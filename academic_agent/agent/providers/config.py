"""Gemini、Qwen（新加坡/北京）与 Ollama 的模型配置选择。"""

from __future__ import annotations

import os
from typing import Any


LIGHTWEIGHT_OLLAMA_MODEL = "qwen3:0.6b"
QWEN_SG_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_BJ_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _qwen_api_key(region: str) -> str | None:
    """读取指定地域的 Qwen/DashScope API Key。"""
    if region == "bj":
        return (
            os.getenv("DASHSCOPE_API_KEY_BJ")
            or os.getenv("QWEN_API_KEY_BJ")
            or os.getenv("QWEN_BEIJING_API_KEY")
        )
    return (
        os.getenv("ALIYUN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY_SG")
        or os.getenv("QWEN_API_KEY_SG")
    )


def _qwen_config(api_key: str, region: str, model_name: str) -> dict[str, Any]:
    """构建 Qwen OpenAI-compatible 配置。"""
    is_beijing = region == "bj"
    configured_model = os.getenv("QWEN_MODEL", "qwen-plus")
    return {
        "model": model_name if model_name.startswith("qwen") else configured_model,
        "provider": "qwen-beijing" if is_beijing else "qwen",
        "model_server": QWEN_BJ_BASE_URL if is_beijing else QWEN_SG_BASE_URL,
        "api_key": api_key,
        "generate_cfg": {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 2048,
        },
    }


def ollama_base_url() -> str:
    """返回可连接的 Ollama OpenAI-compatible 地址。"""
    host = os.getenv("OLLAMA_HOST", "localhost").strip()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"http://{host}:{os.getenv('OLLAMA_PORT', '11434').strip()}/v1"


def get_llm_config_priority(
    model_name: str = "gemini-3.6-flash",
    provider: str | None = None,
) -> dict[str, Any]:
    """按显式选择或 Gemini → Qwen 新加坡/北京 → Ollama 顺序创建模型配置。"""
    requested = (provider or os.getenv("LLM_PROVIDER", "auto")).strip().lower()
    if requested in {"local", "ollama"}:
        model = (
            model_name
            if model_name.startswith(("qwen3:", "qwen3.5:"))
            else os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
        )
        return {
            "model": model,
            "provider": "ollama",
            "model_server": ollama_base_url(),
            "api_key": "EMPTY",
            "generate_cfg": {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 2048,
                "extra_body": {"think": False},
            },
        }

    qwen_key = _qwen_api_key("sg")
    qwen_beijing_key = _qwen_api_key("bj")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if requested == "gemini" and not gemini_key:
        raise ValueError("未配置 GOOGLE_API_KEY 或 GEMINI_API_KEY，无法切换到 Gemini")
    if requested in {"qwen", "aliyun", "dashscope"} and not qwen_key:
        raise ValueError(
            "未配置 QWEN_API_KEY_SG、DASHSCOPE_API_KEY_SG 或 ALIYUN_API_KEY，无法切换到 Qwen"
        )
    beijing_requested = requested in {
        "qwen-beijing",
        "qwen_bj",
        "qwen-beijing-cn",
        "beijing",
    }
    if beijing_requested and not qwen_beijing_key:
        raise ValueError(
            "未配置 DASHSCOPE_API_KEY_BJ、QWEN_API_KEY_BJ 或 QWEN_BEIJING_API_KEY，"
            "无法切换到 Qwen 北京"
        )
    if requested in {"qwen", "aliyun", "dashscope"}:
        gemini_key = None
    if beijing_requested:
        return _qwen_config(qwen_beijing_key, "bj", model_name)

    if gemini_key:
        model = (
            model_name
            if model_name.startswith("gemini")
            else os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        )
        return {
            "model": model,
            "provider": "gemini",
            "model_type": "gemini_oai",
            "model_server": os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "api_key": gemini_key,
            "generate_cfg": {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 2048,
            },
        }
    if qwen_key:
        return _qwen_config(qwen_key, "sg", model_name)
    if qwen_beijing_key:
        return _qwen_config(qwen_beijing_key, "bj", model_name)
    return get_llm_config_priority(os.getenv("OLLAMA_MODEL", "qwen3.5:2b"), "ollama")


def get_qwen_fallback_config() -> dict[str, Any] | None:
    """返回 Gemini 请求失败后的 Qwen 配置，优先新加坡，随后北京。"""
    region = "sg" if _qwen_api_key("sg") else "bj"
    api_key = _qwen_api_key(region)
    if not api_key:
        return None
    return _qwen_config(api_key, region, os.getenv("QWEN_MODEL", "qwen-plus"))


def get_lightweight_text_config() -> dict[str, Any]:
    """供意图、响应和行为识别使用的小模型配置。"""
    config = get_llm_config_priority(LIGHTWEIGHT_OLLAMA_MODEL, "ollama")
    config["generate_cfg"].update({"temperature": 0.2, "top_p": 0.8, "max_tokens": 512})
    return config
