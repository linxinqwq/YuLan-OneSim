from __future__ import annotations

import os
from typing import Any


def get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("请安装 openai：pip install openai") from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY；离线请设 ORCHESTRATOR_NO_LLM=1")
    kwargs: dict = {"api_key": api_key}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**kwargs)


def chat_json(model: str, system: str, user: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError("LLM 返回空内容")
    return text


def chat_text(model: str, system: str, user: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError("LLM 返回空内容")
    return text


def default_model() -> str:
    return os.environ.get("ORCHESTRATOR_MODEL", "gpt-4o")


def no_llm_mode() -> bool:
    return os.environ.get("ORCHESTRATOR_NO_LLM", "").lower() in ("1", "true", "yes")
