"""LLM provider interface.

Two implementations behind one interface so the agent loop never knows or
cares which one it's talking to:

- OllamaClient: native Ollama /api/chat. Used for local dev against qwen3:8b —
  gives full control over GPU placement (num_gpu/num_batch) and Ollama's
  thinking-mode toggle, which the OpenAI-compat shim doesn't expose.
- OpenAICompatClient: standard /v1/chat/completions. Point this at a hosted
  provider (or vLLM serving the same qwen weights on a rented GPU) for
  production — same request/response shape the agent loop already expects.

Switch providers with ATHENA_LLM_PROVIDER=ollama|openai_compat (default ollama).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from harness import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        ...


class OllamaClient(LLMClient):
    def __init__(self, base_url: str = config.LLM_BASE_URL, model: str = config.LLM_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": config.LLM_OPTIONS,
            "think": config.THINK,
        }
        if tools:
            payload["tools"] = tools

        resp = self._http.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})

        tool_calls = [
            ToolCall(
                id=str(i),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", {}),
            )
            for i, tc in enumerate(msg.get("tool_calls", []) or [])
        ]
        return ChatResponse(content=msg.get("content", ""), tool_calls=tool_calls, raw=data)


class OpenAICompatClient(LLMClient):
    """For production: any /v1/chat/completions-compatible endpoint."""

    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        api_key: str = config.LLM_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(
            timeout=120.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools

        resp = self._http.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]

        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=_parse_args(tc["function"].get("arguments", "{}")),
            )
            for tc in msg.get("tool_calls", []) or []
        ]
        return ChatResponse(content=msg.get("content") or "", tool_calls=tool_calls, raw=data)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    import json

    return json.loads(raw)


def get_client() -> LLMClient:
    provider = os.environ.get("ATHENA_LLM_PROVIDER", "ollama")
    if provider == "openai_compat":
        return OpenAICompatClient()
    return OllamaClient()
