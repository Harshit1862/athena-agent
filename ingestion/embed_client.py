"""Batch embedding client against the CPU-only Ollama instance.

Kept separate from harness/llm_client.py: embeddings are a different concern
(batch, no chat/tool semantics) and deliberately hit port 11434 (CPU) so they
never compete with qwen3:8b for the 8 GB of GPU VRAM.
"""

from __future__ import annotations

import httpx

from harness import config


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        f"{config.EMBED_BASE_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]
