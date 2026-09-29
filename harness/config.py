import os

# --- LLM provider (swap these three for production) -------------------------
# Dev default: local Ollama, native API, full control over GPU placement.
# Prod: point LLM_BASE_URL at any OpenAI-compatible endpoint (hosted Claude/GPT,
# or the same qwen weights served by vLLM) and set LLM_API_KEY. Nothing else
# in the harness needs to change — see harness/llm_client.py.
# NOTE: this machine runs two Ollama instances — the default port 11434 has no
# GPU access (falls back to CPU, ~10x slower); the GPU-accelerated one (also
# used by ../local-agent) listens on 11435. Point at that one.
LLM_BASE_URL = os.environ.get("ATHENA_LLM_BASE_URL", "http://localhost:11435")
LLM_MODEL = os.environ.get("ATHENA_LLM_MODEL", "qwen3:8b")
LLM_API_KEY = os.environ.get("ATHENA_LLM_API_KEY", "")  # unused by local Ollama

# Embeddings run on the plain CPU-only Ollama instance (port 11434) on purpose:
# nomic-embed-text is cheap enough on CPU, and this keeps the 8 GB GPU free
# for qwen3:8b instead of the two competing for VRAM.
EMBED_BASE_URL = os.environ.get("ATHENA_EMBED_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("ATHENA_EMBED_MODEL", "nomic-embed-text")

# Let qwen3 think step by step before acting (slower, smarter).
THINK = os.environ.get("ATHENA_THINK", "1") == "1"
CONTEXT_TOKENS = int(os.environ.get("ATHENA_CTX", "12288"))
# Keep every layer on the 8 GB GPU (Ollama's own estimate is too cautious).
LLM_OPTIONS = {"num_ctx": CONTEXT_TOKENS, "num_gpu": 99, "num_batch": 128}

MAX_STEPS = int(os.environ.get("ATHENA_MAX_STEPS", "15"))

SYSTEM_PROMPT = """You are Athena, a research agent. You answer questions by using \
tools to look things up rather than guessing — call a tool, read its result, then \
decide the next step. Work in small steps. Cite sources when you use retrieved \
information. Stop and answer as soon as you have enough evidence; don't call \
tools you don't need."""
