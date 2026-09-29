# Athena — a from-scratch agentic research assistant

Autonomous agent that answers multi-hop research questions over a large local
corpus (arXiv cs.AI / cs.LG papers), with hybrid RAG, tiered memory, a
hand-built tool-calling harness, and a live web UI showing every reasoning
step. No agent framework (no LangChain/LangGraph/CrewAI) — every mechanic is
built and understood by hand.

Runs 100% locally against **qwen3:8b** via Ollama for dev/testing (fits fully
in an 8 GB GPU, strong tool-calling + thinking mode). The LLM is accessed
through a small provider interface, so production deployment is a config
swap to any OpenAI-compatible endpoint (hosted Claude/GPT, or the same qwen
weights served by vLLM on a rented GPU) — no rewrite.

## Why this project

Built to demonstrate, in one coherent system rather than eight demos:
harness engineering, hybrid RAG, context engineering, token optimization,
tiered memory, system design, and a real web UI.

## Architecture

```
                       ┌─────────────────────┐
   arXiv dump  ──────▶ │ ingestion/ (offline) │ ──▶ chunks + embeddings ──▶ data/ (LanceDB + BM25 index)
                       └─────────────────────┘
                                                          │
                                                          ▼
 ┌────────────┐   plan/act/observe   ┌───────────────────────────┐   dense+sparse+rerank   ┌────────────┐
 │  harness/  │ ────────────────────▶│   agent loop (ReAct +      │────────────────────────▶│   rag/     │
 │ (from      │◀──────────────────── │   reflection, tool router) │◀─────────────────────────│ retriever  │
 │  scratch)  │    tool results       └───────────────────────────┘      citations           └────────────┘
 └────────────┘                                  │        │
       │ chat/completions (native API)           ▼        ▼
       ▼                                    memory/   context engine
 ┌────────────┐                             (working/   (assembly,
 │  Ollama     │  dev: qwen3:8b              episodic/   compaction,
 │  (local) or │  prod: any OpenAI-compat    semantic)   token budget)
 │  hosted API │  endpoint (config swap)
 └────────────┘
                                                          │
                                                          ▼
                                              api/ (FastAPI, SSE) ──▶ web/ (chat + live trace + citations)
```

## Model choices

| Role | Model | Why |
|---|---|---|
| Reasoning / tool-calling | `qwen3:8b` (Ollama, local) | Fits fully in 8 GB VRAM, strong function-calling, has a thinking mode for harder multi-hop questions |
| Embeddings | `nomic-embed-text` (Ollama, local) | Small (275 MB), fast on CPU or GPU, good enough retrieval quality, leaves VRAM for the reasoning model |
| Reranker | cross-encoder (CPU, sentence-transformers) | Small enough to run on CPU so it never competes with qwen3 for GPU memory |
| Production swap | any OpenAI-compatible endpoint | Same harness code — only `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` change |

Hardware assumed for dev: RTX 5060 Ti (8 GB VRAM), 16 GB RAM. Kept
intentionally light (embedded vector store, CPU reranker, no external DB
server) so testing isn't hardware-locked beyond "one consumer GPU."

## Repo layout

- `harness/` — agent loop, tool-call protocol, LLM client interface (from scratch)
- `rag/` — hybrid retriever: dense (LanceDB) + sparse (BM25) + fusion + reranker + query rewriting
- `ingestion/` — offline pipeline: download → clean → chunk → embed → index
- `memory/` — working / episodic / semantic memory stores
- `api/` — FastAPI service, SSE streaming, ties harness+rag+memory together
- `web/` — chat UI with live agent-trace panel, citations, memory browser
- `evals/` — golden Q&A set + automated grading harness
- `data/` — local vector index + BM25 index (gitignored)

## Roadmap (7–10 days)

1. **Harness core** — agent loop against local qwen3:8b, hand-built tool-call
   parser/dispatcher, ReAct plan→act→observe, step budget, structured step logging
2. **Ingestion + RAG v1** — pull arXiv cs.AI/cs.LG subset, chunk, embed, dense retrieval + citations tool
3. **RAG v2 (best-in-class)** — add BM25 + reciprocal-rank fusion + cross-encoder reranker + query rewriting for multi-hop questions
4. **Context engineering + token optimization** — structured context blocks (system/tools/memory/retrieval), relevance+recency scoring, compaction/summarization on overflow, prompt caching, live token accounting, model routing (light vs. thinking mode)
5. **Memory** — working memory (buffer), episodic memory (past sessions, vector-searchable), semantic memory (durable facts), wired into context assembly
6. **System design + evals** — clean service boundaries, per-step tracing, golden Q&A eval set with automated grading, latency/cost tracking
7. **Web UI** — streaming chat, live trace panel (steps/tool calls as they happen), citations panel, memory browser
8. **Deployment** — Dockerize, document the local→hosted model swap, deploy API+UI
9–10. **Polish / buffer** — architecture write-up with real numbers (retrieval precision, latency, eval score, token cost per query)

## Status

- **Harness core** — done. ReAct-style loop, hand-built tool dispatcher, provider-agnostic LLM client (verified against qwen3:8b, ~19s round-trip on GPU).
- **Ingestion** — done. 24,669 unique cs.AI/cs.LG/cs.CL papers pulled directly from the arXiv API (no Kaggle account needed).
- **RAG (hybrid)** — done. Dense (LanceDB) + sparse (BM25) + reciprocal rank fusion + cross-encoder reranking, embedded via nomic-embed-text. `search_papers` tool wired into the agent and verified end-to-end: real question → retrieval → correctly cited answer.
- **API + web UI** — done (v1). FastAPI streams every agent step over SSE; chat UI with a live trace panel at `http://localhost:8000`.
- **Next**: context engineering + token optimization, tiered memory, evals, deployment docs.

Notable gotchas hit and fixed along the way (see code comments): this machine
runs two Ollama instances and only one has GPU access; arXiv's edge WAF 406s
on boolean `OR` search queries; bulk embedding should run against the GPU
instance (idle during ingestion) rather than the CPU one used at chat time.
