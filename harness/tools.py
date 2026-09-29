"""Hand-built tool registry: no framework, just a dict + JSON schemas.

Register a tool with @tool(...), giving it the exact JSON schema the model
sees. The agent loop turns REGISTRY into the `tools` list it hands the LLM,
and dispatches by name when the model calls one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

REGISTRY: dict[str, "Tool"] = {}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(name: str, description: str, parameters: dict[str, Any]):
    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = Tool(name=name, description=description, parameters=parameters, func=func)
        return func

    return wrap


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in REGISTRY.values()]


def call(name: str, arguments: dict[str, Any]) -> str:
    if name not in REGISTRY:
        return f"error: no such tool '{name}'"
    try:
        result = REGISTRY[name].func(**arguments)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not raised
        return f"error running {name}: {exc}"
    return str(result)


# --- Day 1 demo tools, just enough to prove the loop end-to-end ------------


@tool(
    "calculator",
    "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    allowed = set("0123456789.+-*/() ")
    if not set(expression) <= allowed:
        return "error: expression contains disallowed characters"
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - char-whitelisted above


@tool(
    "current_time",
    "Get the current date and time.",
    {"type": "object", "properties": {}},
)
def current_time() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


# --- Day 3: RAG over the arXiv corpus --------------------------------------

_retriever = None  # lazy singleton: don't touch LanceDB/BM25 until first use


@tool(
    "search_papers",
    "Search the local corpus of arXiv cs.AI/cs.LG/cs.CL papers. Returns the "
    "top matching papers with id, title, and abstract so you can cite them.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query or research question"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)"},
        },
        "required": ["query"],
    },
)
def search_papers(query: str, top_k: int = 5) -> str:
    global _retriever
    if _retriever is None:
        from rag.retriever import HybridRetriever

        _retriever = HybridRetriever()

    hits = _retriever.search(query, top_k=top_k)
    if not hits:
        return "no matching papers found"

    lines = []
    for h in hits:
        lines.append(f"[{h.id}] {h.title}\n  categories: {h.categories}\n  abstract: {h.abstract[:400]}...")
    return "\n\n".join(lines)
