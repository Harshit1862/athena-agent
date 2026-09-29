"""The agent loop itself: plan -> act -> observe -> (repeat) -> answer.

This is the harness-engineering core of the project. No framework — just a
message list, a tool dispatcher, and a step budget.

`run_stream` yields each Step as it happens (thought / tool_call /
tool_result / final) — this is what lets both the CLI and the web UI's live
trace panel show the agent thinking in real time instead of waiting for the
whole multi-step loop to finish. `run` is a thin wrapper for callers that
just want the final answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from harness import config, tools
from harness.llm_client import LLMClient, get_client


@dataclass
class Step:
    kind: str  # "thought" | "tool_call" | "tool_result" | "final"
    content: str
    tool_name: str | None = None


@dataclass
class AgentResult:
    answer: str
    trace: list[Step] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLMClient | None = None, max_steps: int = config.MAX_STEPS):
        self.llm = llm or get_client()
        self.max_steps = max_steps

    def run_stream(
        self, user_message: str, history: list[dict[str, Any]] | None = None
    ) -> Iterator[Step]:
        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": config.SYSTEM_PROMPT}] + (history or [])
        )
        messages.append({"role": "user", "content": user_message})
        tool_schemas = tools.schemas()

        for _ in range(self.max_steps):
            response = self.llm.chat(messages, tools=tool_schemas)

            if response.content:
                yield Step(kind="thought", content=response.content)

            if not response.tool_calls:
                yield Step(kind="final", content=response.content)
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                yield Step(kind="tool_call", content=str(tc.arguments), tool_name=tc.name)
                result = tools.call(tc.name, tc.arguments)
                yield Step(kind="tool_result", content=result, tool_name=tc.name)
                messages.append({"role": "tool", "content": result, "name": tc.name})

        yield Step(kind="final", content="(stopped: max steps reached)")

    def run(self, user_message: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
        trace = list(self.run_stream(user_message, history=history))
        answer = trace[-1].content if trace else ""
        return AgentResult(answer=answer, trace=trace)
