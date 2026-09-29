"""FastAPI service: ties the harness (agent loop) to the web UI.

POST /chat streams the agent's steps over SSE as they happen (thought /
tool_call / tool_result / final) — the web UI's live trace panel consumes
this directly instead of waiting for one big response.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from harness.agent import Agent

WEB_DIR = Path(__file__).parent.parent / "web"

app = FastAPI(title="Athena")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = Agent()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    def event_stream():
        for step in _agent.run_stream(req.message, history=req.history):
            yield {"event": step.kind, "data": json.dumps(asdict(step))}

    return EventSourceResponse(event_stream())


# mounted last so it doesn't shadow /health or /chat
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
