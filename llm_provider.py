"""
llm_provider.py — the one function app.py calls to talk to the model
during the agent loop. Kept separate from anatomy.py on purpose:
anatomy.py is for a single, tool-free "open the envelope" call; this file
is for the tool-calling agent loop, and additionally times itself so
tracer.py never needs its own stopwatch around the SDK call.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AgentTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    duration_ms: float = 0.0
    raw: object = None  # the original SDK response — tracer.py reads .usage off this


def chat_raw(messages: list[dict], tools: list[dict] | None = None, system: str | None = None) -> AgentTurn:
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. See README.md, Step 2.")

    client = openai.OpenAI()

    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for m in messages:
        if m["role"] == "tool":
            openai_messages.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            openai_messages.append({
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in m["tool_calls"]
                ],
            })
        else:
            openai_messages.append({"role": m["role"], "content": m["content"]})

    kwargs = {}
    if tools:
        kwargs["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools
        ]

    start = time.perf_counter()
    response = client.chat.completions.create(model=MODEL, messages=openai_messages, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000

    choice = response.choices[0]
    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments)))

    return AgentTurn(
        text=choice.message.content or "",
        tool_calls=tool_calls,
        stop_reason=choice.finish_reason,
        duration_ms=duration_ms,
        raw=response,
    )
