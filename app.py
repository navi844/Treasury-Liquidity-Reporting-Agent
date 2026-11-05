"""
app.py — Treasury & Liquidity Reporting Agent backend.

This service supports the end-of-day regulatory liquidity reporting workflow:
- retrieve treasury policy / liquidity guidance from internal documents
- run treasury calculations used in LCR-style reporting
- assemble a concise, auditable answer for analyst review

RUN IT:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from llm_provider import chat_raw
from tools import TOOL_FUNCTIONS, TOOL_SPECS
from tracer import Trace

load_dotenv()

app = FastAPI(
    title="Treasury & Liquidity Reporting Agent",
    description="Agent for regulatory liquidity reporting, policy lookup, and treasury calculations.",
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = (
    "You are a Treasury & Liquidity Reporting Agent helping analysts prepare end-of-day regulatory liquidity reports. "
    "Use treasury policy and reporting guidance to answer questions about liquidity metrics, regulatory thresholds, cash positions, "
    "and exceptions. Use tools when needed for calculations or document retrieval. When facts are missing, say so clearly and "
    "flag what additional data is required. Keep responses concise, business-relevant, and suitable for analyst review."
)


def _run_traced_agent_turn(question: str) -> dict:
    """Shared by /ask: runs the loop, returns the answer plus trace metadata."""
    trace = Trace()
    messages = [{"role": "user", "content": question}]
    answer = None

    for _ in range(MAX_AGENT_STEPS):
        turn = chat_raw(messages, tools=TOOL_SPECS, system=SYSTEM_PROMPT)
        usage = turn.raw.usage
        trace.record(
            kind="model_call",
            name=MODEL,
            duration_ms=turn.duration_ms,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=0.0,
        )

        if not turn.tool_calls:
            answer = turn.text
            messages.append({"role": "assistant", "content": turn.text})
            break

        messages.append({
            "role": "assistant",
            "content": turn.text,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls],
        })
        for tc in turn.tool_calls:
            fn = TOOL_FUNCTIONS.get(tc.name)
            t0 = time.perf_counter()
            try:
                result = fn(**tc.arguments) if fn else f"Error: no such tool '{tc.name}'."
            except Exception as exc:
                result = f"Error running {tc.name}: {exc}"
            tool_ms = (time.perf_counter() - t0) * 1000
            trace.record(kind="tool_call", name=tc.name, duration_ms=tool_ms)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})

    if answer is None:
        answer = "(Stopped after repeated tool calls without a final answer.)"

    tools_used = [s.name for s in trace.steps if s.kind == "tool_call"]
    return {"question": question, "answer": answer, "tools_used": tools_used, "trace": trace.as_dict()}


@app.get("/ask")
def ask(q: str = Query(..., description="Treasury or liquidity reporting question")) -> dict:
    return _run_traced_agent_turn(q)


@app.get("/")
def root() -> dict:
    return {
        "service": "Treasury & Liquidity Reporting Agent",
        "description": "End-of-day regulatory liquidity reporting assistant for treasury teams.",
        "endpoints": [
            "/ask?q=...",
            "/docs",
        ],
        "use_cases": [
            "Liquidity coverage reporting",
            "Cash position reconciliation",
            "Treasury policy lookup",
            "Exception flagging for analyst review",
        ],
    }
