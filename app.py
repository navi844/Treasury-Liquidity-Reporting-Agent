"""
app.py — Treasury & Liquidity Reporting Agent backend skeleton.

This service keeps the same RAG + tool-calling pattern as the base project,
but is tailored to treasury and liquidity operations such as:

- liquidity position queries
- funding and covenant checks
- daily cash forecasting
- treasury policy lookup
- scenario and concentration analysis

RUN IT:
    uvicorn app:app --reload --port 8000

Then visit http://localhost:8000/docs to interact with the API.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from anatomy import NEVER_RETURNED, inspect_call
from llm_provider import chat_raw
from pricing import estimate_cost_usd
from tools import TOOL_FUNCTIONS, TOOL_SPECS
from tracer import Trace

load_dotenv()

app = FastAPI(
    title="Treasury & Liquidity Reporting Agent",
    description="Agentic treasury assistant for liquidity risk queries, policy lookup, and treasury analytics.",
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = (
    "You are a Treasury & Liquidity Reporting Agent for a financial institution. "
    "Use search_report_library or policy lookup tools for any question about treasury policy, "
    "liquidity thresholds, funding constraints, covenant rules, or market assumptions. "
    "Use analytical tools for structured cash and liquidity calculations. Keep answers concise, "
    "practical, and grounded in the available treasury documents and calculations. "
    "If a requested fact is not available, say so clearly and suggest the missing data needed."
)


def _run_traced_agent_turn(question: str) -> dict:
    """Shared by /ask: runs the loop, returns the answer plus a full trace."""
    trace = Trace()
    messages = [{"role": "user", "content": question}]
    answer = None

    for _ in range(MAX_AGENT_STEPS):
        turn = chat_raw(messages, tools=TOOL_SPECS, system=SYSTEM_PROMPT)
        usage = turn.raw.usage
        cost = estimate_cost_usd(MODEL, usage.prompt_tokens, usage.completion_tokens) or 0.0
        trace.record(
            kind="model_call",
            name=MODEL,
            duration_ms=turn.duration_ms,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
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
        answer = "(Stopped after repeated tool calls with no final answer — a runaway-loop pattern.)"

    tools_used = [s.name for s in trace.steps if s.kind == "tool_call"]
    return {"question": question, "answer": answer, "tools_used": tools_used, "trace": trace.as_dict()}


@app.get("/ask")
def ask(q: str = Query(..., description="Treasury or liquidity question")) -> dict:
    return _run_traced_agent_turn(q)


@app.get("/inspect")
def inspect(
    q: str = Query(..., description="Plain question sent without tool use"),
    max_tokens: int | None = Query(None, description="Optional small max_tokens to force a truncated response"),
) -> dict:
    a = inspect_call(q, max_tokens=max_tokens)
    return {
        "api_returned": {
            "response_id": a.response_id,
            "model": a.model,
            "created": a.created,
            "finish_reason": a.finish_reason,
            "content": a.content,
            "prompt_tokens": a.prompt_tokens,
            "completion_tokens": a.completion_tokens,
            "total_tokens": a.total_tokens,
        },
        "we_measured_ourselves": {
            "latency_ms": round(a.latency_ms, 1),
            "estimated_cost_usd": a.estimated_cost_usd,
        },
        "never_returned_by_any_provider": NEVER_RETURNED,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "Treasury & Liquidity Reporting Agent",
        "endpoints": [
            "/ask?q=...",
            "/inspect?q=...&max_tokens=...",
            "/docs",
        ],
        "capabilities": [
            "liquidity position summaries",
            "liquidity policy lookup",
            "cash buffer analysis",
            "treasury reporting assistance",
        ],
    }
