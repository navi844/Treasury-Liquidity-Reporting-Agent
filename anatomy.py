"""
anatomy.py — the one module that calls the model with nothing hidden.

Everywhere else in this repo, a model call happens inside a bigger
pipeline (retrieval, tool execution, context packing). This file exists
to strip all of that away and ask: what EXACTLY comes back from a single
call, and what, provably, never does? That second question is the whole
point of today's lab — the API is extremely precise about tokens and
almost silent about everything else an operator actually needs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from pricing import estimate_cost_usd

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# These are facts about what the API does NOT return, independent of which
# question you ask it — used by app.py's /inspect endpoint and by the
# README/assignment docs. Kept as data, not prose, so it's testable and
# reusable rather than just a comment someone has to re-type elsewhere.
NEVER_RETURNED = [
    "A confidence or certainty score for the answer",
    "A cost figure, in any currency",
    "A breakdown of time spent 'thinking' vs. writing the visible answer",
    "Any signal that the answer might be wrong, outdated, or a guess",
    "Which part of its training data (if any) the answer came from",
]


@dataclass
class CallAnatomy:
    """Everything we know about one call: what the API told us, and what we had to work out ourselves."""
    response_id: str
    model: str
    created: int
    finish_reason: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float          # measured by us — never in the response
    estimated_cost_usd: float | None  # calculated by us — never in the response


def inspect_call(question: str, max_tokens: int | None = None) -> CallAnatomy:
    """
    Make one plain (no-tools) call and return a full CallAnatomy. This is
    the "open the envelope" function — app.py's /inspect endpoint and
    01-style exercises are both thin wrappers around this.
    """
    import openai

    client = openai.OpenAI()
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": question}]}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    choice = response.choices[0]
    cost = estimate_cost_usd(MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)

    return CallAnatomy(
        response_id=response.id,
        model=response.model,
        created=response.created,
        finish_reason=choice.finish_reason,
        content=choice.message.content or "",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=cost,
    )


if __name__ == "__main__":
    a = inspect_call("In one sentence, what is a mortgage?")
    print(f"id={a.response_id} model={a.model} finish_reason={a.finish_reason}")
    print(f"tokens: {a.prompt_tokens} in / {a.completion_tokens} out / {a.total_tokens} total")
    print(f"latency={a.latency_ms:.0f}ms  cost=${a.estimated_cost_usd:.6f}")
    print(f"content: {a.content!r}")
    print("\nNever returned by any provider:")
    for line in NEVER_RETURNED:
        print(f"  - {line}")
