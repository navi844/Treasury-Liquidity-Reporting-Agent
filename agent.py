"""
agent.py — run this in Terminal 2, while app.py keeps running in Terminal 1.

Sends 5 hardcoded domain questions (one per document format) to /ask,
then two BONUS sections dedicated to today's actual topic: the full
anatomy of a raw model call, and a close look at a traced multi-tool
conversation.

RUN IT:
    python agent.py
"""

from __future__ import annotations

import json

import requests

BASE_URL = "http://localhost:8000"

QUESTIONS = [
    ("What is the maximum loan-to-value for a first-time buyer under the assisted scheme?", "PDF (mortgage SOP)"),
    ("What is the overdraft interest rate on a Premier account?", "DOCX (account terms)"),
    ("who is the head of Fraud identification team?", "HTML (FAQ — also a PII redaction test)"),
    ("Who do I contact if I suspect fraud outside business hours?", "HTML (FAQ — also a PII redaction test)"),
    ("Which complaints are still open?", "CSV (complaints register)"),
    ("Which ISA product has the highest interest rate, and what is the early-access penalty on it?", "JSON (product rates)"),
]


def ask(question: str) -> dict:
    resp = requests.get(f"{BASE_URL}/ask", params={"q": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def inspect(question: str, max_tokens: int | None = None) -> dict:
    params = {"q": question}
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    resp = requests.get(f"{BASE_URL}/inspect", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_domain_questions() -> None:
    print("=" * 70)
    print("5 DOMAIN QUESTIONS — one per document format")
    print("=" * 70)
    for question, tag in QUESTIONS:
        print(f"\n[{tag}]")
        print(f"Q: {question}")
        try:
            result = ask(question)
        except requests.exceptions.ConnectionError:
            print("Could not reach Nirikshan — is app.py running in another terminal?")
            return
        print(f"A: {result['answer']}")
        print(f"   tools used: {result['tools_used']}  |  "
              f"steps: {result['trace']['summary']['total_steps']}  |  "
              f"latency: {result['trace']['summary']['total_latency_ms']} ms  |  "
              f"cost: ${result['trace']['summary']['total_cost_usd']}")


def bonus_anatomy() -> None:
    print("\n" + "=" * 70)
    print("BONUS 1 — the anatomy of a raw model call")
    print("=" * 70)
    question = "In one sentence, what is a mortgage?"

    normal = inspect(question)
    print(f"\nNormal call:")
    print(json.dumps(normal["api_returned"], indent=2))
    print(json.dumps(normal["we_measured_ourselves"], indent=2))

    truncated = inspect(question, max_tokens=10)
    print(f"\nDeliberately truncated call (max_tokens=10):")
    print(f"  finish_reason = {truncated['api_returned']['finish_reason']!r}")
    print(f"  content       = {truncated['api_returned']['content']!r}")
    print(
        "\n  Compare finish_reason between the two calls above — 'stop' vs "
        "'length' is the ONLY field that tells you the second answer was "
        "cut off rather than complete."
    )

    print("\nWhat no provider ever returns, from either call:")
    for line in normal["never_returned_by_any_provider"]:
        print(f"  - {line}")


def bonus_trace() -> None:
    print("\n" + "=" * 70)
    print("BONUS 2 — tracing a multi-step tool-calling conversation")
    print("=" * 70)
    question = (
        "What's the maximum loan-to-value for a first-time buyer, and what would the "
        "monthly repayment be on a GBP 250,000 mortgage at 5.2% over 30 years?"
    )
    print(f"Q: {question}\n")
    result = ask(question)
    print(f"A: {result['answer']}\n")

    steps = result["trace"]["steps"]
    print(f"{'#':<3} {'type':<11} {'name':<28} {'time (ms)':>10} {'tokens in/out':>15} {'cost (USD)':>12}")
    print("-" * 85)
    for s in steps:
        tok = f"{s['input_tokens']}/{s['output_tokens']}" if s["kind"] == "model_call" else "-"
        cost = f"${s['cost_usd']:.6f}" if s["cost_usd"] else "-"
        print(f"{s['index']:<3} {s['kind']:<11} {s['name']:<28} {s['duration_ms']:>10.0f} {tok:>15} {cost:>12}")

    summ = result["trace"]["summary"]
    print(f"\nTotal: {summ['total_steps']} steps ({summ['model_calls']} model call(s), "
          f"{summ['tool_calls']} tool call(s))")
    print(f"  {summ['model_time_pct']}% of wall-clock time in model calls, "
          f"{summ['tool_time_pct']}% in tool calls")
    print(f"  {summ['total_tokens_in']} tokens in / {summ['total_tokens_out']} tokens out, "
          f"${summ['total_cost_usd']} total")


if __name__ == "__main__":
    run_domain_questions()
    bonus_anatomy()
    bonus_trace()
