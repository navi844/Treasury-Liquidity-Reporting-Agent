"""
agent.py — sample client for the Treasury & Liquidity Reporting Agent.

This file is intentionally simple and focuses on real treasury use cases,
not raw model-call inspection or pricing experiments.
"""

from __future__ import annotations

import requests

BASE_URL = "http://localhost:8000"

QUESTIONS = [
    "What is the current liquidity coverage position for the reporting period?",
    "What treasury policy applies to excess cash placement limits?",
    "Calculate the end-of-day cash balance from an opening balance of 25,000,000, inflows of 8,500,000, and outflows of 10,000,000.",
    "Which liquidity exceptions require analyst review before submission?",
    "Summarize the key treasury risks impacting the next reporting cycle.",
]


def ask(question: str) -> dict:
    resp = requests.get(f"{BASE_URL}/ask", params={"q": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_questions() -> None:
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {i}: {question}")
        try:
            result = ask(question)
        except requests.exceptions.ConnectionError:
            print("Could not reach the agent — is app.py running in another terminal?")
            return
        print(f"Answer: {result['answer']}")
        print(f"Tools used: {result['tools_used']}")
        print(f"Trace summary: {result['trace']['summary']}")


if __name__ == "__main__":
    run_questions()
