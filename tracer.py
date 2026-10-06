"""
tracer.py — watch a multi-step agent loop from the outside, one step at a
time: a model call, a tool call, a model call, maybe another tool call,
then a final answer. A single visible customer answer can be the result
of several of these chained together. This file is what makes that chain
visible and measurable instead of invisible and assumed.

Every step is logged as either a model_call (with tokens and cost) or a
tool_call (with neither — a tool costs time, not tokens). Keeping the two
kinds separate matters: mixing them into one number would hide which half
of a slow request is actually slow.

This becomes `trace` in app.py's /ask response, and the same idea becomes
a real OpenTelemetry span later in the programme (Module 2) — this is the
hand-rolled version of exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepRecord:
    index: int
    kind: str              # "model_call" or "tool_call"
    name: str              # model name, or tool name
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Trace:
    steps: list[StepRecord] = field(default_factory=list)

    def record(self, **kwargs) -> StepRecord:
        s = StepRecord(index=len(self.steps) + 1, **kwargs)
        self.steps.append(s)
        return s

    def as_dict(self) -> dict:
        """The shape served back in app.py's JSON response."""
        model_steps = [s for s in self.steps if s.kind == "model_call"]
        tool_steps = [s for s in self.steps if s.kind == "tool_call"]
        total_time = sum(s.duration_ms for s in self.steps)
        model_time = sum(s.duration_ms for s in model_steps)
        tool_time = sum(s.duration_ms for s in tool_steps)
        return {
            "steps": [
                {
                    "index": s.index, "kind": s.kind, "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    "input_tokens": s.input_tokens, "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for s in self.steps
            ],
            "summary": {
                "total_steps": len(self.steps),
                "model_calls": len(model_steps),
                "tool_calls": len(tool_steps),
                "total_latency_ms": round(total_time, 1),
                "model_time_ms": round(model_time, 1),
                "tool_time_ms": round(tool_time, 1),
                "model_time_pct": round(100 * model_time / total_time, 1) if total_time else None,
                "tool_time_pct": round(100 * tool_time / total_time, 1) if total_time else None,
                "total_tokens_in": sum(s.input_tokens for s in model_steps),
                "total_tokens_out": sum(s.output_tokens for s in model_steps),
                "total_cost_usd": round(sum(s.cost_usd for s in model_steps), 6),
            },
        }

    def print_table(self) -> None:
        print(f"\n{'#':<3} {'type':<11} {'name':<28} {'time (ms)':>10} {'tokens in/out':>15} {'cost (USD)':>12}")
        print("-" * 85)
        for s in self.steps:
            tok = f"{s.input_tokens}/{s.output_tokens}" if s.kind == "model_call" else "-"
            cost = f"${s.cost_usd:.6f}" if s.cost_usd else "-"
            print(f"{s.index:<3} {s.kind:<11} {s.name:<28} {s.duration_ms:>10.0f} {tok:>15} {cost:>12}")
