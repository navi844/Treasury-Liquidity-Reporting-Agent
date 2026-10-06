"""
pricing.py — manual cost calculation.

The API response tells you exactly how many tokens you used. It never
tells you what that cost. There is no "cost_usd" field anywhere in a
response from any provider. If you want to know what a call cost, you
multiply the token counts yourself against a price list you keep up to
date by hand — which is what this file is.

Prices are per 1,000,000 tokens, in USD, as published at the time this
lab was written. CHECK CURRENT PRICING before relying on this beyond the
lab: https://openai.com/api/pricing/
"""

PRICES_PER_MILLION_TOKENS_USD = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def estimate_cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float | None:
    """Return estimated USD cost, or None if `model` isn't in the table (a deliberate signal, not a silent 0.00)."""
    rates = PRICES_PER_MILLION_TOKENS_USD.get(model)
    if rates is None:
        return None
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


if __name__ == "__main__":
    cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1200, output_tokens=340)
    print(f"Example: 1,200 input + 340 output tokens on gpt-4o-mini \u2248 ${cost:.6f}")
