"""
context.py — turn a list of retrieved hits into the one block of text
that actually gets pasted into the prompt.

Two jobs, both easy to skip and both costly if you do:
  1. DEDUPE — vector search can return two near-identical chunks (e.g.
     the same clause split across overlapping chunks). Sending both
     wastes tokens and can make the model repeat itself.
  2. PACK TO A BUDGET — stuffing in every retrieved chunk regardless of
     size is how a context window quietly fills up with low-value text,
     pushing out the system prompt or crowding what the model can "see."
"""

from __future__ import annotations


def dedupe(hits: list[dict], similarity_chars: int = 60) -> list[dict]:
    """
    Drop a hit if its first `similarity_chars` characters exactly match a
    hit already kept — a cheap, good-enough test for "this is basically
    the same chunk again," without needing a second embedding comparison.
    """
    seen = set()
    kept = []
    for h in hits:
        key = h["text"][:similarity_chars]
        if key in seen:
            continue
        seen.add(key)
        kept.append(h)
    return kept


def pack_context(hits: list[dict], max_total_chars: int = 2800) -> str:
    """
    Deduplicate, then concatenate hits (each labelled with its source)
    until adding the next one would exceed max_total_chars. Returns the
    packed context string, ready to paste into a prompt.
    """
    kept = dedupe(hits)
    parts: list[str] = []
    total = 0
    for h in kept:
        block = f"[Source: {h['source']} | {h.get('unit_id', '')}]\n{h['text']}"
        if total + len(block) > max_total_chars and parts:
            break  # keep at least one chunk even if it alone exceeds the budget
        parts.append(block)
        total += len(block)
    if not parts:
        return "(No sufficiently relevant document was found for this query.)"
    return "\n\n".join(parts)
