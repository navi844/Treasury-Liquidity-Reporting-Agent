"""
pii.py — redact personal data before it ever reaches a model call or gets
stored in the vector index.

This is deliberately simple, regex-based redaction, not a machine-learning
PII detector — good enough to demonstrate the PRINCIPLE (redact before
embedding, redact before sending, never let raw PII flow through a system
you don't fully control) without needing an extra model dependency for a
Day 1 lab. A production system would likely use a dedicated PII detection
library or service; the pattern of WHERE redaction has to happen is the
same either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
UK_PHONE_RE = re.compile(r"\b0\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}\b")
UK_SORT_CODE_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")


@dataclass
class RedactionResult:
    text: str
    redacted_count: int = 0
    redacted_types: list[str] = field(default_factory=list)


def redact(text: str) -> RedactionResult:
    """
    Replace emails, UK phone numbers, and UK sort codes with a labelled
    placeholder, and report what was found. The placeholder names the
    TYPE of data removed (e.g. [REDACTED_EMAIL]) so a reader can still
    tell that a contact detail belonged there, without seeing it.
    """
    types: list[str] = []
    count = 0

    def _sub(pattern, label, s):
        nonlocal count
        def repl(m):
            nonlocal count
            count += 1
            if label not in types:
                types.append(label)
            return f"[REDACTED_{label}]"
        return pattern.sub(repl, s)

    text = _sub(EMAIL_RE, "EMAIL", text)
    text = _sub(UK_SORT_CODE_RE, "SORT_CODE", text)  # before phone: dashes could otherwise be mis-parsed
    text = _sub(UK_PHONE_RE, "PHONE", text)

    return RedactionResult(text=text, redacted_count=count, redacted_types=types)


if __name__ == "__main__":
    sample = (
        "Contact Priya Anand at priya.anand@lloydsfinancial-training.example "
        "or on 07700 900123, sort code 40-02-17."
    )
    result = redact(sample)
    print("Before:", sample)
    print("After: ", result.text)
    print("Found: ", result.redacted_count, result.redacted_types)
