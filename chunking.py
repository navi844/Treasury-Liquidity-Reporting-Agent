"""
chunking.py — turn redacted RawUnits into embedding-sized chunks.

One strategy here (overlap), not five — Day 1 is about seeing the whole
pipeline work end to end, not comparing chunking strategies against each
other. A unit (a PDF page, a DOCX section, an HTML FAQ answer, a CSV row,
a JSON record) that's already short enough is kept whole; a long one is
split with a small overlap so a sentence that happens to sit on a chunk
boundary doesn't lose its context on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

from loaders import RawUnit


@dataclass
class Chunk:
    source: str
    unit_id: str
    chunk_index: int
    text: str


def chunk_units(units: list[RawUnit], max_chars: int = 800, overlap_chars: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units:
        text = unit.text
        if len(text) <= max_chars:
            chunks.append(Chunk(source=unit.source, unit_id=unit.unit_id, chunk_index=0, text=text))
            continue

        start = 0
        idx = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(Chunk(source=unit.source, unit_id=unit.unit_id, chunk_index=idx, text=text[start:end]))
            idx += 1
            if end == len(text):
                break
            start = end - overlap_chars  # step back so chunks overlap, not just abut
    return chunks


if __name__ == "__main__":
    from loaders import RawUnit as _RU
    sample = [_RU(source="demo.txt", unit_id="section 1", text="A. " * 500)]
    out = chunk_units(sample, max_chars=200, overlap_chars=30)
    print(f"{len(sample[0].text)} chars -> {len(out)} chunks")
    for c in out[:3]:
        print(f"  chunk {c.chunk_index}: {len(c.text)} chars")
