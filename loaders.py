"""
loaders.py — one loader per document format, all returning the same shape.

Real organisations don't keep their knowledge in one tidy format. Lloyds'
mortgage SOP is a PDF; account terms are a Word document; the customer FAQ
is an HTML page; the complaints register is a CSV export; the product
rate sheet is a JSON feed from another system. A RAG pipeline that only
reads .txt files is a toy. This file is what makes it not a toy: whatever
the source format, every loader below returns the same thing — a list of
RawUnit, each just {source, unit_id, text} — so everything downstream
(pii.py, chunking.py) never needs to know or care which loader produced
a given piece of text.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass


@dataclass
class RawUnit:
    source: str     # the filename this text came from
    unit_id: str    # a human-readable locator within that file (page, row, section...)
    text: str


def load_pdf(path: str) -> list[RawUnit]:
    """One RawUnit per page, using PyMuPDF (imported as `fitz`)."""
    import fitz  # PyMuPDF

    source = path.split("/")[-1]
    units = []
    doc = fitz.open(path)
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                units.append(RawUnit(source=source, unit_id=f"page {page_num}", text=text))
    finally:
        doc.close()
    return units


def load_docx(path: str) -> list[RawUnit]:
    """
    One RawUnit per heading section: every Heading-styled paragraph starts
    a new section, and body paragraphs accumulate under it. This keeps a
    whole policy clause together rather than splitting it mid-sentence.
    """
    import docx

    source = path.split("/")[-1]
    d = docx.Document(path)
    units: list[RawUnit] = []
    current_heading = "Document start"
    current_text: list[str] = []

    def flush():
        text = "\n".join(current_text).strip()
        if text:
            units.append(RawUnit(source=source, unit_id=current_heading, text=text))

    for para in d.paragraphs:
        style = (para.style.name or "")
        text = para.text.strip()
        if not text:
            continue
        if style.startswith("Heading") or style == "Title":
            flush()
            current_heading = text
            current_text = []
        else:
            current_text.append(text)
    flush()
    return units


def load_html(path: str) -> list[RawUnit]:
    """One RawUnit per <h2> section — matches how the FAQ is actually laid out."""
    from bs4 import BeautifulSoup

    source = path.split("/")[-1]
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    units: list[RawUnit] = []
    for h2 in soup.find_all("h2"):
        question = h2.get_text(strip=True)
        parts = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            parts.append(sib.get_text(" ", strip=True))
        text = (question + "\n" + " ".join(parts)).strip()
        if text:
            units.append(RawUnit(source=source, unit_id=question, text=text))
    return units


def load_csv(path: str) -> list[RawUnit]:
    """One RawUnit per row, rendered as 'column: value' lines — keeps it readable and embeddable."""
    source = path.split("/")[-1]
    units: list[RawUnit] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = "\n".join(f"{k}: {v}" for k, v in row.items())
            row_id = row.get("complaint_id") or f"row {i}"
            units.append(RawUnit(source=source, unit_id=str(row_id), text=text))
    return units


def load_json(path: str) -> list[RawUnit]:
    """One RawUnit per top-level record in a {"products": [...]} -style file."""
    source = path.split("/")[-1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    units: list[RawUnit] = []
    records = data.get("products") or data.get("items") or data.get("records") or []
    for i, record in enumerate(records, start=1):
        name = record.get("product_name") or record.get("name") or f"record {i}"
        lines = [f"{k}: {v}" for k, v in record.items()]
        units.append(RawUnit(source=source, unit_id=str(name), text="\n".join(lines)))
    return units


# Dispatch table: file extension -> loader function. build_vectorstore.py
# picks a loader by extension so adding a 6th format is a two-line change
# here, not a rewrite of the ingestion pipeline.
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".csv": load_csv,
    ".json": load_json,
}


def load_document(path: str) -> list[RawUnit]:
    ext = "." + path.rsplit(".", 1)[-1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"No loader registered for {ext!r} ({path}). Add one to LOADERS.")
    return loader(path)
