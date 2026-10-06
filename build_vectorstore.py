"""
build_vectorstore.py — run once (or re-run after changing documents/).

    documents/*.{pdf,docx,html,csv,json}
        |  loaders.py        -- one loader per format, same output shape
        v
    raw text units
        |  pii.py             -- redact emails/phones/sort codes
        v
    redacted text
        |  chunking.py        -- overlap strategy
        v
    chunks, tagged with source + unit_id + chunk_index
        |  OpenAI embeddings
        v
    ./chroma_store  (ChromaDB vector store, persisted to disk)
"""

from __future__ import annotations

import glob
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from chunking import chunk_units
from loaders import load_document
from pii import redact

load_dotenv()

HERE = os.path.dirname(__file__)
DOCS_DIR = os.path.join(HERE, "documents")
PERSIST_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "lloyds_nirikshan"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    paths = sorted(
        p for p in glob.glob(os.path.join(DOCS_DIR, "*"))
        if os.path.splitext(p)[1].lower() in (".pdf", ".docx", ".html", ".csv", ".json")
    )
    if not paths:
        raise ValueError(f"No documents found in {DOCS_DIR!r}.")

    ids, documents, metadatas = [], [], []
    total_redacted = 0

    for path in paths:
        units = load_document(path)
        redacted_units = []
        for u in units:
            r = redact(u.text)
            total_redacted += r.redacted_count
            u.text = r.text
            redacted_units.append(u)

        chunks = chunk_units(redacted_units)
        for c in chunks:
            ids.append(f"{c.source}::{c.unit_id}::{c.chunk_index}")
            documents.append(c.text)
            metadatas.append({"source": c.source, "unit_id": c.unit_id, "chunk_index": c.chunk_index})

        print(f"  {os.path.basename(path):35s} -> {len(units):2d} unit(s), {len(chunks):2d} chunk(s)")

    print(f"\nTotal: {len(documents)} chunks from {len(paths)} documents "
          f"({total_redacted} PII item(s) redacted during ingestion).")

    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Embedded and stored {collection.count()} chunks in {PERSIST_DIR}")


if __name__ == "__main__":
    main()
