"""
inspect_vectorstore.py — look at what ./chroma_store actually holds,
two ways: through Chroma's own API, and then directly through the
underlying chroma.sqlite3 file, to show that "a vector database" is, at
the end of the day, a real file on disk with real rows in it — not a
mysterious black box.

RUN IT:
    python inspect_vectorstore.py
"""

from __future__ import annotations

import os
import sqlite3

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "lloyds_nirikshan"


def via_chroma_api() -> None:
    print("=" * 70)
    print("PART A — via the Chroma API")
    print("=" * 70)
    api_key = os.getenv("OPENAI_API_KEY")
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    print(f"Collection: {collection.name}")
    print(f"Chunk count: {collection.count()}")

    sample = collection.peek(limit=3)
    for doc, meta in zip(sample["documents"], sample["metadatas"]):
        print(f"\n  source   : {meta.get('source')}")
        print(f"  unit_id  : {meta.get('unit_id')}")
        print(f"  text     : {doc[:90]!r}...")


def via_sqlite_directly() -> None:
    print("\n" + "=" * 70)
    print("PART B — directly via chroma.sqlite3")
    print("=" * 70)
    db_path = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    if not os.path.exists(db_path):
        print(f"No {db_path} found — run build_vectorstore.py first.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Tables inside chroma.sqlite3: {tables}")
    conn.close()
    print(
        "\nThe embeddings themselves live in Chroma's own binary index files "
        "alongside this database, not as plain floats in a SQL column — "
        "sqlite3 here mainly holds the metadata and bookkeeping. The point "
        "isn't to parse this by hand; it's to see that 'the vector store' "
        "is ordinary files on disk, not magic."
    )


if __name__ == "__main__":
    via_chroma_api()
    via_sqlite_directly()
