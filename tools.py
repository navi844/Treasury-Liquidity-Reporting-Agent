"""
tools.py — Treasury & Liquidity Reporting Agent tool skeleton.

This file defines the retrieval and calculation utilities the agent can call.
It keeps the same design as the original base project: a list of tool specs
for the model and a function map that the FastAPI loop invokes.
"""

from __future__ import annotations

import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from context import pack_context

load_dotenv()

HERE = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "treasury_liquidity_agent"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. See README.md and add your .env file.")
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
        if _collection.count() == 0:
            raise RuntimeError(
                "The vector store is empty. Run 'python build_vectorstore.py' first to ingest treasury documents."
            )
    return _collection


def lookup_policy(query: str) -> str:
    """Retrieve the most relevant treasury policy or reporting guidance from the document store."""
    collection = _get_collection()
    result = collection.query(query_texts=[query], n_results=4)
    hits = [
        {"text": doc, "source": meta["source"], "unit_id": meta.get("unit_id", "")}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]
    return pack_context(hits)


def calculate_liquidity_coverage(cash: float, inflows: float, outflows: float) -> str:
    """Simple liquidity coverage ratio estimate for a reporting scenario."""
    cash = float(cash)
    inflows = float(inflows)
    outflows = float(outflows)

    if cash < 0 or inflows < 0 or outflows < 0:
        return "Error: cash, inflows, and outflows must be non-negative."
    if outflows == 0:
        return "Error: outflows cannot be zero for a meaningful liquidity coverage calculation."

    coverage_ratio = (cash + inflows) / outflows
    return (
        f"Liquidity coverage estimate: {coverage_ratio:.2f}x\n"
        f"Inputs: cash={cash:,.2f}, inflows={inflows:,.2f}, outflows={outflows:,.2f}\n"
        "This is an illustrative treasury calculation and should be validated against policy and actual balances."
    )


def summarize_cash_position(beginning_balance: float, inflows: float, outflows: float) -> str:
    """Compute end-of-period cash position from treasury movement assumptions."""
    beginning_balance = float(beginning_balance)
    inflows = float(inflows)
    outflows = float(outflows)

    if beginning_balance < 0:
        return "Error: beginning_balance cannot be negative."

    ending_balance = beginning_balance + inflows - outflows
    return (
        f"Ending cash position: {ending_balance:,.2f}\n"
        f"Opening balance: {beginning_balance:,.2f} | inflows: {inflows:,.2f} | outflows: {outflows:,.2f}"
    )


TOOL_SPECS = [
    {
        "name": "lookup_policy",
        "description": (
            "Search the treasury policies, controls, and reporting guidance repository for the most relevant text to answer "
            "a treasury or liquidity question. Use this for liquidity rules, funding constraints, covenant policy, or reporting standards."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The treasury question or policy fact to look up."}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_liquidity_coverage",
        "description": "Estimate liquidity coverage based on available cash, inflows, and outflows.",
        "parameters": {
            "type": "object",
            "properties": {
                "cash": {"type": "number", "description": "Available cash balance."},
                "inflows": {"type": "number", "description": "Expected cash inflows over the period."},
                "outflows": {"type": "number", "description": "Expected cash outflows over the period."},
            },
            "required": ["cash", "inflows", "outflows"],
        },
    },
    {
        "name": "summarize_cash_position",
        "description": "Calculate end-of-period cash position from opening balance and expected cash movements.",
        "parameters": {
            "type": "object",
            "properties": {
                "beginning_balance": {"type": "number", "description": "Opening cash balance."},
                "inflows": {"type": "number", "description": "Cash inflows during the period."},
                "outflows": {"type": "number", "description": "Cash outflows during the period."},
            },
            "required": ["beginning_balance", "inflows", "outflows"],
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_policy": lookup_policy,
    "calculate_liquidity_coverage": calculate_liquidity_coverage,
    "summarize_cash_position": summarize_cash_position,
}


if __name__ == "__main__":
    print(calculate_liquidity_coverage(5_000_000, 2_000_000, 3_500_000))
