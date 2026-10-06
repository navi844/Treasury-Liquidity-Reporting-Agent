"""
tools.py — the two tools Nirikshan's agent can call.

A tool is a Python function, described to the model with a name, a plain
description, and a JSON schema for its arguments. The model never runs
the function — it only ever asks for it to be run, by name, with
arguments. app.py's agent loop is what actually calls the function and
feeds the result back.
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

from context import pack_context

load_dotenv()

HERE = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "lloyds_nirikshan"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. See README.md, Step 2.")
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
        if _collection.count() == 0:
            raise RuntimeError(
                "The vector store is empty. Run 'python build_vectorstore.py' first. See README.md, Step 4."
            )
    return _collection


def lookup_policy(query: str) -> str:
    """Retrieval tool: search the Lloyds Financial knowledge base for the passage most relevant to `query`."""
    collection = _get_collection()
    result = collection.query(query_texts=[query], n_results=4)
    hits = [
        {"text": doc, "source": meta["source"], "unit_id": meta.get("unit_id", "")}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]
    return pack_context(hits)


def calculate_monthly_repayment(principal: float, annual_rate_percent: float, term_years: int) -> str:
    """Calculator tool: standard repayment-mortgage monthly payment formula."""
    principal = float(principal)
    annual_rate_percent = float(annual_rate_percent)
    term_years = int(term_years)

    if principal <= 0 or term_years <= 0:
        return "Error: principal and term_years must both be greater than zero."
    if annual_rate_percent < 0:
        return "Error: annual_rate_percent cannot be negative."

    monthly_rate = (annual_rate_percent / 100) / 12
    n_payments = term_years * 12
    if monthly_rate == 0:
        payment = principal / n_payments
    else:
        payment = (
            principal * monthly_rate * (1 + monthly_rate) ** n_payments
            / ((1 + monthly_rate) ** n_payments - 1)
        )
    total_paid = payment * n_payments
    return (
        f"Estimated monthly repayment: GBP {payment:,.2f}\n"
        f"Over {term_years} years ({n_payments} payments) at {annual_rate_percent}% per year.\n"
        f"Total repaid: GBP {total_paid:,.2f}\n"
        "(Illustrative only — ignores fees, overpayments, and rate changes after any fixed period.)"
    )


TOOL_SPECS = [
    {
        "name": "lookup_policy",
        "description": (
            "Search the Lloyds Financial knowledge base (mortgages, account terms, fraud handling, "
            "complaints, savings products) for the passage most relevant to a question. Always use "
            "this before answering any question about rates, limits, eligibility, or procedure."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The question, or the specific fact to check."}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_monthly_repayment",
        "description": "Calculate an illustrative monthly repayment for a loan, given the amount, rate, and term.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount in GBP."},
                "annual_rate_percent": {"type": "number", "description": "Annual interest rate, e.g. 5.25."},
                "term_years": {"type": "integer", "description": "Loan term in whole years."},
            },
            "required": ["principal", "annual_rate_percent", "term_years"],
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_policy": lookup_policy,
    "calculate_monthly_repayment": calculate_monthly_repayment,
}


if __name__ == "__main__":
    print(calculate_monthly_repayment(250_000, 5.25, 25))
