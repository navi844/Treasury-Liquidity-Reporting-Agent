# Treasury & Liquidity Reporting Agent

This repository is a base skeleton for a Treasury & Liquidity Reporting Agent built on the same architecture pattern as a retrieval-augmented financial assistant:

- ingest documents into a vector store
- retrieve the most relevant treasury and liquidity context
- allow the LLM to call tools for calculations and policy lookup
- expose the workflow through a FastAPI application
- trace model calls and tool usage for cost and latency visibility

The project is intentionally structured as a starter foundation for future enhancements such as:

- daily liquidity dashboards
- intraday cash forecasting
- covenant and funding monitoring
- policy search over treasury manuals and procedures
- scenario analysis for stress testing
- report generation for treasury committees

## Architecture Overview

The base project is organized around a simple agent loop:

1. User asks a treasury or liquidity question
2. LLM decides whether it needs document retrieval or calculation tools
3. Tools run and return structured results
4. Final answer is produced by the model
5. Each step is logged for tracing, latency, and cost analysis

## Repository Structure

- `app.py` — FastAPI app with `/ask` and `/inspect` endpoints
- `agent.py` — sample client for calling the service
- `llm_provider.py` — OpenAI wrapper and tool-call handling
- `tools.py` — treasury-specific tools and tool schemas
- `tracer.py` — step logging for model calls and tool calls
- `build_vectorstore.py` — creates the vector store from documents
- `loaders.py` — document ingestion logic by format
- `chunking.py` — chunking strategy for retrieval
- `context.py` — response packaging for tool outputs
- `pii.py` — optional personal-data redaction support
- `pricing.py` — cost estimation helpers
- `anatomy.py` — raw model call inspection utilities
- `documents/` — source treasury documents or reference materials
- `chroma_store/` — local vector database persistence

## Typical Use Cases

This starter is suitable for use cases such as:

- "What is the current liquidity coverage position?"
- "Which treasury policy applies to excess cash placement limits?"
- "What is the projected closing cash balance for the next 7 days?"
- "Which funding sources are most exposed to rollover risk this month?"
- "Summarize the treasury risk position based on latest reporting documents."

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY="your-api-key"
OPENAI_MODEL="gpt-4o"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

### 4. Build the vector store

If you already have treasury documents or reference files in `documents/`, run:

```bash
python build_vectorstore.py
```

This creates the ChromaDB vector store in `chroma_store/`.

### 5. Run the API

```bash
uvicorn app:app --reload --port 8000
```

### 6. Test the service

You can use the built-in client:

```bash
python agent.py
```

Or open:

- `http://localhost:8000/docs`

## Example Questions

Try questions like:

- What is our current liquidity coverage estimate?
- Show the policy guidance for excess cash deployment limits.
- Calculate closing cash balance for opening balance 25M, inflows 8M, and outflows 10M.
- What treasury document is most relevant to a liquidity stress scenario?
- Which reporting rules should we follow for daily liquidity monitoring?

## Suggested Next Enhancements

This is a base skeleton. Recommended next steps:

1. Define actual treasury documents and data sources
2. Create domain-specific tool functions for covenant, funding, and cash positioning
3. Add structured reporting templates for daily and weekly treasury reports
4. Add authentication and audit logging for operational use
5. Integrate with internal data systems and Excel/CSV exports
6. Store results in a retrieval-ready knowledge base for policy and reporting standards

## Notes

This project intentionally provides a working foundational pattern and can be adapted to a full Treasury & Liquidity Reporting Agent once the business requirements, document corpus, tool logic, and reporting workflows are finalized.
