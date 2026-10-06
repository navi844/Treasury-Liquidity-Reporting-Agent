# Nirikshan — Anatomy of a Model Call, Tool Loops, and Where Time & Cost Go

*A hands-on lab on opening up the black box: what a raw model call actually
returns, what it never does, and how to trace a multi-step tool-calling
conversation so "where did the time and cost go?" has a measured answer —
built on the same ingest → chunk → embed → retrieve → answer pattern as
Disha, now instrumented end to end.*

## python

## 1. What This Is, in Plain Language

Every earlier lab treated a model call as something you send a question
to and get an answer back from. **Nirikshan** stops treating it that way.
It wraps the same kind of RAG + single-agent pipeline you've already
built in a FastAPI service that, on every single request, opens the
envelope: every token count, every timing, every tool call, laid out as
data in the response — not just printed to a terminal and forgotten.

Two things this lab makes unavoidable to notice:

- **The API tells you tokens. It never tells you time or cost.** Those
  are measured and calculated by this repo's own code, in the open, in
  `anatomy.py` and `pricing.py` — not hidden inside a library.
- **A single visible answer can hide several model calls and tool calls
  chained together.** `tracer.py` logs every one of them, separately, so
  "the request was slow" becomes "67% of the time was inside the second
  model call," not a guess.

## 2. The Five Documents, Five Formats

| # | File | Format | Loader used |
|---|------|--------|-------------|
| 1 | `01_mortgage_lending_sop.pdf` | PDF | PyMuPDF |
| 2 | `02_premier_account_terms.docx` | Word | python-docx |
| 3 | `03_customer_faq.html` | HTML | BeautifulSoup |
| 4 | `04_complaints_register.csv` | CSV | built-in `csv` module |
| 5 | `05_savings_product_rates.json` | JSON | built-in `json` module |

All fictional documents written for this training lab — not real Lloyds
policy, rates, or figures. Document 3 (the FAQ) contains a deliberately
planted email address and phone number, specifically so you can watch
PII redaction catch and mask them during ingestion.

## 3. Architecture — the Full Flow

```
STEP 1 -- build_vectorstore.py  (run once, or re-run after editing documents/)

    documents/*.{pdf,docx,html,csv,json}
        |
        |  loaders.py        -- one loader per format, same output shape
        v
    raw text units (one per page / section / FAQ answer / row / record)
        |
        |  pii.py            -- redact emails, phone numbers, sort codes
        v
    redacted text
        |
        |  chunking.py       -- overlap strategy
        v
    chunks, tagged with source + unit_id + chunk_index
        |
        |  OpenAI embeddings
        v
    ./chroma_store  (ChromaDB vector store, persisted to disk)

STEP 2 -- app.py  (FastAPI backend, keep running in Terminal 1)

    GET /ask?q=...
        1. tools.py's lookup_policy / calculate_monthly_repayment, called
           by the model as it needs them
        2. tracer.py       -- every model call AND every tool call timed,
                               tokens and cost captured per model call
        3. return JSON      -- {answer, tools_used, trace: {steps, summary}}

    GET /inspect?q=...&max_tokens=...
        1. anatomy.py        -- ONE plain call, no tools, no retrieval
        2. return JSON        -- every raw field the API returned, PLUS
                                  latency and cost we measured ourselves,
                                  PLUS a list of what no provider returns

STEP 3 -- agent.py  (Terminal 2, while app.py keeps running)

    5 domain questions -- one per document format -- sent to /ask, THEN:
      BONUS 1: /inspect, normal vs. deliberately truncated (max_tokens=10)
               -- watch finish_reason flip from "stop" to "length"
      BONUS 2: one question needing TWO tool calls, with the full
               step-by-step trace printed: type, duration, tokens, cost

STEP 4 (optional) -- inspect_vectorstore.py

    Peeks inside ./chroma_store -- through the Chroma API, then directly
    into chroma.sqlite3 -- to show where chunk text and metadata
    physically live on disk.
```

## 4. Setup — Step by Step

### Step 1 — Clone the repo

```
git clone <the link shared with you in the lab>
cd nirikshan-model-anatomy
```

### Step 2 — Create your `.env` file

Copy `.env.example` to a new file named exactly `.env`:

```
OPENAI_API_KEY="paste-the-key-shared-in-the-lab-here"
```

**Never share this file or commit it to Git** — it's already in `.gitignore`.

### Step 3 — Set up your environment and install dependencies

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 4 — Build the vector store (run once)

```
python build_vectorstore.py
```

Watch the console output — it reports how many units and chunks came
from each document, and how many PII items were redacted along the way
(it should report 3: one email, two phone numbers, from the FAQ).

### Step 5 — Start Nirikshan (Terminal 1)

```
uvicorn app:app --reload --port 8000
```

Leave this running. Visit `http://localhost:8000/docs` any time to try
`/ask` or `/inspect` interactively from a browser.

### Step 6 — Ask Nirikshan your questions (Terminal 2)

```
python agent.py
```

You'll see all 5 hardcoded questions, the two BONUS sections (raw-call
anatomy, and a traced multi-tool conversation), and every number printed
to the terminal.

## 5. The 5 Hardcoded Questions in `agent.py`

1. What is the maximum loan-to-value for a first-time buyer under the assisted scheme? → *PDF (SOP)*
2. What is the overdraft interest rate on a Premier account? → *DOCX (account terms)*
3. Who do I contact if I suspect fraud outside business hours? → *HTML (FAQ) — also tests PII redaction*
4. Which complaints are still open? → *CSV (complaints register)*
5. Which ISA product has the highest interest rate, and what's the early-access penalty on it? → *JSON (product rates)*

## 6. What to Notice While Running This

- **The API never tells you cost — you can watch us calculate it.** Run
  `agent.py`'s Bonus 1 and look at `we_measured_ourselves` in the
  `/inspect` response: `latency_ms` and `estimated_cost_usd` are both
  computed by this repo's own code (`time.perf_counter()` in `anatomy.py`,
  a rate table in `pricing.py`), not returned by the API.
- **`finish_reason` is the only truncation signal.** Compare the two
  `/inspect` calls in Bonus 1 — the truncated answer's `content` can
  still read as a plausible sentence fragment; only `finish_reason`
  flipping from `"stop"` to `"length"` proves it was cut off.
- **A two-tool question is really three model calls.** Run Bonus 2 and
  count the steps: ask → request tool 1 → request tool 2 → final answer
  is usually 3 model calls plus 2 tool calls, all separately timed.
- **Tool time and model time are tracked separately, on purpose.** Check
  `trace.summary.model_time_pct` vs `tool_time_pct` on any `/ask`
  response — mixing the two into one latency number would hide which
  half of a slow request is actually the slow one.
- **PII redaction happens at ingest, not just by convention.** Run
  `build_vectorstore.py` and read its console output — it should report
  3 redactions from the FAQ. Ask Question 3 via `agent.py` and confirm
  the real email/phone never appears in the answer.

## 7. A Note on the Pricing Table

`pricing.py`'s `PRICES_PER_MILLION_TOKENS_USD` reflects published OpenAI
rates at the time this lab was written. Prices change. If `estimate_cost_usd()`
returns `None` for a model you've switched to, that's the function telling
you its table is missing a row — add one, following the existing pattern,
rather than assuming the cost is zero.

## 8. Troubleshooting

| Problem | Likely cause |
|---|---|
| `ModuleNotFoundError` | Virtual environment not activated, or `pip install -r requirements.txt` not run |
| `app.py` fails with "OPENAI_API_KEY is not set" | Copy `.env.example` to `.env` and fill in your key |
| `/ask` fails with "vector store is empty" | Run `build_vectorstore.py` first, from this same folder |
| `agent.py` says "Could not reach Nirikshan" | `app.py` isn't running in the other terminal |
| `estimated_cost_usd` is `null` in a response | The model in `.env` isn't in `pricing.py`'s table — see Section 7 |
| Every `/ask` answer is generic or wrong | Confirm `build_vectorstore.py` completed without errors and reported all 5 documents |

## 9. Sample Assignment Outline (for Participants)

Use this as a graded or self-check exercise after completing the guided
walkthrough above.

**Assignment: Build Lekha, a Cost & Latency Audit Pipeline**

1. Starting from a copy of this repo, add ONE new tool alongside
   `lookup_policy` and `calculate_monthly_repayment` — for example, an
   ISA-allowance-remaining calculator using the rates in
   `05_savings_product_rates.json`.
2. Extend `tracer.py`'s `Trace.as_dict()` summary with a new field,
   `cost_per_tool_call`, reporting total cost broken down by which tool(s)
   were involved in the conversation that led to each model call.
3. Add a new endpoint, `GET /audit`, to `app.py` that accepts a list of
   questions (as repeated `q` query parameters), runs each through
   `_run_traced_agent_turn()`, and returns a single JSON summary: total
   questions, total cost across all of them, and the single most
   expensive question.
4. Run at least 5 questions through `/audit` in one call and confirm the
   summary's total cost matches the sum of each question's own trace.
5. **Written reflection (150–200 words):** Looking at your `/audit`
   results, which single question cost the most, and why? Was it the
   number of tool calls, the length of the retrieved context, or
   something else? What would you change about the system prompt or the
   tools to bring that cost down without losing correctness?

**Definition of done:**

- A third tool, registered and working through the normal `/ask` loop
- `cost_per_tool_call` present and correct in every `/ask` response's trace
- A working `/audit` endpoint, tested against at least 5 questions
- A short written reflection answering the question in step 5

## 10. Where This Goes Next

This lab measures one request at a time, by hand, printed to a terminal
or returned as JSON. Module 2 (Day 4–5) replaces `anatomy.py`'s manual
stopwatch and `tracer.py`'s hand-rolled step log with real OpenTelemetry
spans, exported to a proper trace backend — so the exact same questions
this lab answers ("where did the time go," "what did this cost," "was
the answer complete") become searchable across thousands of requests, not
just the one you happen to be looking at in a terminal.
