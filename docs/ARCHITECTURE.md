# Architecture

Two separate paths through this system: an offline indexing path that runs once (or on a schedule) to build the searchable index, and an online query path that runs on every user question. Keeping these separate matters because they have different constraints: indexing can be slow and batch-oriented, querying has to be fast and safe.

## Offline: building the index

```
data/raw/papers/*.pdf
        │
        ▼
  ingestion.py     — extract text per page, attach {source, page} metadata
        │
        ▼
  chunking.py      — split into overlapping chunks, metadata carries through
        │
        ▼
  embeddings.py    — chunk text -> vector
        │
        ▼
  vectorstore.py   — persist {text, vector, metadata} to Qdrant
```

Output: a Qdrant collection (Qdrant Cloud free tier). This step is re-run whenever the corpus changes, not on every query.

## Online: answering a question

This is a LangGraph state machine, not a linear chain, because it branches and loops.

```
                 ┌─────────────┐
   question ───▶ │  retrieve   │  embed query, search vectorstore for top-k chunks
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │    grade    │  are these chunks actually relevant?
                 └──────┬──────┘
              relevant  │  not relevant
                        │  └──────────────┐
                        ▼                 ▼
                 ┌─────────────┐   ┌──────────────┐
                 │  generate   │   │ rewrite query │
                 └──────┬──────┘   └──────┬───────┘
                        │                 │
                        ▼                 └──▶ back to retrieve
                 {answer, source_chunk_ids}
```

Why a graph and not a chain: a plain LangChain chain (`retrieve | generate`) can't express the "grade, and loop back if the retrieval was bad" step, since that's a conditional edge with a cycle. LangGraph models this as an explicit state machine, which is also what makes it inspectable and testable node by node.

`generation.py` forces the LLM's output through a Pydantic schema (`answer: str`, `source_chunk_ids: list[str]`) rather than free text, so every claim can be checked against the chunk it supposedly came from. This is what makes citations possible and what makes hallucination detectable instead of just "looking right."

## Serving

`api/main.py` wraps the compiled LangGraph graph in a FastAPI endpoint. Request/response bodies are Pydantic models, so a bad request fails validation before it reaches the graph. `frontend/` is a Next.js client that calls this API; it holds no pipeline logic of its own.

## Evaluation

Evaluation happens outside the request path entirely, against `eval/dataset.jsonl` (a hand-written set of questions with known-good answers, including a few questions with no answer in the corpus, to test that the system says "I don't know" instead of guessing). `eval/run_eval.py` runs the full pipeline against this set and scores it with RAGAS (faithfulness, answer relevancy, context precision, context recall). This is also where chunking or retrieval strategies get compared against each other with actual numbers, rather than by eyeballing a few example answers.

## Observability

LangSmith traces every graph run: which node ran, how long it took, how many tokens it used, and what it returned. This is separate from the eval step above — tracing tells you what happened on a specific request, eval tells you how good the system is on average.
