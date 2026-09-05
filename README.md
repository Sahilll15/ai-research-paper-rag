# AI Research Paper Assistant

**Live demo:** https://ai-research-paper-rag.vercel.app

A retrieval-augmented generation (RAG) system that answers questions about AI/ML research papers, with citations back to the source paper and page. Built to demonstrate a production RAG pipeline end to end: real evaluation numbers, a self-correcting retrieval loop, cost-safe deployment, and tracing, not just a notebook demo.

> The public demo runs on `z-ai/glm-5.2:free`, a free OpenRouter model, to keep the deployment from touching anyone's OpenAI billing. Free-tier OpenRouter rate limits apply (20 requests/minute, 50/day), so it can go quiet under load and answers won't match the quality of local dev, which runs `gpt-4o-mini` directly against OpenAI.

![Demo](docs/demo.gif)

## What it does

Ask a question like "how does Constitutional AI differ from InstructGPT's RLHF approach?" and get an answer grounded in a local corpus of 41 papers (the Transformer through 2026 releases), with the specific paper and page cited for every claim. If the corpus doesn't contain the answer, the system says so instead of guessing: a self-correcting LangGraph loop grades each retrieval and rewrites the query before falling back to a refusal.

## Stack

| Layer | Choice |
|---|---|
| Ingestion | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant Cloud (free tier) |
| Orchestration | LangGraph (retrieve → grade → generate, with a query-rewrite retry loop) |
| Generation | `gpt-4o-mini` locally via OpenAI; free `z-ai/glm-5.2:free` via OpenRouter in production (structured output via Pydantic — grounded answer + cited chunk IDs) |
| Serving | FastAPI |
| UI | Next.js (App Router) + Tailwind, manuscript-style design (see screenshot above) |
| Evaluation | RAGAS — faithfulness, answer relevancy, context precision, context recall, scored against a hand-written 51-question set |
| Observability | LangSmith tracing, automatic across every LangGraph node |
| Deployment | Next.js on Vercel, FastAPI on Render |

## Evaluation

`eval/dataset.jsonl` is 51 hand-written questions across three categories: single-document factual, cross-document comparisons (forces real multi-paper retrieval), and adversarial/out-of-scope questions that should produce a refusal, not a guess. `eval/run_eval.py` runs every question through the real pipeline and scores the actual retrieved context and generated answer with RAGAS.

Latest run:

| Metric | Score |
|---|---|
| Faithfulness | 0.75 |
| Answer relevancy | 0.75 |
| Context precision | 0.80 |
| Context recall | 0.77 |

Full per-question results are saved under `eval/runs/`, so a later change (chunking strategy, retrieval strategy, prompt tweak) can be diffed against this baseline instead of eyeballing a handful of example answers.

## Observability

Every `rag_graph.invoke()` call is traced automatically through LangSmith, no manual instrumentation required since the whole pipeline is built on LangChain/LangGraph primitives. Each node (`retrieve_node`, `grade_node`, `generate_node`) shows up as its own span with latency, token counts, and cost:

![LangSmith trace](docs/langsmith-trace.png)

## Project structure

```
RAG_PROJECTS/
├── data/raw/papers/       # source PDFs (gitignored, see below)
├── src/rag/
│   ├── ingestion.py       # PDFs -> Document objects
│   ├── chunking.py        # Document objects -> chunks
│   ├── embeddings.py      # chunk text -> vectors
│   ├── vectorstore.py     # Qdrant Cloud index (build + load)
│   ├── retrieval.py       # query -> top-k chunks
│   ├── generation.py      # chunks + query -> grounded, cited answer
│   ├── llm.py             # shared chat model: OpenAI locally, OpenRouter in production
│   └── pipeline.py        # LangGraph graph: retrieve -> grade -> generate (+ retry)
├── api/main.py             # FastAPI serving layer
├── frontend/               # Next.js UI (App Router)
├── eval/
│   ├── dataset.jsonl       # 51 hand-written Q&A pairs
│   ├── run_eval.py         # RAGAS runner
│   └── runs/               # timestamped per-run results
├── vercel.json             # single-project Vercel Services config (frontend + Python API)
└── docs/ARCHITECTURE.md    # how the pieces fit together
```

## Setup

Backend:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API keys to `.env`:
```
OPENAI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```
`QDRANT_URL`/`QDRANT_API_KEY` come from a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) (no card required).

Optional: LangSmith tracing (no code changes needed):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ai-research-paper-rag
```

Data isn't committed to this repo (see `.gitignore`) — point `data/raw/papers/` at your own PDFs, then build the index once (wipes and recreates the Qdrant collection):
```bash
python3 -c "from src.rag.ingestion import load_documents; from src.rag.chunking import chunk_documents; from src.rag.vectorstore import build_vectorstore; build_vectorstore(chunk_documents(load_documents()))"
```

Run the API:
```bash
uvicorn api.main:app --port 8000
```

Run the evals (extra deps, kept out of `requirements.txt` on purpose — `ragas`/`datasets` are heavy and would blow past Vercel's function size limit if the deployed API installed them too):
```bash
pip install -r requirements-eval.txt
python3 -m eval.run_eval
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`.

## Deployment

Frontend and backend are split across two hosts:

- **Frontend** — Next.js on Vercel (`ai-research-paper-rag.vercel.app`), `frontend/` deployed via [Vercel Services](https://vercel.com/docs/services).
- **Backend** — FastAPI on [Render](https://render.com) (`ai-research-paper-rag.onrender.com`), free web service, deployed straight from this repo (`pip install -r requirements.txt`, `uvicorn api.main:app --host 0.0.0.0 --port $PORT`).

This started as a single Vercel project running both, with the FastAPI backend as a second Vercel Service. It hit Vercel's Python function bundle limit even after replacing Chroma with Qdrant Cloud (chromadb's own dependencies — onnxruntime, grpcio, kubernetes, all unused in embedded mode — were the original blocker; `qdrant-client` itself still carries a hard `grpcio` dependency baked into its main import path, unsafe to strip). Moving just the backend to Render, which has no such packaging ceiling, was simpler than fighting the dependency tree further.

The vector index lives in Qdrant Cloud (free tier, no card required) rather than a file bundled with either deployment — the client is a thin API wrapper, the actual vector search runs on Qdrant's own server.

Required backend environment variables (set on Render):
- `OPENAI_API_KEY` — used for embeddings (`text-embedding-3-small`) even in production; cheap enough per query to leave on direct OpenAI billing.
- `OPENROUTER_API_KEY` — required for chat/generation in production. If it's missing, the app deliberately fails at startup rather than silently falling back to `gpt-4o-mini` on the maintainer's OpenAI key.
- `QDRANT_URL` / `QDRANT_API_KEY` — the Qdrant Cloud cluster holding the index.

Required frontend environment variable (set on Vercel): `NEXT_PUBLIC_API_URL` — the Render backend URL, since the two hosts are genuinely cross-origin now (CORS in `api/main.py` allows the Vercel frontend's origin).

Optional, for tracing the backend too: `LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` as in local dev.

## Status

Complete and working end to end: ingestion → chunking → embeddings → vectorstore → retrieval → LangGraph self-correction loop → generation, served over FastAPI (Render) with a Next.js UI (Vercel), behind a cost-safe production model swap, evaluated with RAGAS against a 51-question hand-written set, and traced end to end with LangSmith.
