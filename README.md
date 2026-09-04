# AI Research Paper Assistant

A retrieval-augmented generation (RAG) system that answers questions about AI/ML research papers, with citations back to the source paper and page. Built to demonstrate a production RAG pipeline end to end, not just a notebook demo.

![Demo](docs/demo.gif)

## What it does

Ask a question like "how does Constitutional AI differ from InstructGPT's RLHF approach?" and get an answer grounded in a local corpus of 41 papers (Transformer through 2026 releases), with the specific paper and page cited for every claim. If the corpus doesn't contain the answer, the system says so instead of guessing — a self-correcting LangGraph loop grades each retrieval and rewrites the query before falling back to a refusal.

## Stack

| Layer | Choice |
|---|---|
| Ingestion | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma (local) |
| Orchestration | LangGraph (retrieve → grade → generate, with a query-rewrite retry loop) |
| Generation | GPT-4o-mini, structured output via Pydantic (grounded answer + cited chunk IDs) |
| Serving | FastAPI |
| UI | Next.js (App Router) + Tailwind |
| Evaluation | RAGAS (faithfulness, context precision/recall) — in progress |

## Project structure

```
RAG_PROJECTS/
├── data/raw/papers/       # source PDFs (gitignored, see below)
├── src/rag/
│   ├── ingestion.py       # PDFs -> Document objects
│   ├── chunking.py        # Document objects -> chunks
│   ├── embeddings.py      # chunk text -> vectors
│   ├── vectorstore.py     # Chroma index (build + load)
│   ├── retrieval.py       # query -> top-k chunks
│   ├── generation.py      # chunks + query -> grounded, cited answer
│   └── pipeline.py        # LangGraph graph: retrieve -> grade -> generate (+ retry)
├── api/main.py             # FastAPI serving layer
├── frontend/               # Next.js UI (App Router)
├── eval/                   # hand-written Q&A set + RAGAS runner (in progress)
└── docs/ARCHITECTURE.md    # how the pieces fit together
```

## Setup

Backend:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:
```
OPENAI_API_KEY=...
```

Data isn't committed to this repo (see `.gitignore`) — point `data/raw/papers/` at your own PDFs, then build the index once:
```bash
python3 -c "from src.rag.ingestion import load_documents; from src.rag.chunking import chunk_documents; from src.rag.vectorstore import build_vectorstore; build_vectorstore(chunk_documents(load_documents()))"
```

Run the API:
```bash
uvicorn api.main:app --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`.

## Status

Core pipeline complete and working end to end: ingestion → chunking → embeddings → vectorstore → retrieval → LangGraph self-correction loop → generation, served over FastAPI with a Next.js UI. Evals (`eval/`) are the current focus — a hand-written 30-50 question set scored with RAGAS, to compare retrieval/chunking strategies with real numbers instead of eyeballing answers.
