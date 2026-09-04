# AI Research Paper Assistant

A retrieval-augmented generation (RAG) system that answers questions about AI/ML research papers, with citations back to the source paper and page. Built to demonstrate a production RAG pipeline end to end, not just a notebook demo.

## What it does

Ask a question like "how does Constitutional AI differ from InstructGPT's RLHF approach?" and get an answer grounded in a local corpus of 40+ papers (Transformer through 2026 releases), with the specific paper and page cited for every claim.

## Stack

| Layer | Choice |
|---|---|
| Ingestion | PyPDFLoader / pdfplumber |
| Chunking | RecursiveCharacterTextSplitter, compared against semantic chunking |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma (local) |
| Orchestration | LangGraph (retrieve → grade → generate → self-correct) |
| Generation | GPT-4o-mini, structured output via Pydantic |
| Evaluation | RAGAS (faithfulness, context precision/recall) + LangSmith tracing |
| Serving | FastAPI |
| Demo UI | Streamlit |

## Project structure

```
RAG_PROJECTS/
├── data/raw/papers/       # source PDFs (gitignored, see below)
├── src/rag/
│   ├── ingestion.py       # PDFs -> Document objects
│   ├── chunking.py        # Document objects -> chunks
│   ├── embeddings.py      # chunk text -> vectors
│   ├── vectorstore.py     # Chroma index
│   ├── retrieval.py       # query -> top-k chunks
│   ├── generation.py      # chunks + query -> grounded answer
│   └── pipeline.py        # LangGraph graph wiring it together
├── api/main.py            # FastAPI serving layer
├── app/streamlit_app.py   # demo UI
├── eval/                  # hand-written Q&A set + RAGAS runner
└── tests/                 # pytest coverage on chunking/retrieval
```

See `docs/ARCHITECTURE.md` for how the pieces fit together.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:

```
OPENAI_API_KEY=...
```

Data isn't committed to this repo (see `.gitignore`). Fetch the corpus with `scripts/download_papers.py` (not yet written) or point `data/raw/papers/` at your own PDFs.

## Status

Work in progress. Pipeline stages are scaffolded in `src/rag/`; corpus of 41 papers downloaded; implementation in progress starting with `ingestion.py`.
