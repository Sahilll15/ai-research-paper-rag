import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import RateLimitError
from pydantic import BaseModel

from src.rag.pipeline import rag_graph

app = FastAPI(title="AI Research Paper Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-research-paper-rag.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class SourceRef(BaseModel):
    chunk_id: str
    paper: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    retries: int


ACRONYMS = {"ai", "gpt", "llm", "llms", "rlhf", "rag", "lora", "clip", "bert",
            "t5", "moe", "dpo", "rlaif", "io", "hbm", "sram", "cuda"}


def paper_title_from_filename(path: str) -> str:
    name = os.path.basename(path).removesuffix(".pdf")
    words = name.split("-")
    return " ".join(w.upper() if w.lower() in ACRONYMS else w.capitalize() for w in words)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # Returning a real Response for known failures (rather than letting the
    # exception propagate) matters here: an unhandled exception in this
    # handler skips CORSMiddleware, so the browser sees a bare CORS error
    # instead of the actual cause - e.g. the free OpenRouter model's shared
    # pool getting rate-limited under load, which is real and expected.
    try:
        result = rag_graph.invoke({"query": request.query})
    except RateLimitError:
        return JSONResponse(
            status_code=503,
            content={"detail": "The free model is rate-limited right now. Try again in a few seconds."},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong answering that. Try again."},
        )

    cited_ids = set(result["source_chunk_ids"])
    sources = [
        SourceRef(
            chunk_id=doc.metadata.get("_id", doc.id),
            paper=paper_title_from_filename(doc.metadata.get("source", "")),
            page=doc.metadata.get("page", 0),
        )
        for doc in result["chunks"]
        if doc.metadata.get("_id", doc.id) in cited_ids
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        retries=result["retry_count"],
    )
