from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.documents import Document

from src.rag.llm import get_chat_llm, invoke_structured

load_dotenv()

llm = get_chat_llm()


class RAGAnswer(BaseModel):
    answer: str
    source_chunk_ids: list[str]


def generate(query: str, chunks: list[Document]) -> RAGAnswer:
    chunk_ids = [str(doc.metadata.get("_id", doc.id)) for doc in chunks]
    context = "\n\n".join(
        f"[{cid}] {doc.page_content}" for cid, doc in zip(chunk_ids, chunks)
    )

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know - do not guess. "
        "Cite the chunk IDs your answer draws from.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    result, raw = invoke_structured(llm, RAGAnswer, prompt)
    if result is not None:
        return result

    # Prose reply: keep the answer and whatever chunk ids it mentioned inline.
    return RAGAnswer(
        answer=raw.strip(),
        source_chunk_ids=[cid for cid in chunk_ids if cid in raw],
    )
