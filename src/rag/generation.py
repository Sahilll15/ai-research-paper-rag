from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.documents import Document

from src.rag.llm import get_chat_llm

load_dotenv()

llm = get_chat_llm()


class RAGAnswer(BaseModel):
    answer: str
    source_chunk_ids: list[str]


structured_llm_call = llm.with_structured_output(RAGAnswer)


def generate(query: str, chunks: list[Document]) -> RAGAnswer:
    context = "\n\n".join(f"[{doc.id}] {doc.page_content}" for doc in chunks)

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know - do not guess. "
        "Cite the chunk IDs your answer draws from.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    return structured_llm_call.invoke(prompt)
