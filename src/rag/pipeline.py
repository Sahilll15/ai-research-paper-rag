from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.rag.retrieval import retrieval
from src.rag.generation import generate
from src.rag.llm import get_chat_llm
from src.rag.reranker import rerank

load_dotenv()

MAX_RETRIES = 2
CANDIDATE_K = 20  # hybrid candidates handed to the reranker
TOP_K = 5  # chunks that survive reranking into grading and generation


class RAGState(TypedDict):
    query: str
    candidates: list[Document]
    chunks: list[Document]
    answer: str
    source_chunk_ids: list[str]
    is_relevant: bool
    retry_count: int


class GradeResult(BaseModel):
    is_relevant: bool
    rewritten_query: str


grading_llm = get_chat_llm().with_structured_output(GradeResult)


def retrieve_node(state: RAGState) -> dict:
    candidates = retrieval(state['query'], k=CANDIDATE_K)
    return {
        "candidates": candidates
    }


def rerank_node(state: RAGState) -> dict:
    chunks = rerank(state['query'], state['candidates'], top_n=TOP_K)
    return {
        "chunks": chunks
    }


def grade_node(state: RAGState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["chunks"])
    prompt = (
        "Question: " + state["query"] + "\n\n"
        "Retrieved context:\n" + context + "\n\n"
        "Be strict. Set is_relevant to true ONLY if the context contains specific "
        "information that directly and substantively answers the question. "
        "Being on a loosely related topic is NOT enough - the context must "
        "actually contain the answer. If the context is about something else "
        "entirely, or only tangentially related, set is_relevant to false.\n"
        "If false, rewrite the query (rewritten_query) to be more specific or "
        "use different terminology than the original, so retrieval is more "
        "likely to find relevant chunks."
    )
    result = grading_llm.invoke(prompt)

    updates = {
        "is_relevant": result.is_relevant,
        "retry_count": state.get("retry_count", 0) + 1,
    }
    if not result.is_relevant:
        updates["query"] = result.rewritten_query
    return updates


def generate_node(state: RAGState) -> dict:
    result = generate(state['query'], state['chunks'])
    return {
        "answer": result.answer,
        "source_chunk_ids": result.source_chunk_ids
    }


def route_after_grade(state: RAGState) -> str:
    if state["is_relevant"] or state["retry_count"] >= MAX_RETRIES:
        return "generate_node"
    return "retrieve_node"


graph = StateGraph(RAGState)

graph.add_node('retrieve_node', retrieve_node)
graph.add_node('rerank_node', rerank_node)
graph.add_node('grade_node', grade_node)
graph.add_node('generate_node', generate_node)

graph.add_edge(START, 'retrieve_node')
graph.add_edge('retrieve_node', 'rerank_node')
graph.add_edge('rerank_node', 'grade_node')
graph.add_conditional_edges(
    'grade_node',
    route_after_grade,
    {"generate_node": "generate_node", "retrieve_node": "retrieve_node"},
)
graph.add_edge('generate_node', END)

rag_graph = graph.compile()

if __name__ == "__main__":
    result = rag_graph.invoke({"query": "how does flash attention reduce memory usage"})
    print(result["answer"])
    print(result["source_chunk_ids"])
    print("retries:", result["retry_count"])
