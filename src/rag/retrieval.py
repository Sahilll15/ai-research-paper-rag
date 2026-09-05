from langchain_core.documents import Document

from src.rag.vectorstore import hybrid_search


def retrieval(query: str, k: int = 5) -> list[Document]:
    return hybrid_search(query, k)


if __name__ == "__main__":
    print(retrieval("how does flash attention reduce memory usage"))
