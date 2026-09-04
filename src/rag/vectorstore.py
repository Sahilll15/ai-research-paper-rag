from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.embeddings import embeddings_model


def build_vectorstore(chunks: list[Document], persist_directory="chroma_db") -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
    )
    
def load_vectorstore(persist_directory="chroma_db"):
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
