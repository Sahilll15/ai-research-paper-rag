import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

EMBEDDING_DIM = 1536  # text-embedding-3-small

from src.rag.embeddings import embeddings_model

load_dotenv()

COLLECTION_NAME = "ai_research_papers"


def _client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])


def build_vectorstore(chunks: list[Document]) -> QdrantVectorStore:
    """One-time (re)index: wipes and recreates the collection. Don't call this
    from request-serving code - it's for the local indexing script only."""
    client = _client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    vs = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings_model)
    vs.add_documents(chunks)
    return vs


def load_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=_client(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model,
    )
