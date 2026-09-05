import os
import uuid
from itertools import batched

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

EMBEDDING_DIM = 1536  # text-embedding-3-small

from src.rag.embeddings import embeddings_model

load_dotenv()

COLLECTION_NAME = "ai_research_papers"

# Mirror QdrantVectorStore's own naming so load_vectorstore() can still read a
# collection written by build_vectorstore(). "" is Qdrant's unnamed dense vector.
DENSE_VECTOR_NAME = QdrantVectorStore.VECTOR_NAME
SPARSE_VECTOR_NAME = QdrantVectorStore.SPARSE_VECTOR_NAME
CONTENT_KEY = QdrantVectorStore.CONTENT_KEY
METADATA_KEY = QdrantVectorStore.METADATA_KEY

# BM25 runs on Qdrant Cloud's inference service, not here: the local
# alternative (fastembed) drags in onnxruntime and ~145MB of site-packages,
# which the 512MB Render instance can't afford.
SPARSE_MODEL = "Qdrant/bm25"

UPSERT_BATCH_SIZE = 64
PREFETCH_MULTIPLIER = 4


def _client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])


def _points(chunks: list[Document]) -> list[models.PointStruct]:
    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embeddings_model.embed_documents(texts)
    return [
        models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: models.Document(text=text, model=SPARSE_MODEL),
            },
            payload={CONTENT_KEY: text, METADATA_KEY: chunk.metadata},
        )
        for chunk, text, dense_vector in zip(chunks, texts, dense_vectors, strict=True)
    ]


def _document_from_point(point: models.ScoredPoint) -> Document:
    payload = point.payload or {}
    metadata = payload.get(METADATA_KEY) or {}
    metadata["_id"] = point.id
    metadata["_collection_name"] = COLLECTION_NAME
    return Document(page_content=payload.get(CONTENT_KEY, ""), metadata=metadata)


def build_vectorstore(chunks: list[Document]) -> QdrantVectorStore:
    """One-time (re)index: wipes and recreates the collection. Don't call this
    from request-serving code - it's for the local indexing script only."""
    client = _client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        sparse_vectors_config={
            # Points carry term frequencies only; IDF is applied server-side at
            # query time, which is what Qdrant/bm25 expects.
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    for batch in batched(chunks, UPSERT_BATCH_SIZE):
        client.upsert(collection_name=COLLECTION_NAME, points=_points(list(batch)))

    return load_vectorstore()


def hybrid_search(query: str, k: int = 5) -> list[Document]:
    """Dense (OpenAI) and sparse (BM25) candidates fused by Qdrant's native
    reciprocal rank fusion. Each arm prefetches wider than k so the lexical
    side can pull in exact-term matches the dense arm ranks below the cut."""
    prefetch_limit = max(PREFETCH_MULTIPLIER * k, 20)
    response = _client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=embeddings_model.embed_query(query),
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=SPARSE_MODEL),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [_document_from_point(point) for point in response.points]


def load_vectorstore() -> QdrantVectorStore:
    """Dense-only handle on the collection. Hybrid queries go through
    hybrid_search() instead - langchain-qdrant's RetrievalMode.HYBRID requires a
    local sparse encoder, and we deliberately don't ship one."""
    return QdrantVectorStore(
        client=_client(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model,
    )
