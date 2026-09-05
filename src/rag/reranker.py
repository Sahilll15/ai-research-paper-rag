import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

# Verified against the live /v1/models?endpoint=rerank listing on 2026-09-05,
# where it is the account's default rerank model. rerank-v4.0-fast and
# rerank-v4.0-pro are also available via COHERE_RERANK_MODEL.
DEFAULT_RERANK_MODEL = "rerank-v3.5"


@lru_cache(maxsize=1)
def _client():
    import cohere

    return cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    """Reorder hybrid candidates with Cohere's cross-encoder and keep the top_n.

    Without COHERE_API_KEY, or if the API call fails, this falls back to the
    fusion order so retrieval quality degrades rather than the request."""
    if not docs:
        return []

    if not os.environ.get("COHERE_API_KEY"):
        return docs[:top_n]

    try:
        response = _client().rerank(
            model=os.environ.get("COHERE_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=min(top_n, len(docs)),
        )
    except Exception:
        logger.warning("Cohere rerank failed; falling back to fusion order", exc_info=True)
        return docs[:top_n]

    return [docs[result.index] for result in response.results]
