import os
import tarfile
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.embeddings import embeddings_model

VERCEL_CHROMA_DIR = "/tmp/chroma_db"


def build_vectorstore(chunks: list[Document], persist_directory="chroma_db") -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
    )


def _hydrate_from_blob() -> str:
    """Vercel Functions can't read the repo's gitignored chroma_db/, so on cold
    start we pull the prebuilt index from Blob into writable /tmp instead."""
    target = Path(VERCEL_CHROMA_DIR)
    marker = target / ".hydrated"
    if marker.exists():
        return str(target)

    blob_url = os.environ["CHROMA_BLOB_URL"]
    target.mkdir(parents=True, exist_ok=True)
    archive_path = "/tmp/chroma_db.tar.gz"
    urllib.request.urlretrieve(blob_url, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target, filter="data")
    os.remove(archive_path)
    marker.touch()
    return str(target)


def load_vectorstore(persist_directory="chroma_db"):
    if os.environ.get("VERCEL"):
        persist_directory = _hydrate_from_blob()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
