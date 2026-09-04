from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def create_embedding(docs: list[Document]):
    texts = [doc.page_content for doc in docs]
    return embeddings_model.embed_documents(texts)