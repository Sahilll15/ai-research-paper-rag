from src.rag.vectorstore import load_vectorstore

def retrieval(query:str,k:int=5):
    vector_store=load_vectorstore()
    return vector_store.similarity_search(query,k)


if __name__ == "__main__":
    print(retrieval("how does flash attention reduce memory usage"))
