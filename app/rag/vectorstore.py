import os
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from app.rag.embeddings import get_embeddings

def create_vector_store(
    chunks: list,
    collection_name: str = "default_collection", 
    embedding_function: Embeddings = None, 
    persist_directory: str = "./chroma_langchain_db"):
    ''' Create and return a Chroma vector store and persist to disk '''
    if embedding_function is None:
        embedding_function = get_embeddings()
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    vector_store.add_documents(chunks)
    return vector_store


def load_vector_store(
    collection_name: str = "default_collection", 
    embedding_function: Embeddings = None, 
    persist_directory: str = "./chroma_langchain_db"):
    ''' Load and return an existing Chroma vector store from disk '''

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Persist directory '{persist_directory}' does not exist. Please create the vector store first.")
    
    if embedding_function is None:
        embedding_function = get_embeddings()
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    