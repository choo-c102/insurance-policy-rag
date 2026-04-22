from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from app.rag.embeddings import get_embeddings
from app.core.config import settings
from pathlib import Path

def create_vector_store(
    chunks: list,
    collection_name: str = None, 
    embedding_function: Embeddings = None
):
    ''' Create and return a Chroma vector store and persist to disk '''

    if collection_name is None:
        raise ValueError("Collection name is required")
    
    if embedding_function is None:
        embedding_function = get_embeddings()
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=settings.persist_directory
    )
    vector_store.add_documents(chunks)
    return vector_store


def load_vector_store(
    collection_name: str = None, 
    embedding_function: Embeddings = None
):
    ''' Load and return an existing Chroma vector store from disk '''

    if collection_name is None:
        raise ValueError("Collection name is required")
    
    if not Path(settings.persist_directory).exists():
        raise FileNotFoundError(f"Persist directory '{settings.persist_directory}' does not exist. Please create the vector store first.")
    
    if embedding_function is None:
        embedding_function = get_embeddings()
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=settings.persist_directory
    )
    