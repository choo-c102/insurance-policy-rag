from langchain_voyageai import VoyageAIEmbeddings

supported_embeddings = {
    "voyage": VoyageAIEmbeddings
}

def get_embeddings(provider: str = "voyage", model: str = "voyage-3"):
    ''' Initialize and return the specified embedding model. '''
    if provider not in supported_embeddings:
        raise ValueError(f"Unsupported embedding provider '{provider}'. Supported providers are: {list(supported_embeddings.keys())}")
    
    embedding_class = supported_embeddings[provider]
    return embedding_class(model=model)

