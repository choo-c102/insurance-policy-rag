import os
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.rag.vectorstore import create_vector_store, load_vector_store

def get_retriever(collection_name: str):
    return load_vector_store(collection_name=collection_name).as_retriever()

def pull_rag_prompt(prompt_name: str = "rlm/rag-prompt"):
    return hub.pull(prompt_name)

supported_llms = {
    "anthropic" : ChatAnthropic
}

def load_llm(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5-20250929",
    temperature: int=0,
    api_key: str = None
):
    if provider not in supported_llms:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Supported providers are: {list(supported_llms.keys())}")
    
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("No API key provided for Anthropic. Please set the ANTHROPIC_API_KEY environment variable or provide an API key.")
    
    llm_class = supported_llms[provider]
    return llm_class(model=model, temperature=temperature, api_key=api_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def run_rag_chain(
    query: str,
    collection_name: str = "default_collecion",
    prompt_name: str = "rlm/rag-prompt",
    provider: str = "anthropic"
):
    ''' Full RAG chain that takes a query and returns an answer '''
    retriever = get_retriever(collection_name=collection_name)
    prompt = pull_rag_prompt(prompt_name=prompt_name)
    llm = load_llm(provider=provider)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)