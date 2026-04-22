from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.rag.vectorstore import load_vector_store
from app.llm.client import load_llm
from app.llm.prompts import pull_rag_prompt

def get_retriever(collection_name: str):
    return load_vector_store(collection_name=collection_name).as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def run_rag_chain(
    query: str,
    collection_name: str = None,
    prompt_name: str = "rag",
    provider: str = "anthropic"
):
    ''' Full RAG chain that takes a query and returns an answer '''
    
    if collection_name is None:
        raise ValueError("Collection name is required")
    
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