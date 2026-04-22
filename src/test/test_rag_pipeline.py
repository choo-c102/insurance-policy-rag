'''
Test RAG pipeline
Before running the script, be sure to set PYTHONPATH to the root of the project:
- in powershell > $env:PYTHONPATH = "."
'''

import os
from dotenv import load_dotenv

from app.rag.loader import load_documents
from app.rag.chunker import split_into_chunks
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import create_vector_store
from app.rag.retriever import run_rag_chain

load_dotenv()

policy_docs = load_documents(os.getenv("PDF_PATH"))
print(f"Number of pages loaded: {len(policy_docs)}")

doc_chunks = split_into_chunks(documents=policy_docs)
print(f"Split documents into {len(doc_chunks)} sub-documents.")

embeddings = get_embeddings()
print("Initialized embedding model.")

vector_store = create_vector_store(
    chunks=doc_chunks, 
    collection_name='insurance_collection',
    embedding_function=embeddings,
    persist_directory=os.getenv('PERSIST_DIRECTORY'))
print("Created and persisted vector store. Collection name: 'insurance_collection'")

test_qna = run_rag_chain(
    query = "How is Critical Illness defined in the policy?",
    collection_name='insurance_collection'
)

print(test_qna)