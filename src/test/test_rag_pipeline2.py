'''
Test RAG pipeline with existing vector store
Before running the script, be sure to set PYTHONPATH to the root of the project:
- in powershell > $env:PYTHONPATH = "."
'''

import os
from dotenv import load_dotenv

from app.rag.loader import load_documents
from app.rag.chunker import split_into_chunks
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import load_vector_store, create_vector_store
from app.rag.retriever import run_rag_chain

load_dotenv()

vector_store = load_vector_store(collection_name='insurance_collection', persist_directory=os.getenv('PERSIST_DIRECTORY'))
print("Loaded vector store. Collection name: 'insurance_collection'")

test_qna = run_rag_chain(
    query = "How is Critical Illness defined in the policy?",
    collection_name='insurance_collection'
)

print(test_qna)