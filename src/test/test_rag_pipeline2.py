'''
Test RAG pipeline with existing vector store
Before running the script, be sure to set PYTHONPATH to the root of the project:
- in powershell > $env:PYTHONPATH = "."
'''

from app.core.config import settings
from app.rag.vectorstore import load_vector_store
from app.rag.retriever import run_rag_chain

vector_store = load_vector_store(collection_name='insurance_collection')
print("Loaded vector store. Collection name: 'insurance_collection'")

test_qna = run_rag_chain(
    query = "How is Critical Illness defined in the policy?",
    collection_name='insurance_collection'
)

print(test_qna)