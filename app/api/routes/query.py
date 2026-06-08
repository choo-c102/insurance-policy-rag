from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag.retriever import run_rag_chain
from app.core.collections import get_stored_collections

router = APIRouter()
class QueryRequest(BaseModel):
    query: str

@router.post("/query/{collection_name}")
async def query_policy(collection_name: str, query_request: QueryRequest):
    if collection_name not in get_stored_collections(): #Check if collection name is valid
        raise HTTPException(status_code=404, detail=f"Collection name '{collection_name}' not found. Please select a policy from the list of available policies.")
    
    try:
        return run_rag_chain(query=query_request.query, collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying policy: {str(e)}")
