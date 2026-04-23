'''
Accept a user query and a collection name
Run the query through the RAG chain
Return the generated answer
'''

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.retriever import run_rag_chain

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    collection_name: str
    prompt_name: str = "rag"
    provider: str = "anthropic"


class QueryResponse(BaseModel):
    question: str
    answer: str
    collection_name: str


@router.post("/query", response_model=QueryResponse)
async def query_policy(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    try:
        answer = run_rag_chain(
            query=request.question,
            collection_name=request.collection_name,
            prompt_name=request.prompt_name,
            provider=request.provider,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")

    return QueryResponse(
        question=request.question,
        answer=answer,
        collection_name=request.collection_name,
    )
