from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, QueryResponse
import uuid
from search import KeyframeSearcher

router = APIRouter(tags=["videos"])

@router.get("", response_model=QueryResponse)
def query(request: QueryRequest):
    request_id = str(uuid.uuid4())
    response = KeyframeSearcher.search(
        query = request.query,
        limit = request.limit
    )
    return QueryResponse(
        results = [response],
        total = len(results)
    )
