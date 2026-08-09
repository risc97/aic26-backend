from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, QueryResponse
import uuid
from keyframe import KeyframeSearcher

router = APIRouter(tags=["Query"])

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
