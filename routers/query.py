from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, QueryResponse, LogEntry
import uuid
from keyframe import KeyframeSearcher
from datetime import datetime
from logs import LogDatabase
from config import LOG_DB_PATH

router = APIRouter(tags=["Query"])

@router.post("", response_model=QueryResponse)
def query(request: QueryRequest):
    request_id = str(uuid.uuid4())
    timestamp = datetime.now()

    response = KeyframeSearcher.search(
        query = request.query,
        limit = request.limit
    )

    total = len(response)

    with LogDatabase(LOG_DB_PATH) as db:
        log_entry = LogEntry(
            request_id=request_id,
            timestamp=timestamp,
            query=request.query,
            limit=request.limit,
            mode=request.mode,
            results=response,
            total=total
        )
        db.logs.create(log_entry)

    return QueryResponse(
        results = response,
        total = total
    )
