from fastapi import APIRouter, HTTPException, Request
from schemas import Item, QueryRequest, QueryResponse, LogEntry
import uuid
from datetime import datetime
from routers.logs import LogDatabase
from config import LOG_DB_PATH

router = APIRouter(tags=["Query"])


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    searcher = http_request.app.state.searcher
    if searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search index is not loaded; build the keyframe index first",
        )

    request_id = str(uuid.uuid4())
    timestamp = datetime.now()

    hits = http_request.app.state.search_pool.submit(
        searcher.search, request.query, request.limit
    ).result()

    results = [
        Item(keyframe_id=hit.keyframe.keyframe_id, video_id=hit.keyframe.video_id)
        for hit in hits
    ]
    total = len(results)

    with LogDatabase(LOG_DB_PATH) as db:
        log_entry = LogEntry(
            request_id=request_id,
            timestamp=timestamp,
            query=request.query,
            limit=request.limit,
            mode=request.mode,
            results=results,
            total=total
        )
        db.logs.create(log_entry)

    return QueryResponse(
        results = results,
        total = total
    )
