from fastapi import APIRouter, HTTPException, Request
from schemas import Item, QueryRequest, QueryResponse, LogEntry
import uuid
from datetime import datetime
from routers.logs import LogDatabase
from config import LOG_DB_PATH

router = APIRouter(tags=["Query"])


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    registry = http_request.app.state.searchers
    pool = http_request.app.state.search_pool

    try:
        # loading happens on the search thread, so the model lands in one place
        searcher = pool.submit(registry.get, request.model).result()
    except (KeyError, FileNotFoundError):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Search index for model '{request.model}' is not loaded; "
                f"available models: {registry.available() or 'none'}"
            ),
        )

    request_id = str(uuid.uuid4())
    timestamp = datetime.now()

    # calling keyframe/search.py: def search(self, query: str, k: int = 20) -> list[SearchHit]
    hits = pool.submit(searcher.search, request.query, request.limit).result()

    results = [
        Item(keyframe_id=hit.keyframe.keyframe_id, video_id=hit.keyframe.video_id, timestamp_ms=hit.keyframe.timestamp_ms)
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
            model=request.model,
            results=results,
            total=total
        )
        db.logs.create(log_entry)

    return QueryResponse(
        results = results,
        total = total
    )
