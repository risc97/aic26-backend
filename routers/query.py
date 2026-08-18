from fastapi import APIRouter, HTTPException, Request
from schemas import Item, QueryRequest, QueryResponse, LogEntry
import uuid
from datetime import datetime
from routers.logs import LogDatabase
from config import LOG_DB_PATH

router = APIRouter(tags=["Query"])

TRANSCRIPT_MODEL = "gte"


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    registry = http_request.app.state.searchers.get(request.mode)
    if registry is None:
        raise HTTPException(status_code=501, detail=f"mode '{request.mode}' is not implemented")
    pool = http_request.app.state.search_pool

    
    if request.mode == "transcript_semantic":
        model = TRANSCRIPT_MODEL
    else:
        model = request.model

    try:
        # loading happens on the search thread, so the model lands in one place
        searcher = pool.submit(registry.get, model).result()
    except (KeyError, FileNotFoundError):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Search index for model '{model}' is not loaded; "
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
        for keyframe in (hit.keyframes if hasattr(hit, "keyframes") else [hit.keyframe])
    ]
    total = len(results)

    with LogDatabase(LOG_DB_PATH) as db:
        log_entry = LogEntry(
            request_id=request_id,
            timestamp=timestamp,
            query=request.query,
            limit=request.limit,
            mode=request.mode,
            model=model,
            results=results,
            total=total
        )
        db.logs.create(log_entry)

    return QueryResponse(
        results = results,
        total = total
    )
