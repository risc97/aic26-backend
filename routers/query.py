from fastapi import APIRouter, HTTPException, Request
from schemas import (
    Item, QueryRequest, QueryResponse, LogEntry,
    TemporalQueryRequest, TemporalQueryResponse, TemporalItem, TemporalMatch,
)
import uuid
from datetime import datetime
from routers.logs import LogDatabase
from config import LOG_DB_PATH
from temporal import Gap, TemporalParams, temporal_search

router = APIRouter(tags=["Query"])

STRICT_SKIP_PENALTY = 1000.0  # far beyond any reachable score, so skips never win


def _searcher(http_request: Request, model: str):
    """The pool plus a loaded searcher, or 503 if that model has no index."""
    registry = http_request.app.state.searchers
    pool = http_request.app.state.search_pool
    try:
        # loading happens on the search thread, so the model lands in one place
        return pool, pool.submit(registry.get, model).result()
    except (KeyError, FileNotFoundError):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Search index for model '{model}' is not loaded; "
                f"available models: {registry.available() or 'none'}"
            ),
        )


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    pool, searcher = _searcher(http_request, request.model)

    request_id = str(uuid.uuid4())
    timestamp = datetime.now()

    hits = pool.submit(searcher.search, request.query, request.limit).result()

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
            model=request.model,
            results=results,
            total=total
        )
        db.logs.create(log_entry)

    return QueryResponse(
        results = results,
        total = total
    )


@router.post("/temporal", response_model=TemporalQueryResponse)
def temporal_query(request: TemporalQueryRequest, http_request: Request):
    pool, searcher = _searcher(http_request, request.model)

    if request.gaps is not None and len(request.gaps) != len(request.stages) - 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"gaps must hold one entry per transition: expected "
                f"{len(request.stages) - 1}, got {len(request.gaps)}"
            ),
        )

    stages = [[s.query, *s.variants] for s in request.stages]
    params = TemporalParams(
        candidate_k=request.candidate_k,
        gaps=[Gap(**g.model_dump()) for g in request.gaps] if request.gaps else Gap(),
        skip_penalty=(STRICT_SKIP_PENALTY if request.require_all_stages
                      else request.skip_penalty),
        span_penalty=request.span_penalty,
    )

    request_id = str(uuid.uuid4())
    timestamp = datetime.now()

    hits = pool.submit(
        temporal_search, searcher.index, searcher.db, searcher.encode_texts, stages,
        k=request.limit, params=params,
    ).result()

    results = [
        TemporalItem(
            rank=hit.rank,
            video_id=hit.video_id,
            score=hit.score,
            skipped_stages=hit.skipped,
            matches=[
                TemporalMatch(
                    stage=m.stage,
                    query=m.query,
                    score=m.score,
                    video_id=m.keyframe.video_id,
                    keyframe_id=m.keyframe.keyframe_id,
                    timestamp_ms=m.keyframe.timestamp_ms,
                )
                for m in hit.matches
            ],
        )
        for hit in hits
    ]

    with LogDatabase(LOG_DB_PATH) as db:
        # reuse the existing log table: the sequence flattens to one query string
        db.logs.create(LogEntry(
            request_id=request_id,
            timestamp=timestamp,
            query=" -> ".join(s.query for s in request.stages),
            limit=request.limit,
            mode="temporal",
            model=request.model,
            results=[Item(video_id=m.video_id, keyframe_id=m.keyframe_id)
                     for item in results for m in item.matches],
            total=len(results),
        ))

    return TemporalQueryResponse(results=results, total=len(results))
