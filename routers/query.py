from fastapi import APIRouter, HTTPException, Request
from schemas import Item, OcrItem, LogEntry, TranscriptItem, TemporalQueryRequest, TemporalQueryResponse, TemporalItem, TemporalMatch
from schemas import KeyframeQueryRequest, TranscriptQueryRequest, OcrQueryRequest, KeyframeQueryResponse, TranscriptQueryResponse, OcrQueryResponse
import uuid
from datetime import datetime
from routers.logs import LogDatabase
from config import LOG_DB_PATH
from temporal import Gap, TemporalParams, temporal_search

router = APIRouter(tags=["Query"])

TRANSCRIPT_MODEL = "gte"


def _searcher(http_request: Request, mode: str, model: str):
    registry = http_request.app.state.searchers.get(mode)
    if registry is None:
        raise HTTPException(status_code=501, detail=f"mode '{mode}' is not implemented")
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

def _log(mode: str, model: str | None, query: str, limit: int, results: list) -> str:
    request_id = str(uuid.uuid4())
    with LogDatabase(LOG_DB_PATH) as db:
        db.logs.create(LogEntry(
            request_id=request_id, timestamp=datetime.now(), query=query,
            limit=limit, mode=mode, model=model, results=results, total=len(results),
        ))
    return request_id

@router.post("/keyframe", response_model=KeyframeQueryResponse)
def query_keyframe(request: KeyframeQueryRequest, http_request: Request):
    pool, searcher = _searcher(http_request, "keyframe", request.model)
    # calling keyframe/search.py: def search(self, query: str, k: int = 20) -> list[SearchHit]
    hits = pool.submit(searcher.search, request.query, request.limit).result()
    results = [
        Item(
            keyframe_id=hit.keyframe.keyframe_id, video_id=hit.keyframe.video_id,
            timestamp_ms=hit.keyframe.timestamp_ms, frame_idx=hit.keyframe.frame_idx,
            video_fps=hit.video_fps, score=hit.score,
        ) for hit in hits
    ]
    request_id = _log("keyframe", request.model, request.query, request.limit, results)
    return KeyframeQueryResponse(request_id=request_id, model=request.model, results=results, total=len(results))

@router.post("/transcript", response_model=TranscriptQueryResponse)
def query_transcript(request: TranscriptQueryRequest, http_request: Request):
    pool, searcher = _searcher(http_request, "transcript_semantic", request.model)
    hits = pool.submit(searcher.search, request.query, request.limit).result()
    results = [
        TranscriptItem(
            video_id=hit.transcript.video_id,
            transcript_id=hit.transcript.transcript_id,
            text=hit.transcript.text,
            time_start_ms=hit.transcript.time_start_ms,
            time_end_ms=hit.transcript.time_end_ms,
            keyframes=[
                Item(
                    keyframe_id=kf.keyframe_id, video_id=kf.video_id,
                    timestamp_ms=kf.timestamp_ms, frame_idx=kf.frame_idx,
                    video_fps=hit.video_fps, score=hit.score,
                ) for kf in hit.keyframes
            ],
        ) for hit in hits
    ]
    request_id = _log("transcript_semantic", request.model, request.query, request.limit, results)
    return TranscriptQueryResponse(request_id=request_id, model=request.model, results=results, total=len(results))

@router.post("/ocr", response_model=OcrQueryResponse)
def query_ocr(request: OcrQueryRequest, http_request: Request):
    pool, searcher = _searcher(http_request, "ocr_exact", None)
    hits = pool.submit(searcher.search, request.query, request.limit, request.phrase).result()
    results = [
        OcrItem(
            keyframe_id=hit.keyframe.keyframe_id, video_id=hit.keyframe.video_id,
            timestamp_ms=hit.keyframe.timestamp_ms, frame_idx=hit.keyframe.frame_idx,
            video_fps=hit.video_fps, text=hit.text, score=hit.score,
        ) for hit in hits
    ]
    request_id = _log("ocr_exact", None, request.query, request.limit, results)
    return OcrQueryResponse(request_id=request_id, results=results, total=len(results))

@router.post("/temporal", response_model=TemporalQueryResponse)
def temporal_query(request: TemporalQueryRequest, http_request: Request):
    pool, searcher = _searcher(http_request, "keyframe", request.model)

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
        skip_penalty=request.skip_penalty,
        max_skips=request.max_skips,
        score_cap=request.score_cap,
        span_penalty=request.span_penalty,
        beta=request.beta,
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
                    frame_idx=m.keyframe.frame_idx,
                    video_fps=m.video_fps,
                )
                for m in hit.matches
            ],
        )
        for hit in hits
    ]

    request_id = _log("temporal", request.model,
                  " -> ".join(s.query for s in request.stages),
                  request.limit, results)
    return TemporalQueryResponse(request_id=request_id, model=request.model, results=results, total=len(results))
