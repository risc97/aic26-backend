from fastapi import APIRouter, HTTPException, Request, Query
from .query import _searcher
from config import DEFAULT_MODEL
from schemas import Item, KeyframeModel, SimilarResponse

router = APIRouter(tags=["Similar"])

@router.get("/{video_id}/{keyframe_id}", response_model=SimilarResponse)
def similar_keyframes(video_id: str, keyframe_id: str,
                      http_request: Request,
                      model: KeyframeModel = Query(DEFAULT_MODEL, description="Model để so sánh"),
                      limit: int = 100
                      ) -> SimilarResponse:
    pool, searcher = _searcher(http_request, "keyframe", model)
    try:
        hits = pool.submit(searcher.similar, video_id, keyframe_id, limit).result()
    except FileNotFoundError as e:
        raise HTTPException(503, f"embeddings for model '{model}' not on disk: {e}")
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))

    results = [
        Item(
            keyframe_id=h.keyframe.keyframe_id, video_id=h.keyframe.video_id,
            timestamp_ms=h.keyframe.timestamp_ms, frame_idx=h.keyframe.frame_idx,
            video_fps=h.video_fps, score=h.score,
        )
        for h in hits
    ]
    return SimilarResponse(video_id=video_id, keyframe_id=keyframe_id, model=model, total=len(results), results=results)