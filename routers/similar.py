from fastapi import APIRouter, HTTPException, Request, Query, UploadFile, File
from .query import _searcher
from config import DEFAULT_MODEL
from schemas import Item, SimilarModel, SimilarResponse
from PIL import Image
import io

router = APIRouter(tags=["Similar"])

@router.get("/{video_id}/{keyframe_id}", response_model=SimilarResponse)
def similar_keyframes(video_id: str, keyframe_id: str,
                      http_request: Request,
                      model: SimilarModel = Query(DEFAULT_MODEL, description="Model để so sánh"),
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

@router.post("/upload", response_model=SimilarResponse)
async def similar_by_image(http_request: Request,
                           file: UploadFile = File(...),
                           model: SimilarModel = Query(DEFAULT_MODEL, description="Model để so sánh"),
                           limit: int = 100) -> SimilarResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File uploaded is not an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, detail=f"Error reading image: {e}")

    # calling keyframe/search.py: def similar_by_image(self, image: Image.Image, k: int = 20) -> list[SearchHit]
    pool, searcher = _searcher(http_request, "keyframe", model)
    try:
        hits = pool.submit(searcher.similar_by_image, image, limit).result()
    except FileNotFoundError as e:
        raise HTTPException(503, f"embeddings for model '{model}' not on disk: {e}")

    results = [
        Item(
            keyframe_id=h.keyframe.keyframe_id, video_id=h.keyframe.video_id,
            timestamp_ms=h.keyframe.timestamp_ms, frame_idx=h.keyframe.frame_idx,
            video_fps=h.video_fps, score=h.score,
        )
        for h in hits
    ]
    return SimilarResponse(video_id="", keyframe_id="", model=model, total=len(results), results=results)