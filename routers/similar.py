from fastapi import APIRouter, HTTPException, Request, Query, UploadFile, File
from .query import _searcher
from config import DEFAULT_MODEL, DEFAULT_VISUAL_MODEL
from schemas import Item, KeyframeModel, VisualModel, SimilarResponse
from PIL import Image
import io

router = APIRouter(tags=["Similar"])


def _items(hits) -> list[Item]:
    return [
        Item(
            keyframe_id=h.keyframe.keyframe_id, video_id=h.keyframe.video_id,
            timestamp_ms=h.keyframe.timestamp_ms, frame_idx=h.keyframe.frame_idx,
            video_fps=h.video_fps, score=h.score,
        )
        for h in hits
    ]


def _by_keyframe(http_request: Request, mode: str, model: str,
                 video_id: str, keyframe_id: str, limit: int) -> SimilarResponse:
    pool, searcher = _searcher(http_request, mode, model)
    try:
        hits = pool.submit(searcher.similar, video_id, keyframe_id, limit).result()
    except FileNotFoundError as e:
        raise HTTPException(503, f"embeddings for model '{model}' not on disk: {e}")
    except (KeyError, ValueError, IndexError):
        raise HTTPException(404, f"keyframe {video_id}/{keyframe_id} has no '{model}' embedding")

    results = _items(hits)
    return SimilarResponse(video_id=video_id, keyframe_id=keyframe_id, model=model,
                           total=len(results), results=results)


async def _by_image(http_request: Request, mode: str, model: str,
                    file: UploadFile, limit: int) -> SimilarResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, detail="File uploaded is not an image")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, detail=f"Error reading image: {e}")

    pool, searcher = _searcher(http_request, mode, model)
    hits = pool.submit(searcher.similar_by_image, image, limit).result()

    results = _items(hits)
    return SimilarResponse(video_id="", keyframe_id="", model=model,
                           total=len(results), results=results)


# --- semantic: CLIP-style models, same space as text queries ---

@router.get("/semantic/{video_id}/{keyframe_id}", response_model=SimilarResponse)
def similar_semantic(video_id: str, keyframe_id: str, http_request: Request,
                     model: KeyframeModel = Query(DEFAULT_MODEL, description="Model để so sánh"),
                     limit: int = 100) -> SimilarResponse:
    return _by_keyframe(http_request, "keyframe", model, video_id, keyframe_id, limit)


@router.post("/semantic/upload", response_model=SimilarResponse)
async def similar_semantic_by_image(http_request: Request,
                                    file: UploadFile = File(...),
                                    model: KeyframeModel = Query(DEFAULT_MODEL, description="Model để so sánh"),
                                    limit: int = 100) -> SimilarResponse:
    return await _by_image(http_request, "keyframe", model, file, limit)


# --- visual: DINOv3, pure image similarity (layout, texture, near-duplicates) ---

@router.get("/visual/{video_id}/{keyframe_id}", response_model=SimilarResponse)
def similar_visual(video_id: str, keyframe_id: str, http_request: Request,
                   model: VisualModel = Query(DEFAULT_VISUAL_MODEL, description="Model để so sánh"),
                   limit: int = 100) -> SimilarResponse:
    return _by_keyframe(http_request, "visual", model, video_id, keyframe_id, limit)


@router.post("/visual/upload", response_model=SimilarResponse)
async def similar_visual_by_image(http_request: Request,
                                  file: UploadFile = File(...),
                                  model: VisualModel = Query(DEFAULT_VISUAL_MODEL, description="Model để so sánh"),
                                  limit: int = 100) -> SimilarResponse:
    return await _by_image(http_request, "visual", model, file, limit)