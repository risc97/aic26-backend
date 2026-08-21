from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from db import MetadataDatabase
from media.response import file_response
from config import METADATA_DB_PATH
from schemas import Item, KeyframeListResponse

router = APIRouter(tags=["Keyframe"])

IMAGE_MAX_AGE = 60 * 60 * 24 * 30

@router.get("/{video_id}/keyframes", response_model=KeyframeListResponse)
def list_keyframes(
    video_id: str,
    start_ms: int | None = Query(None, ge=0, description="Lấy keyframe từ mốc này"),
    end_ms: int | None = Query(None, ge=0, description="Lấy keyframe đến mốc này"),
) -> KeyframeListResponse:
    with MetadataDatabase(METADATA_DB_PATH) as db:
        video = db.videos.get(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        keyframes = db.keyframes.list_by_video(video_id, start_ms, end_ms)

    items = [
        Item(
            keyframe_id=kf.keyframe_id, video_id=kf.video_id,
            timestamp_ms=kf.timestamp_ms, frame_idx=kf.frame_idx, video_fps=video.fps,
        )
        for kf in keyframes
    ]
    return KeyframeListResponse(video_id=video_id, total=len(items), keyframes=items)


@router.api_route(
    "/{video_id}/{keyframe_id}",
    methods=["GET", "HEAD"],
    response_class=FileResponse,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "Not found"}},
)
def get_keyframe(video_id: str, keyframe_id: str) -> FileResponse:
    with MetadataDatabase(METADATA_DB_PATH) as db:
        keyframe = db.keyframes.get(video_id, keyframe_id)
        if keyframe is None:
            raise HTTPException(
                status_code=404, detail=f"Keyframe {video_id}/{keyframe_id} not found"
            )

    return file_response(keyframe.image_path, max_age=IMAGE_MAX_AGE)
