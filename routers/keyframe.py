from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from db import MetadataDatabase
from media.response import file_response
from config import METADATA_DB_PATH

router = APIRouter(tags=["Keyframe"])

IMAGE_MAX_AGE = 60 * 60 * 24 * 30


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
