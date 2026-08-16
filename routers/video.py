from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from db import MetadataDatabase
from media.response import file_response
from config import METADATA_DB_PATH
router = APIRouter(tags=["Video"])

VIDEO_MAX_AGE = 60 * 60 * 24 * 7


@router.api_route(
    "/{video_id}",
    methods=["GET", "HEAD"],
    response_class=FileResponse,
    responses={200: {"content": {"video/webm": {}}}, 404: {"description": "Not found"}},
)
def get_video(video_id: str) -> FileResponse:
    with MetadataDatabase(METADATA_DB_PATH) as db:
        video = db.videos.get(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    return file_response(video.video_path, max_age=VIDEO_MAX_AGE)
