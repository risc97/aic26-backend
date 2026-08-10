from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from config import MEDIA_ROOT


def resolve_media_path(stored_path: str) -> Path:
    path = Path(stored_path)
    if not path.is_absolute():
        path = MEDIA_ROOT / path

    root = MEDIA_ROOT.resolve()
    path = path.resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=404, detail="Media not found")
    return path


def file_response(stored_path: str, *, max_age: int) -> FileResponse:
    path = resolve_media_path(stored_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file missing on disk")

    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        headers={"Cache-Control": f"public, max-age={max_age}"},
    )
