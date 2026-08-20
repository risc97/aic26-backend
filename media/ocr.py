from __future__ import annotations

import json
from pathlib import Path

OCR_SUFFIX = ".json"


def find_ocr(ocr_dir: Path, video_id: str) -> Path | None:
    path = ocr_dir / f"{video_id}{OCR_SUFFIX}"
    return path if path.is_file() else None


def read_ocr(ocr_file: Path) -> list[tuple[str, str]]:
    """Read {keyframe_id: text} into (keyframe_id, text), dropping empty text."""
    with open(ocr_file, encoding="utf-8") as f:
        data = json.load(f)
    return [
        (keyframe_id, text.strip())
        for keyframe_id, text in sorted(data.items())
        if text and text.strip()
    ]
