from __future__ import annotations

from pathlib import Path

KEYFRAME_SUFFIX = ".jpg"


def find_keyframes(keyframes_dir: Path, video_id: str) -> list[Path]:
    kf_dir = keyframes_dir / video_id
    if not kf_dir.is_dir():
        return []
    return sorted(kf_dir.glob(f"*{KEYFRAME_SUFFIX}"))

def list_keyframe_ids(keyframes_dir: Path, video_id: str) -> list[str]:
    return [p.stem for p in find_keyframes(keyframes_dir, video_id)]