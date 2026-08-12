from __future__ import annotations
import re

KEYFRAMES_PER_VIDEO = 10_000
VIDEOS_PER_GROUP = 1_000

_VIDEO_ID_RE = re.compile(r"^L(\d+)_V(\d+)$")


def vector_id(video_id: str, keyframe_id: str) -> int:
    """Pack (video_id, keyframe_id) into the int64 id used by the vector index."""
    #     L21_V001 / "0001"  ->  21001_0001  ->  210010001
    match = _VIDEO_ID_RE.match(video_id)
    if match is None:
        raise ValueError(f"video_id {video_id!r} is not in L<group>_V<num> form")
    group, num = int(match.group(1)), int(match.group(2))
    keyframe = int(keyframe_id)
    if num >= VIDEOS_PER_GROUP:
        raise ValueError(f"video number {num} exceeds {VIDEOS_PER_GROUP - 1}")
    if not 0 <= keyframe < KEYFRAMES_PER_VIDEO:
        raise ValueError(f"keyframe_id {keyframe_id!r} exceeds {KEYFRAMES_PER_VIDEO - 1}")
    return (group * VIDEOS_PER_GROUP + num) * KEYFRAMES_PER_VIDEO + keyframe


def split_vector_id(vid: int) -> tuple[str, str]:
    """Inverse of vector_id: int64 -> (video_id, keyframe_id)."""
    head, keyframe = divmod(int(vid), KEYFRAMES_PER_VIDEO)
    group, num = divmod(head, VIDEOS_PER_GROUP)
    return f"L{group:02d}_V{num:03d}", f"{keyframe:04d}"
