from __future__ import annotations

import re

KEYFRAMES_PER_VIDEO = 10_000
VIDEOS_PER_GROUP = 1_000
GROUPS_PER_PREFIX = 1_000

_VIDEO_ID_RE = re.compile(r"^([LSNM])(\d+)[-_]V(\d+)$")

_PREFIXES = {
    "L": (0, "L{group:02d}_V{num:03d}"), # example: L21_V001
    "S": (1, "S{group:02d}-V{num:03d}"), # example: S01-V001
    "N": (2, "N{group:03d}-V{num:03d}"), # example: N001-V001
    "M": (3, "M{group:02d}_V{num:03d}"), # example: M01_V001
}

_FORMATS = {code: fmt for code, fmt in _PREFIXES.values()}


def video_head(video_id: str) -> int:
    """Pack a video_id into the per-video half of the vector id."""
    match = _VIDEO_ID_RE.match(video_id)
    if match is None:
        raise ValueError(f"video_id {video_id!r} is not in <L|S|N><group>_V<num> form")
    prefix, group, num = match.group(1), int(match.group(2)), int(match.group(3))
    if group >= GROUPS_PER_PREFIX:
        raise ValueError(f"group number {group} exceeds {GROUPS_PER_PREFIX - 1}")
    if num >= VIDEOS_PER_GROUP:
        raise ValueError(f"video number {num} exceeds {VIDEOS_PER_GROUP - 1}")
    return (_PREFIXES[prefix][0] * GROUPS_PER_PREFIX + group) * VIDEOS_PER_GROUP + num


def video_id_from_key(key: int) -> str:
    """Inverse of video_head: 21001 -> L21_V001"""
    head, num = divmod(int(key), VIDEOS_PER_GROUP)
    code, group = divmod(head, GROUPS_PER_PREFIX)
    return _FORMATS[code].format(group=group, num=num)


def vector_id(video_id: str, keyframe_id: str) -> int:
    """Pack (video_id, keyframe_id) into the int64 id used by the vector index."""
    #     L21_V001 / "0001"  ->  21001_0001  ->  210010001
    keyframe = int(keyframe_id)
    if not 0 <= keyframe < KEYFRAMES_PER_VIDEO:
        raise ValueError(f"keyframe_id {keyframe_id!r} exceeds {KEYFRAMES_PER_VIDEO - 1}")
    return video_head(video_id) * KEYFRAMES_PER_VIDEO + keyframe


def split_vector_id(vid: int) -> tuple[str, str]:
    """Inverse of vector_id: int64 -> (video_id, keyframe_id)."""
    head, keyframe = divmod(int(vid), KEYFRAMES_PER_VIDEO)
    return video_id_from_key(head), f"{keyframe:04d}"