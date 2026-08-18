from __future__ import annotations
import re

TRANSCRIPTS_PER_VIDEO = 10_000
VIDEOS_PER_GROUP = 1_000

_VIDEO_ID_RE = re.compile(r"^L(\d+)_V(\d+)$")


def vector_id(video_id: str, transcript_id: str) -> int:
    """Pack (video_id, transcript_id) into the int64 id used by the vector index."""
    match = _VIDEO_ID_RE.match(video_id)
    if match is None:
        raise ValueError(f"video_id {video_id!r} is not in L<group>_V<num> form")
    group, num = int(match.group(1)), int(match.group(2))
    transcript = int(transcript_id)
    if num >= VIDEOS_PER_GROUP:
        raise ValueError(f"video number {num} exceeds {VIDEOS_PER_GROUP - 1}")
    if not 0 <= transcript < TRANSCRIPTS_PER_VIDEO:
        raise ValueError(f"transcript_id {transcript_id!r} exceeds {TRANSCRIPTS_PER_VIDEO - 1}")
    return (group * VIDEOS_PER_GROUP + num) * TRANSCRIPTS_PER_VIDEO + transcript


def split_vector_id(vid: int) -> tuple[str, str]:
    """Inverse of vector_id: int64 -> (video_id, transcript_id)."""
    head, transcript = divmod(int(vid), TRANSCRIPTS_PER_VIDEO)
    group, num = divmod(head, VIDEOS_PER_GROUP)
    return f"L{group:02d}_V{num:03d}", f"{transcript:04d}"
