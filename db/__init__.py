from __future__ import annotations

from pathlib import Path

from .connection import connect, transaction
from .models import Keyframe, Segment, Transcript, Video
from .schema import init_schema
from .segments import SegmentRepo
from .keyframes import KeyframeRepo
from .transcripts import TranscriptRepo
from .videos import VideoRepo


class MetadataDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = connect(db_path)
        self.videos = VideoRepo(self.conn)
        self.segments = SegmentRepo(self.conn)
        self.keyframes = KeyframeRepo(self.conn)
        self.transcripts = TranscriptRepo(self.conn)

    def init_schema(self) -> None:
        init_schema(self.conn)

    def transaction(self):
        return transaction(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MetadataDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["MetadataDatabase", "Video", "Segment", "Keyframe", "Transcript"]
