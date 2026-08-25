from __future__ import annotations
from pathlib import Path
from db import MetadataDatabase
from fts import build_match
from .search import TranscriptHit


class TranscriptExactSearcher:
    def __init__(self, db_path: Path):
        self.db = MetadataDatabase(db_path)

    def search(self, query: str, k: int = 100, phrase: bool = False) -> list[TranscriptHit]:
        match = build_match(query, phrase)
        if match is None:
            return []
        rows = self.db.transcripts.search(match, k)

        fps_cache: dict[str, float] = {}

        def video_fps(video_id: str) -> float:
            if video_id not in fps_cache:
                video = self.db.videos.get(video_id)
                fps_cache[video_id] = video.fps if video else 0.0
            return fps_cache[video_id]

        hits = []
        for transcript, score in rows:
            keyframes = self.db.keyframes.list_by_video(
                transcript.video_id,
                start_ms=transcript.time_start_ms,
                end_ms=transcript.time_end_ms,
            )
            hits.append(TranscriptHit(
                len(hits) + 1, score, transcript, keyframes,
                video_fps(transcript.video_id),
            ))
        return hits

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "TranscriptExactSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
