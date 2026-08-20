from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from db import Keyframe, MetadataDatabase
from .match import build_match


@dataclass
class OcrHit:
    rank: int
    score: float
    text: str
    keyframe: Keyframe
    video_fps: float = 0.0


class OcrSearcher:
    def __init__(self, db_path: Path):
        self.db = MetadataDatabase(db_path)

    def search(self, query: str, k: int = 100, phrase: bool = False) -> list[OcrHit]:
        match = build_match(query, phrase)
        if match is None:
            return []
        rows = self.db.ocr.search(match, k)
        if not rows:
            return []

        keys = [(o.video_id, o.keyframe_id) for o, _ in rows]
        found = self.db.keyframes.get_many(keys)
        by_key = {(kf.video_id, kf.keyframe_id): kf for kf in found}

        fps_cache: dict[str, float] = {}

        def video_fps(video_id: str) -> float:
            if video_id not in fps_cache:
                video = self.db.videos.get(video_id)
                fps_cache[video_id] = video.fps if video else 0.0
            return fps_cache[video_id]

        hits = []
        for ocr, score in rows:
            keyframe = by_key.get((ocr.video_id, ocr.keyframe_id))
            if keyframe is None:
                continue
            hits.append(OcrHit(len(hits) + 1, score, ocr.text, keyframe,
                               video_fps(keyframe.video_id)))
        return hits

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "OcrSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
