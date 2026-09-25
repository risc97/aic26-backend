from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from db import Keyframe, MetadataDatabase
from fts import FUZZY_CANDIDATES, build_fuzzy_match, build_match, rank_fuzzy, trigrams


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

    def search(self, query: str, k: int = 100, fuzzy: bool = True) -> list[OcrHit]:
        """
        fuzzy=False: search exact phrase, adjacent and in order.
        fuzzy=True: search fuzzy: the words in any order, tolerating typos, missing accents and OCR misreads."""
        rows = self._fuzzy_rows(query, k) if fuzzy else self._exact_rows(query, k)
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

    def _exact_rows(self, query: str, k: int):
        match = build_match(query)
        return self.db.ocr.search(match, k) if match else []

    def _fuzzy_rows(self, query: str, k: int):
        grams = trigrams(query)
        match = build_fuzzy_match(grams)
        if match is None:
            return []
        candidates = self.db.ocr.fuzzy_search(match, max(FUZZY_CANDIDATES, k))
        return rank_fuzzy(candidates, query, k)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "OcrSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
