from __future__ import annotations
from pathlib import Path
from db import MetadataDatabase
from fts import FUZZY_CANDIDATES, build_fuzzy_match, build_match, rank_fuzzy, trigrams
from .search import TranscriptHit


class TranscriptExactSearcher:
    def __init__(self, db_path: Path):
        self.db = MetadataDatabase(db_path)

    def search(self, query: str, k: int = 100, fuzzy: bool = True) -> list[TranscriptHit]:
        """
        fuzzy=False: the exact phrase, adjacent and in order.
        fuzzy=True: the words in any order, tolerating typos and missing accents
        """
        rows = self._fuzzy_rows(query, k) if fuzzy else self._exact_rows(query, k)
        if not rows:
            return []

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

    def _exact_rows(self, query: str, k: int):
        match = build_match(query)
        return self.db.transcripts.search(match, k) if match else []

    def _fuzzy_rows(self, query: str, k: int):
        grams = trigrams(query)
        match = build_fuzzy_match(grams)
        if match is None:
            return []
        candidates = self.db.transcripts.fuzzy_search(match, max(FUZZY_CANDIDATES, k))
        return rank_fuzzy(candidates, query, k)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "TranscriptExactSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
