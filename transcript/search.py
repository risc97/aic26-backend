from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from models import MODEL_CHOICES
from db import MetadataDatabase, Keyframe, Transcript
from .ids import split_vector_id


@dataclass
class TranscriptHit:
    rank: int
    score: float
    transcript: Transcript
    keyframes: list[Keyframe] = field(default_factory=list)
    video_fps: float = 0.0


class TranscriptSemanticSearcher:
    def __init__(self, index_path: Path, db_path: Path, model: str = "gte",
                 ckpt: Path | None = None, device: str = "cuda"):
        from turbovec import IdMapIndex

        self.index = IdMapIndex.load(str(index_path))
        self.db = MetadataDatabase(db_path)
        cls = MODEL_CHOICES[model]
        self.model = cls(device=device)
        self.model_name = model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Texts -> (n, dim) L2-normalized float32"""
        return np.ascontiguousarray(self.model.encode_texts(texts), dtype=np.float32)

    def search(self, query: str, k: int = 20) -> list[TranscriptHit]:
        vec = self.encode_texts([query])
        scores, ids = self.index.search(vec, k=k)
        # a 1-row query may come back 1D or 2D depending on the build
        scores, ids = np.atleast_2d(scores)[0], np.atleast_2d(ids)[0]

        hits = []
        for score, vid in zip(scores, ids):
            vid = int(vid)
            if vid < 0:
                continue
            video_id, transcript_id = split_vector_id(vid)
            transcript = self.db.transcripts.get(video_id, transcript_id)
            if transcript is None:
                continue
            keyframes = self.db.keyframes.list_by_video(
                video_id,
                start_ms=transcript.time_start_ms,
                end_ms=transcript.time_end_ms,
            )
            video = self.db.videos.get(video_id)
            hits.append(TranscriptHit(
                len(hits) + 1, float(score), transcript, keyframes,
                video.fps if video else 0.0,
            ))
        return hits

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "TranscriptSemanticSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
