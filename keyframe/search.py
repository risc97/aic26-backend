from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from models import MODEL_CHOICES
from db import Keyframe, MetadataDatabase
from .ids import split_vector_id


@dataclass
class SearchHit:
    rank: int
    score: float
    keyframe: Keyframe


def format_timestamp(ms: int) -> str:
    seconds, ms = divmod(int(ms), 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{ms // 100}"


class KeyframeSearcher:
    def __init__(self, index_path: Path, db_path: Path, model: str = "siglip",
                 ckpt: Path | None = None, device: str = "cuda"):
        from turbovec import IdMapIndex

        self.index = IdMapIndex.load(str(index_path))
        self.db = MetadataDatabase(db_path)
        cls = MODEL_CHOICES[model]
        self.model = cls(ckpt, device="cpu")
        self._free_image_encoder()
        self.model_name = model
        self.model.to(device)
        self.model.amp = device == "cuda"

    def _free_image_encoder(self):
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "visual"):
            del inner.visual
            torch.cuda.empty_cache()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Texts -> (n, dim) L2-normalized float32"""
        return np.ascontiguousarray(self.model.encode_texts(texts), dtype=np.float32)

    def search(self, query: str, k: int = 20) -> list[SearchHit]:
        vec = self.encode_texts([query])
        scores, ids = self.index.search(vec, k=k)
        # a 1-row query may come back 1D or 2D depending on the build
        scores, ids = np.atleast_2d(scores)[0], np.atleast_2d(ids)[0]

        keys = [split_vector_id(int(i)) for i in ids if int(i) >= 0]
        found = self.db.keyframes.get_many(keys)
        by_key = {(kf.video_id, kf.keyframe_id): kf for kf in found}

        hits = []
        for score, key in zip(scores, keys):
            keyframe = by_key.get(key)
            if keyframe is None:
                continue
            hits.append(SearchHit(len(hits) + 1, float(score), keyframe))
        return hits

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "KeyframeSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
