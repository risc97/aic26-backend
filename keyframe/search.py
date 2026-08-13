from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import sys
from models import load_encoder
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
    def __init__(self, index_path: Path, db_path: Path, model: str = "c2lip",
                 ckpt: Path | None = None, device: str = "cuda"):
        from turbovec import IdMapIndex

        self.index = IdMapIndex.load(str(index_path))
        self.db = MetadataDatabase(db_path)
        self.model_name = model
        self.model = load_encoder(model, ckpt=ckpt, device=device)

        # delete visual encoder
        del self.model.model.visual
        torch.cuda.empty_cache()

    def search(self, query: str, k: int = 20) -> list[SearchHit]:
        vec = np.ascontiguousarray(self.model.encode_texts([query]), dtype=np.float32)
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
