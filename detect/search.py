from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import torch
from torchvision.ops import nms
import numpy as np
from keyframe.ids import split_vector_id, vector_id
from db import Keyframe, MetadataDatabase
from config import DETECT_PROMPT, DETECT_REPOS, DETECT_SHARD_DIRS
from .store import PAD_OBJECTNESS, DetectionStore
from .text import Owlv2TextEncoder
from .corpus import StageCache
from .score import ObjectQuery, evaluate, geometry_mask

@dataclass
class DetectHit:
    rank: int
    score: float
    keyframe: Keyframe
    video_fps: float
    counts: list[int] = field(default_factory=list)
    boxes: list[list[tuple[float, float, float, float, float]]] = field(default_factory=list)

class DetectSearcher:
    def __init__(self, shard_dir: Path, db_path: Path, model: str = "owlv2-base",
                 ckpt: Path | None = None, device: str = "cuda"):
        self.store = DetectionStore(shard_dir or DETECT_SHARD_DIRS[model])
        self.encoder = Owlv2TextEncoder(DETECT_REPOS[model], device=device)
        self.db = MetadataDatabase(db_path)
        self.model_name = model
        self.stages = StageCache(self.store, self.encoder)

    def search(self, objects: list[ObjectQuery], limit: int = 100,
               candidates: list[tuple[str, str]] | None = None,
               nms_iou: float = 0.5) -> list[DetectHit]:
        if not objects:
            return []

        queries = self.encoder.encode([DETECT_PROMPT.format(o.phrase.strip().lower()) for o in objects])
        thresh = torch.tensor([o.min_score for o in objects])
        need = torch.tensor([o.min_count for o in objects])

        rows = None
        if candidates is not None:
            if not candidates:
                return []
            ids = np.array([vector_id(v, k) for v, k in candidates], dtype=np.int64)
            rows = self.store.rows_for(ids)

        kept_rows, kept_scores = [], []
        for global_rows, emb, box, aux in self.store.chunks(rows):
            probs = torch.einsum("mkd,dq->mkq", emb, queries)
            probs = torch.sigmoid((probs + aux[..., 0:1]) * aux[..., 1:2])
            ok = (probs >= thresh) & geometry_mask(box, aux, objects)

            # counts here are pre-NMS, so they over-count and never drop a
            # frame that would survive; the survivors get NMS'd exactly below
            passing = (ok.sum(dim=1) >= need).all(dim=-1)
            if not passing.any():
                continue
            idx = passing.nonzero(as_tuple=True)[0]
            best = (probs * ok).amax(dim=1)[idx]
            kept_rows.append(torch.as_tensor(global_rows)[idx])
            kept_scores.append(best.mean(dim=-1))

        if not kept_rows:
            return []

        all_rows = torch.cat(kept_rows).numpy()
        order = torch.cat(kept_scores).argsort(descending=True).numpy()
        # refine a generous slice: NMS only removes frames, never adds them
        short = all_rows[order[:max(limit * 8, limit)]]

        hits: list[DetectHit] = []
        for global_rows, emb, box, aux in self.store.chunks(short):
            probs = torch.einsum("mkd,dq->mkq", emb, queries)
            probs = torch.sigmoid((probs + aux[..., 0:1]) * aux[..., 1:2])
            ok = (probs >= thresh) & geometry_mask(box, aux, objects)
            for i, row in enumerate(global_rows):
                counts, per_object, total = [], [], 0.0
                for q in range(len(objects)):
                    sel = ok[i, :, q].nonzero(as_tuple=True)[0]
                    if len(sel) == 0:
                        counts.append(0)
                        per_object.append([])
                        continue
                    b, s = box[i][sel], probs[i, sel, q]
                    keep = nms(b, s, nms_iou)
                    counts.append(len(keep))
                    total += float(s.max())
                    per_object.append([
                        (*[round(float(v), 4) for v in b[j]], round(float(s[j]), 4))
                        for j in keep
                    ])
                if any(c < objects[q].min_count for q, c in enumerate(counts)):
                    continue
                video_id, keyframe_id = split_vector_id(int(self.store.ids[row]))
                hits.append((total / len(objects), video_id, keyframe_id,
                             counts, per_object))

        hits.sort(key=lambda h: -h[0])
        return self._resolve(hits[:limit])

    def _resolve(self, rows) -> list[DetectHit]:
        found = self.db.keyframes.get_many([(v, k) for _, v, k, _, _ in rows])
        by_key = {(kf.video_id, kf.keyframe_id): kf for kf in found}
        fps_cache: dict[str, float] = {}

        def video_fps(video_id: str) -> float:
            if video_id not in fps_cache:
                video = self.db.videos.get(video_id)
                fps_cache[video_id] = video.fps if video else 0.0
            return fps_cache[video_id]

        out = []
        for score, video_id, keyframe_id, counts, boxes in rows:
            keyframe = by_key.get((video_id, keyframe_id))
            if keyframe is None:
                continue
            out.append(DetectHit(len(out) + 1, score, keyframe,
                                 video_fps(video_id), counts, boxes))
        return out

    def close(self) -> None:
        self.db.close()