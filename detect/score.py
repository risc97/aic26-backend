from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import nms

from config import DETECT_PROMPT
from .store import PAD_OBJECTNESS

REGIONS = {
    "left":   lambda cx, cy: cx < 0.40,
    "right":  lambda cx, cy: cx > 0.60,
    "center": lambda cx, cy: (cx > 0.30) & (cx < 0.70),
    "top":    lambda cx, cy: cy < 0.40,
    "bottom": lambda cx, cy: cy > 0.60,
}


@dataclass(frozen=True)
class ObjectQuery:
    phrase: str
    min_count: int = 1
    min_score: float = 0.25
    region: str | None = None
    min_area: float = 0.0
    max_area: float = 1.0


def prompts(objects) -> list[str]:
    return [DETECT_PROMPT.format(o.phrase.strip().lower()) for o in objects]


def geometry_mask(box, aux, objects) -> torch.Tensor:
    """(m, K, P): boxes eligible for each object, before the phrase is involved."""
    cx, cy = (box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2
    area = ((box[..., 2] - box[..., 0]).clamp(min=0)
            * (box[..., 3] - box[..., 1]).clamp(min=0))
    valid = aux[..., 2] > PAD_OBJECTNESS
    masks = []
    for obj in objects:
        m = valid & (area >= obj.min_area) & (area <= obj.max_area)
        if obj.region in REGIONS:
            m = m & REGIONS[obj.region](cx, cy)
        masks.append(m)
    return torch.stack(masks, dim=-1)


def evaluate(emb, box, aux, queries, objects):
    """OWLv2's own logit, restored from the stored per-patch shift and scale."""
    probs = torch.einsum("mkd,dq->mkq", emb, queries)
    probs = torch.sigmoid((probs + aux[..., 0:1]) * aux[..., 1:2])
    thresh = torch.tensor([o.min_score for o in objects])
    return probs, (probs >= thresh) & geometry_mask(box, aux, objects)


def nms_counts(box, probs, ok, iou: float = 0.5) -> torch.Tensor:
    m, _, P = probs.shape
    counts = torch.zeros(m, P, dtype=torch.int32)
    for i in range(m):
        for q in range(P):
            sel = ok[i, :, q].nonzero(as_tuple=True)[0]
            if len(sel):
                counts[i, q] = len(nms(box[i][sel], probs[i, sel, q], iou))
    return counts


def verdict(box, probs, ok, objects, nms_iou: float = 0.5):
    """(m,) does this keyframe satisfy the stage, and (m,) how strongly."""
    need = torch.tensor([o.min_count for o in objects], dtype=torch.int32)
    counts = ok.sum(dim=1).to(torch.int32)     # pre-NMS: over-counts, never under
    passes = (counts >= need).all(dim=-1)
    if passes.any() and bool((need > 1).any()):
        # one object can fire on two patches, so only counts that must exceed
        # one are worth the per-frame NMS; it can only ever remove rows
        idx = passes.nonzero(as_tuple=True)[0]
        refined = nms_counts(box[idx], probs[idx], ok[idx], nms_iou)
        passes[idx] = (refined >= need).all(dim=-1)
    return passes, (probs * ok).amax(dim=1).mean(dim=-1)
