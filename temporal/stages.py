"""Turning a list of stages into scored, per-video candidate blocks."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from db import Keyframe
from keyframe.ids import vector_id, video_id_from_key, video_key
from .types import Stage, TemporalParams

NEG = -1e18
SENTINEL = int(np.iinfo(np.uint64).max)  # turbovec pads short result rows with this


def flatten(stages: Sequence[Stage]) -> tuple[list[str], np.ndarray]:
    """Stages (each possibly several paraphrases) -> flat texts + owning stage."""
    texts: list[str] = []
    owner: list[int] = []
    for i, stage in enumerate(stages):
        variants = [stage] if isinstance(stage, str) else [v for v in stage if v.strip()]
        if not variants:
            raise ValueError(f"stage {i} has no query text")
        texts.extend(variants)
        owner.extend([i] * len(variants))
    return texts, np.asarray(owner, dtype=np.int64)


def pool(rows: np.ndarray, owner: np.ndarray, n_stages: int) -> np.ndarray:
    """(R, C) per-variant scores -> (M, C), max over each stage's variants.

    Max, not mean: a paraphrase that misses should not drag down one that hits.
    """
    out = np.full((n_stages, rows.shape[1]), NEG, dtype=np.float32)
    for r, i in enumerate(owner.tolist()):
        np.maximum(out[i], rows[r], out=out[i])
    return out


def zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rescale against what this stage typically retrieves corpus-wide.

    Raw cosine scores are not comparable across queries -- without this one
    'easy' stage dominates the sum and you are back to single-query search.
    """
    sd = float(reference.std())
    return (values - float(reference.mean())) / (sd if sd > 1e-6 else 1.0)


def squash(values: np.ndarray, cap: float) -> np.ndarray:
    """Bound what one stage can contribute to the summed score.

    The reference is the truncated top-k head, so its std is far tighter than
    the corpus std and z-scores come out inflated with a heavy right tail. Left
    unbounded, a single freak keyframe outscores three honest matches and the
    ranking stops being about the sequence at all.

    `cap * tanh(v / cap)` is the identity near zero and saturates at +-cap, so
    it is monotone: within a stage nothing is reordered, only the weight a
    runaway score carries into the sum across stages.
    """
    return cap * np.tanh(values / cap)


def candidate_videos(scores, ids, owner, n_stages: int, params: TemporalParams,
                     on_note: Callable[[str], None]) -> list[str]:
    """Videos worth an exact rescore, ranked by summed per-stage best score."""
    best_per_stage: list[dict[int, float]] = [{} for _ in range(n_stages)]
    for r in range(ids.shape[0]):
        best = best_per_stage[int(owner[r])]
        for score, vid in zip(scores[r].tolist(), ids[r].tolist()):
            if int(vid) == SENTINEL:  # padding on a short result row
                continue
            key = video_key(vid)
            if score > best.get(key, -np.inf):
                best[key] = score

    stage_hits: dict[int, int] = {}
    prior: dict[int, float] = {}
    for best in best_per_stage:
        for key, score in best.items():
            stage_hits[key] = stage_hits.get(key, 0) + 1
            prior[key] = prior.get(key, 0.0) + score

    # tolerate one stage missing entirely -- vague queries do fail outright, and
    # this filter runs on truncated top-k evidence. Enforcing "every stage must
    # appear" belongs in skip_penalty, which sees the exact scores instead.
    floor = params.min_stages if params.min_stages is not None else max(1, n_stages - 1)
    kept = [key for key, n in stage_hits.items() if n >= floor]
    kept.sort(key=lambda key: prior[key], reverse=True)
    if len(kept) > params.max_videos:
        on_note(f"temporal: {len(kept)} candidates -> keeping top {params.max_videos}")
        kept = kept[:params.max_videos]
    return [video_id_from_key(key) for key in kept]


def build_allowlist(index, videos: Sequence[str], frames: dict[str, list[Keyframe]]):
    """Packed ids for every embedded candidate keyframe, grouped per video.

    Returns (allowlist, {video_id: (col_start, col_end, keyframes)}). Columns are
    contiguous per video so the DP can slice its block straight out of the score
    matrix. The `contains` filter is not optional: an allowlist id missing from
    the index makes turbovec raise KeyError for the whole request.
    """
    allow: list[int] = []
    layout: dict[str, tuple[int, int, list[Keyframe]]] = {}
    for video_id in videos:
        start, kept = len(allow), []
        for kf in frames.get(video_id, ()):
            try:
                packed = vector_id(kf.video_id, kf.keyframe_id)
            except ValueError:
                continue
            if not index.contains(packed):  # in the db but never embedded
                continue
            allow.append(packed)
            kept.append(kf)
        if len(kept) >= 2:  # a sequence needs at least two distinct frames
            layout[video_id] = (start, len(allow), kept)
        else:
            del allow[start:]
    return np.asarray(allow, dtype=np.uint64), layout
