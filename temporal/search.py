"""Temporal search: match an ordered sequence of vague queries to one video.

Any single vague query retrieves mostly noise. The disambiguating signal is the
*order* of the events, so scoring happens per video over the whole sequence at
once, never per keyframe in isolation.

    1. recall   -- wide top-k ANN per stage, only to decide which videos deserve
                   a closer look.
    2. rescore  -- score every candidate keyframe against every stage via the
                   index allowlist. This removes the top-k truncation bias that
                   otherwise sinks vague queries.
    3. align    -- per video, a DP for the best strictly-increasing-in-time
                   assignment of stages to keyframes.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .dp import align
from .stages import (SENTINEL, build_allowlist, candidate_videos, flatten, pool,
                     squash, zscore)
from .types import Stage, StageMatch, TemporalHit, TemporalParams


def temporal_search(
    index,
    db,
    encode: Callable[[list[str]], np.ndarray],
    stages: Sequence[Stage],
    *,
    k: int = 20,
    params: TemporalParams | None = None,
    on_note: Callable[[str], None] = print,
) -> list[TemporalHit]:
    """Rank videos by how well they contain `stages` in order.

    index  -- a turbovec IdMapIndex over keyframe embeddings
    db     -- a MetadataDatabase
    encode -- texts -> (n, dim) L2-normalized float32, from the matching encoder
    """
    params = params or TemporalParams()
    n_stages = len(stages)
    if n_stages < 2:
        raise ValueError("temporal search needs at least 2 stages")

    texts, owner = flatten(stages)
    labels = [s if isinstance(s, str) else list(s)[0] for s in stages]
    queries = np.ascontiguousarray(encode(texts), dtype=np.float32)

    # --- 1. recall: wide on purpose, precision comes later ------------------
    raw_scores, raw_ids = index.search(queries, k=params.candidate_k)
    raw_scores, raw_ids = np.atleast_2d(raw_scores), np.atleast_2d(raw_ids)

    videos = candidate_videos(raw_scores, raw_ids, owner, n_stages, params, on_note)
    if not videos:
        return []

    # --- 2. exact rescore over the candidates -------------------------------
    frames = db.keyframes.list_by_videos(videos)
    allowlist, layout = build_allowlist(index, videos, frames)
    if not layout:
        return []

    hit_scores, hit_ids = index.search(queries, k=len(allowlist), allowlist=allowlist)
    hit_scores, hit_ids = np.atleast_2d(hit_scores), np.atleast_2d(hit_ids)

    # allowlist rows come back score-sorted; scatter them into allowlist order
    column_of = {int(v): c for c, v in enumerate(allowlist.tolist())}
    dense = np.zeros((len(texts), allowlist.shape[0]), dtype=np.float32)
    for r in range(dense.shape[0]):
        cols = np.fromiter((column_of[int(v)] for v in hit_ids[r].tolist()),
                           dtype=np.int64, count=hit_ids.shape[1])
        dense[r, cols] = hit_scores[r]

    scores = pool(dense, owner, n_stages)
    for i in range(n_stages):
        rows = owner == i
        reference = raw_scores[rows][raw_ids[rows] != SENTINEL]
        scores[i] = zscore(scores[i], reference if reference.size else raw_scores[rows])
        if params.score_cap is not None:
            # keep one runaway keyframe from outweighing the rest of the sequence
            scores[i] = squash(scores[i], params.score_cap)

    # --- 3. align each candidate video --------------------------------------
    fps_cache: dict[str, float] = {}

    def video_fps(video_id: str) -> float:
        if video_id not in fps_cache:
            video = db.videos.get(video_id)
            fps_cache[video_id] = video.fps if video else 0.0
        return fps_cache[video_id]

    span = params.span_penalty / 60_000.0
    ranked = []
    for video_id, (lo, hi, kfs) in layout.items():
        block = scores[:, lo:hi]
        ts = np.asarray([kf.timestamp_ms for kf in kfs], dtype=np.float64)
        total, path = align(
            block, ts, params.gap_for,
            skip_penalty=params.skip_penalty,
            span_penalty_ms=span,
            max_skips=params.max_skips,
            beta=params.beta,
        )
        steps = path.tolist()
        matches = [
            StageMatch(i, labels[i], float(block[i, j]), kfs[j], video_fps(video_id))
            for i, j in enumerate(steps) if j >= 0
        ]
        if not matches:
            continue
        ranked.append((total, video_id, matches,
                       [i for i, j in enumerate(steps) if j < 0]))

    ranked.sort(key=lambda row: -row[0])
    return [
        TemporalHit(rank, float(total), video_id, matches, skipped)
        for rank, (total, video_id, matches, skipped) in enumerate(ranked[:k], 1)
    ]
