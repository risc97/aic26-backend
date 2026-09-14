"""Temporal search: find videos where a sequence of events (q1, q2, ... qn) happens in order.

Search the first query q1. Its hits decide which videos to
look at and where in each one to start; every later query only ever looks forward in time from there.

    1. seeds  -- q1 retrieves its top r keyframes (S_R).
    2. pool   -- in those videos, every keyframe after a seed becomes a
                 candidate (S_C), so ordering is baked into the pool itself.
    3. score  -- q2..qn are searched against S_C in one batched pass.
    4. chain  -- from each seed, walk forward taking each query's best-ranked
                 keyframe that comes after the previous hop.
    5. rank   -- longest chain wins; RRF over the chain's ranks breaks ties.
"""
from __future__ import annotations

from typing import Callable, Sequence
import numpy as np
from .candidates import SENTINEL, build_pool, score_pool
from .chains import build_hits, walk
from .types import Stage, TemporalParams


def temporal_search(
    index,
    db,
    encode: Callable[[list[str]], np.ndarray],
    stages: Sequence[Stage],
    params: TemporalParams | None = None,
) -> list[TemporalHit]:

    params = params or TemporalParams()
    n_stages = len(stages)
    if n_stages < 2:
        raise ValueError("temporal search needs at least 2 stages")
    
    labels = [s if isinstance(s, str) else list(s)[0] for s in stages]
    queries = np.ascontiguousarray(encode(labels), dtype=np.float32)

    # retrieve S_R

    # Drop the variant for now
    scores, ids = index.search(queries[:1], k=params.r)
    scores, ids = np.atleast_2d(scores)[0], np.atleast_2d(ids)[0]
    keep = (ids != SENTINEL)
    scores, ids = scores[keep], ids[keep].astype(np.uint64)

    if scores.size == 0:
        return []

    # build pool
    pool = build_pool(index, db, ids, scores, n_stages)
    if pool is None:
        return []

    # score q2..qn over the pool
    score_pool(index, queries, pool)

    # chain + rank
    def make_chain(seed, video_id, spans):
        return walk(pool, video_id, seed, params, n_stages, spans)

    return build_hits(pool, db, params, n_stages, labels, make_chain)
