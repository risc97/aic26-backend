"""Temporal search over OWLv2 detections.

The same five phases as search.py, over the same Pool, walk, dedupe and
select. One semantic difference drives every change here: a semantic stage
is a similarity and a detection stage is a constraint. Cosine is always
defined, so a semantic chain breaks only when the clock runs out; "2
motorbikes" is simply absent from most keyframes, and the chain has to
break there. That makes `length` a count of stages that genuinely hold.

walk() cannot express "nothing here", so failures keep the sentinel rank
build_pool wrote and the chain is cut afterwards -- see _truncate.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from detect.score import ObjectQuery
from .candidates import build_pool
from .chains import _span, build_hits, walk
from .types import Chain, TemporalParams

DetectStage = Sequence[ObjectQuery]


class _Contains:
    """build_pool only ever asks the index whether an id has vectors at all."""

    def __init__(self, store):
        self.store = store

    def contains(self, packed: int) -> bool:
        return bool(self.store.rows_for(np.array([packed], dtype=np.int64))[0] >= 0)


def _seeds(store, passes, scores, r: int):
    """Phase 1: the top r keyframes satisfying stage 0, and their S_R ranks."""
    ids, sc = store.ids[passes], scores[passes]
    order = np.argsort(-sc, kind="stable")[:r]
    return ids[order].astype(np.uint64), sc[order].astype(np.float32)


def _fill_pool(store, cache, stages, pool, nms_iou: float, seed: int) -> None:
    rows = store.rows_for(pool.allowlist.astype(np.int64))
    for stage, objects in enumerate(stages):
        if stage == seed:
            continue
        passes, scores = cache.get(objects, nms_iou)
        hit = np.flatnonzero(passes[rows])
        pool.scores[stage, hit] = scores[rows[hit]]
        order = hit[np.argsort(-pool.scores[stage, hit], kind="stable")]
        pool.ranks[stage, order] = np.arange(1, len(order) + 1)


def _truncate(chain: Chain, params: TemporalParams, spans, sentinel: int) -> Chain:
    """Cut a chain back to the run of satisfied stages around its seed.

    walk() is greedy and one-directional per side: it sets `cur` from whatever it
    just took, so every hop past an unsatisfied one descends from a hop that should
    not exist. Dropping both tails is what breaking inside walk() would have
    produced. The seed hop always holds.
    """
    hops = chain.hops
    pivot = next(i for i, h in enumerate(hops) if h.stage == params.seed)
    lo = hi = pivot
    while hi + 1 < len(hops) and hops[hi + 1].rank <= sentinel:
        hi += 1
    while lo - 1 >= 0 and hops[lo - 1].rank <= sentinel:
        lo -= 1
    if lo == 0 and hi == len(hops) - 1:
        return chain
    hops = hops[lo:hi + 1]
    rrf = sum(params.weight(h.stage) / (params.rrf_k + h.rank) for h in hops)
    start = _span(hops[0], spans)[0]
    end = _span(hops[-1], spans)[1]
    return Chain(chain.video_id, hops, rrf, start, max(end, start + 1))


def label(objects: DetectStage) -> str:
    return " + ".join(f"{o.min_count}x {o.phrase}" for o in objects)


def detect_temporal_search(
    store, cache, db,
    stages: Sequence[DetectStage],
    params: TemporalParams | None = None,
    nms_iou: float = 0.5,
) -> list[TemporalHit]:
    params = params or TemporalParams()
    n_stages = len(stages)
    if n_stages < 2:
        raise ValueError("temporal search needs at least 2 stages")
    if any(not objects for objects in stages):
        raise ValueError("every stage needs at least one object")
    if not 0 <= params.seed < n_stages:
        raise ValueError(f"seed must be a stage index in 0..{n_stages - 1}")
    stages = [tuple(objects) for objects in stages]

    passes, scores = cache.get(stages[params.seed], nms_iou)
    if not passes.any():
        return []
    ids, seed_scores = _seeds(store, passes, scores, params.r)

    pool = build_pool(_Contains(store), db, ids, seed_scores, n_stages,
                      params.seed, params.max_gap_ms)

    if pool is None:
        return []

    _fill_pool(store, cache, stages, pool, nms_iou, params.seed)

    sentinel = len(pool.allowlist)   # build_pool fills unscored ranks with n + 1
    labels = [label(objects) for objects in stages]

    def make_chain(seed, video_id, spans):
        chain = walk(pool, video_id, seed, params, n_stages, spans)
        return _truncate(chain, params, spans, sentinel)

    return build_hits(pool, db, params, n_stages, labels, make_chain)
