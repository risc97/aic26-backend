"""Phase 4-5: walk each seed forward into a chain, then thin and rank them."""
from __future__ import annotations

import numpy as np

from .candidates import Pool
from .types import Chain, Hop, TemporalParams


def _span(hop: Hop, spans: dict[int, tuple[int, int]]) -> tuple[int, int]:
    """The shot a hop sits in, falling back to the keyframe's own instant"""
    span = spans.get(hop.keyframe.segment_id)
    if span is None:
        return hop.keyframe.timestamp_ms, hop.keyframe.timestamp_ms
    return span


def walk(pool: Pool, video_id: str, seed: Hop, params: TemporalParams,
         n_stages: int, spans: dict[int, tuple[int, int]]) -> Chain:
    """Grow one chain forward in time from a single q1 seed.

    Each query takes its best-ranked keyframe after the previous hop and within
    max_gap_ms. The first query with nothing to take ends the chain, so a short
    chain always misses a tail of queries, never one in the middle.
    """
    lo, hi = pool.layout[video_id]
    times = pool.times[lo:hi]
    hops = [seed]
    cur = seed.keyframe.timestamp_ms

    for stage in range(1, n_stages):
        first = int(np.searchsorted(times, cur, side="right"))   # strictly after
        last = (len(times) if params.max_gap_ms is None
                else int(np.searchsorted(times, cur + params.max_gap_ms, side="right")))
        if first >= last:
            break                                                # chain ends here
        window = pool.ranks[stage, lo + first:lo + last]
        col = lo + first + int(window.argmin())                  # best rank in the window
        hops.append(Hop(stage, col, pool.keyframes[col],
                        float(pool.scores[stage, col]), int(pool.ranks[stage, col])))
        cur = int(pool.times[col])

    rrf = sum(params.weight(h.stage) / (params.rrf_k + h.rank) for h in hops)
    start_ms = _span(hops[0], spans)[0]
    end_ms = _span(hops[-1], spans)[1]
    # keep the interval non-empty so IoU never divides by zero
    return Chain(video_id, hops, rrf, start_ms, max(end_ms, start_ms + 1))


def iou(a: Chain, b: Chain) -> float:
    """Overlap between two chains' time spans"""
    inter = max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))
    union = (a.end_ms - a.start_ms) + (b.end_ms - b.start_ms) - inter
    return inter / union if union > 0 else 0.0


def dedupe(chains: list[Chain], phi: float) -> list[Chain]:
    """Drop chains covering the same stretch. Sorted first, so the longer chain survives, and RRF breaks ties."""
    kept: list[Chain] = []
    for c in sorted(chains, key=lambda c: c.key, reverse=True):
        if all(iou(c, k) <= phi for k in kept):
            kept.append(c)
    return kept


def select(by_video: dict[str, list[Chain]], params: TemporalParams) -> list[Chain]:
    """Phase 5: longest chains first, RRF breaking ties.

    max_videos caps distinct videos; chains_per_video caps each video's share.
    """
    ranked = sorted((c for cs in by_video.values() for c in cs),
                    key=lambda c: c.key, reverse=True)
    taken: dict[str, int] = {}
    out: list[Chain] = []
    for c in ranked:
        n = taken.get(c.video_id, 0)
        if n >= params.chains_per_video:
            continue
        if n == 0 and len(taken) >= params.max_videos:
            continue
        taken[c.video_id] = n + 1
        out.append(c)
    return out
