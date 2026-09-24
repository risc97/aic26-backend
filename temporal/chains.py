"""Phase 4-5: walk each seed forward into a chain, then thin and rank them."""
from __future__ import annotations

import numpy as np

from .candidates import Pool, segment_spans
from .types import Chain, Hop, StageMatch, TemporalHit, TemporalParams


def _span(hop: Hop, spans: dict[int, tuple[int, int]]) -> tuple[int, int]:
    """The shot a hop sits in, falling back to the keyframe's own instant"""
    span = spans.get(hop.keyframe.segment_id)
    if span is None:
        return hop.keyframe.timestamp_ms, hop.keyframe.timestamp_ms
    return span


def walk(pool: Pool, video_id: str, seed: Hop, params: TemporalParams,
         n_stages: int, spans: dict[int, tuple[int, int]]) -> Chain:
    """Grow chain outward in time from a single seed.

    """
    lo, hi = pool.layout[video_id]
    times = pool.times[lo:hi]
    gap = params.max_gap_ms

    def step(stage: int, cur: int, forward: bool) -> Hop | None:
        if forward:
            first = int(np.searchsorted(times, cur, side="right"))   # strictly after
            last = (len(times) if gap is None
                    else int(np.searchsorted(times, cur + gap, side="right")))
        else:
            last = int(np.searchsorted(times, cur, side="left"))     # strictly before
            first = (0 if gap is None
                     else int(np.searchsorted(times, cur - gap, side="left")))
        if first >= last:
            return None
        window = pool.ranks[stage, lo + first:lo + last]
        col = lo + first + int(window.argmin())                      # best rank in the window
        return Hop(stage, col, pool.keyframes[col],
                   float(pool.scores[stage, col]), int(pool.ranks[stage, col]))

    ahead: list[Hop] = []
    cur = seed.keyframe.timestamp_ms
    for stage in range(seed.stage + 1, n_stages):
        hop = step(stage, cur, forward=True)
        if hop is None:
            break                                                    # this direction ends here
        ahead.append(hop)
        cur = int(pool.times[hop.column])

    behind: list[Hop] = []
    cur = seed.keyframe.timestamp_ms
    for stage in range(seed.stage - 1, -1, -1):
        hop = step(stage, cur, forward=False)
        if hop is None:
            break
        behind.append(hop)
        cur = int(pool.times[hop.column])

    hops = behind[::-1] + [seed] + ahead                             # stage order == time order

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

def build_hits(pool: Pool, db, params: TemporalParams, n_stages: int,
               labels: list[str], make_chain) -> list[TemporalHit]:
    """
    make_chain(seed, video_id, spans) -> Chain turns one seed into a finished
    chain; callers differ only in how that chain is produced (walk() alone
    for a semantic search, walk() + _truncate() for a detection search).
    """
    fps: dict[str, float] = {}
    for video_id in pool.layout:
        video = db.videos.get(video_id)
        fps[video_id] = video.fps if video else 25.0
    spans = segment_spans(db, list(pool.layout), fps)

    by_video = {
        video_id: dedupe(
            [make_chain(seed, video_id, spans) for seed in seeds],
            params.iou_threshold)
        for video_id, seeds in pool.seeds.items()
    }

    return [
        TemporalHit(
            rank=rank, score=chain.rrf, length=chain.length,
            video_id=chain.video_id,
            matches=[StageMatch(h.stage, labels[h.stage], h.score, h.rank,
                                h.keyframe, fps.get(chain.video_id, 0.0))
                     for h in chain.hops],
            # a chain only ever runs out at the end, so this is always a tail
            skipped=sorted(set(range(n_stages)) - {h.stage for h in chain.hops}),
        )
        for rank, chain in enumerate(select(by_video, params), start=1)
    ]
