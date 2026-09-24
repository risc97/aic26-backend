from dataclasses import dataclass
import numpy as np
from db import Keyframe
from keyframe.ids import split_vector_id, vector_id
from .types import Hop
import math
SENTINEL = int(np.iinfo(np.uint64).max)

@dataclass
class Pool:
    """Candidate keyframes S_C, scored against q2..qn. Indexed by column; columns are grouped per video and ordered by time."""
    allowlist: np.ndarray
    layout: dict[str, tuple[int, int]]
    keyframes: list[Keyframe]
    times: np.ndarray
    seeds: dict[str, list[Hop]]
    scores: np.ndarray
    ranks: np.ndarray

def _reach(t: int, n_away: int, max_gap_ms: int | None, direction: int) -> float:
    if n_away == 0:
        return float(t)
    if max_gap_ms is None:
        return direction * math.inf
    return float(t + direction * n_away * max_gap_ms)

def build_pool(index, db, seed_ids, seed_scores, n_stages: int, seed, max_gap_ms) -> Pool | None:
    """Phase 2: every keyframe a chain could reach from a seed, grouped per video

    The pool spans both sides of the seed
    """
    seed_stage = seed
    before, after = seed_stage, n_stages - 1 - seed_stage

    seeds_by_video: dict[str, list[tuple[str, int, float]]] = {}
    for rank, (packed, score) in enumerate(
            zip(seed_ids.tolist(), seed_scores.tolist()), start=1):
        video_id, keyframe_id = split_vector_id(packed)
        seeds_by_video.setdefault(video_id, []).append((keyframe_id, rank, float(score)))

    frames = db.keyframes.list_by_videos(list(seeds_by_video))

    allowlist: list[int] = []
    keyframes: list[Keyframe] = []
    times: list[int] = []
    layout: dict[str, tuple[int, int]] = {}
    seeds: dict[str, list[Hop]] = {}

    for video_id, wanted in seeds_by_video.items():
        kfs = frames.get(video_id, ())
        if not kfs:
            continue
        by_id = {kf.keyframe_id: kf for kf in kfs}

        hops = [Hop(seed_stage, -1, by_id[kid], score, rank)     # column -1: q1 is not in S_C
                for kid, rank, score in wanted if kid in by_id]
        if not hops:
            continue
        stamps = [hop.keyframe.timestamp_ms for hop in hops]
        lo_ms = _reach(min(stamps), before, max_gap_ms, -1)
        hi_ms = _reach(max(stamps), after, max_gap_ms, +1)


        start = len(allowlist)
        for kf in kfs:
            if kf.timestamp_ms <= lo_ms or kf.timestamp_ms >= hi_ms:
                continue
            try:
                packed = vector_id(kf.video_id, kf.keyframe_id)
            except ValueError:
                continue
            if not index.contains(packed):
                continue
            allowlist.append(packed)
            keyframes.append(kf)
            times.append(kf.timestamp_ms)

        if len(allowlist) == start: # # nothing reachable from the seed
            continue
        layout[video_id] = (start, len(allowlist))
        seeds[video_id] = hops

    if not allowlist:
        return None

    n = len(allowlist)
    return Pool(
        allowlist=np.asarray(allowlist, dtype=np.uint64),
        layout=layout,
        keyframes=keyframes,
        times=np.asarray(times, dtype=np.int64),
        seeds=seeds,
        scores=np.zeros((n_stages, n), dtype=np.float32),
        # worst possible rank, not 0: the walk picks the *smallest* rank, so a
        # column the index never returned must not look like a perfect match
        ranks=np.full((n_stages, n), n + 1, dtype=np.int32),
    )


def score_pool(index, queries: np.ndarray, pool: Pool, seed: int = 0) -> None:
    """score every stage but the seed over the whole pool in one pass, then rank them
    Fills pool.scores and pool.ranks in place; row i holds query i.
    """
    others = [s for s in range(queries.shape[0]) if s != seed]
    rest = np.ascontiguousarray(queries[others])
    hit_scores, hit_ids = index.search(rest, k=len(pool.allowlist),
                                       allowlist=pool.allowlist)
    hit_scores, hit_ids = np.atleast_2d(hit_scores), np.atleast_2d(hit_ids)

    column_of = {int(v): c for c, v in enumerate(pool.allowlist.tolist())}
    for row, stage in enumerate(others):
        cols = np.fromiter((column_of[int(v)] for v in hit_ids[row].tolist()),
                           dtype=np.int64, count=hit_ids.shape[1])
        pool.scores[stage, cols] = hit_scores[row]
        pool.ranks[stage, cols] = np.arange(1, len(cols) + 1)




def segment_spans(db, video_ids, fps: dict[str, float]) -> dict[int, tuple[int, int]]:
    """segment_id -> (start_ms, end_ms), so a chain gets a real interval for IoU."""
    out: dict[int, tuple[int, int]] = {}
    for video_id, segs in db.segments.list_by_videos(video_ids).items():
        rate = fps.get(video_id) or 25.0
        for s in segs:
            out[s.segment_id] = (int(s.frame_start * 1000 / rate),
                                 int(s.frame_end * 1000 / rate))
    return out
