"""The sequence alignment DP using backward DP & sliding window max.

Ref: temporal/desc.md
Supports both strict sequence alignment and skip-layer alignment.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .sliding_window import sliding_window_max
from .types import Gap

NEG = -1e18


def _compute_bounds(ts: np.ndarray, gap: Gap) -> tuple[np.ndarray, np.ndarray]:
    """Computes monotonic l_bounds and r_bounds for keyframes ts under Gap constraints."""
    n = ts.shape[0]
    l_bounds = np.empty(n, dtype=np.int64)
    r_bounds = np.empty(n, dtype=np.int64)

    min_frames = max(1, gap.min_frames)
    max_frames = gap.max_frames if gap.max_frames is not None else n - 1
    min_gap_ms = gap.min_gap_ms
    max_gap_ms = gap.max_gap_ms

    for j in range(n):
        # 1. Min frame & min time bound
        min_k_frame = j + min_frames
        min_k_ts = int(np.searchsorted(ts, ts[j] + min_gap_ms, side='left'))
        l_j = max(min_k_frame, min_k_ts)

        # 2. Max frame & max time bound
        max_k_frame = j + max_frames
        if max_gap_ms is not None:
            max_k_ts = int(np.searchsorted(ts, ts[j] + max_gap_ms, side='right')) - 1
        else:
            max_k_ts = n - 1

        r_j = min(max_k_frame, max_k_ts)
        
        l_bounds[j] = l_j
        r_bounds[j] = r_j

    # Enforce monotonicity across j
    for j in range(1, n):
        if l_bounds[j] < l_bounds[j - 1]:
            l_bounds[j] = l_bounds[j - 1]
        if r_bounds[j] < r_bounds[j - 1]:
            r_bounds[j] = r_bounds[j - 1]

    return l_bounds, r_bounds


def align(
    S: np.ndarray,
    ts: np.ndarray,
    gap_of: Callable[[int], Gap],
    *,
    skip_penalty: float = 0.5,
    span_penalty_ms: float = 0.0,
    max_skips: int | None = None,
    beta: float | None = None,
) -> tuple[float, np.ndarray]:
    """Best strictly-increasing assignment of stages to keyframes via backward DP.

    S         -- (M, N) normalized similarity, S[i, j] = stage i vs keyframe j
    ts        -- (N,) keyframe timestamps in ms, ascending
    gap_of    -- stage index -> Gap guarding arrival at that stage
    max_skips -- max allowed skipped stages (0 or None)

    Returns (score, path) tuple.
    """
    M, N = S.shape
    capped = max_skips is not None
    layers = min(max_skips, M) + 1 if capped else 1

    if N == 0 or M == 0:
        return float("-inf"), np.full(M, -1, dtype=np.int64)

    # Optional log(1 + beta * sim) transformation
    if beta is not None and beta > 0:
        S = np.log1p(np.maximum(-0.999, beta * S))

    # F[i, s, j] = max score from stage i to M-1 with s skips remaining at frame j
    F = np.full((M, layers, N), NEG, dtype=np.float64)
    opt = np.full((M, layers, N), -1, dtype=np.int64)
    is_skip = np.zeros((M, layers, N), dtype=bool)

    # Base case: last stage (i = M - 1)
    # Matching stage M - 1 at frame j
    for s in range(layers):
        F[M - 1, s] = S[M - 1] - span_penalty_ms * ts
        
        # Or skipping stage M - 1 if s > 0
        if capped and s > 0:
            skip_val = -skip_penalty
            take_skip = skip_val > F[M - 1, s]
            F[M - 1, s][take_skip] = skip_val
            is_skip[M - 1, s][take_skip] = True

    # Backward DP loop: i = M - 2 down to 0
    for i in range(M - 2, -1, -1):
        l_bounds, r_bounds = _compute_bounds(ts, gap_of(i + 1))

        for s in range(layers):
            # Option A: Match stage i at frame j
            max_vals, opt_indices = sliding_window_max(F[i + 1, s], l_bounds, r_bounds)

            valid = max_vals > NEG / 2
            cand_match = np.full(N, NEG, dtype=np.float64)
            cand_match[valid] = S[i, valid] + max_vals[valid]

            F[i, s] = cand_match
            opt[i, s] = opt_indices

            # Option B: Skip stage i (consume 1 skip from s - 1)
            if capped and s > 0:
                cand_skip = F[i + 1, s - 1] - skip_penalty
                take_skip = cand_skip > F[i, s]
                F[i, s][take_skip] = cand_skip[take_skip]
                is_skip[i, s][take_skip] = True

    # Find optimal starting state at stage 0
    best_layer, best_j = np.unravel_index(int(np.argmax(F[0])), F[0].shape)
    best_score = float(F[0, best_layer, best_j])

    if best_score <= NEG / 2:
        return float("-inf"), np.full(M, -1, dtype=np.int64)

    # Backtracking path
    path = np.full(M, -1, dtype=np.int64)
    curr_layer = int(best_layer)
    curr_j = int(best_j)

    for i in range(M):
        skipped = bool(is_skip[i, curr_layer, curr_j])
        if not skipped:
            path[i] = curr_j
            next_j = int(opt[i, curr_layer, curr_j])
            curr_j = next_j if next_j >= 0 else curr_j
        else:
            path[i] = -1
            curr_layer = max(0, curr_layer - 1)

    return float(best_score + span_penalty_ms * ts[path[0] if path[0] >= 0 else 0]), path
