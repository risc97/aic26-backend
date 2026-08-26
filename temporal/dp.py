"""The sequence alignment DP using backward DP & sliding window max.

Ref: temporal/desc.md
Supports both strict sequence alignment, skip-layer alignment, and Top-K candidate alignment.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .sliding_window import sliding_window_max, sliding_window_topk
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


def align_topk(
    S: np.ndarray,
    ts: np.ndarray,
    gap_of: Callable[[int], Gap],
    *,
    top_k: int = 5,
    skip_penalty: float = 0.5,
    span_penalty_ms: float = 0.0,
    max_skips: int | None = None,
    beta: float | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    """Top-K strictly-increasing assignment of stages to keyframes via backward DP.

    Returns (topk_scores, topk_paths) tuple where topk_scores is a list of up to K floats
    and topk_paths is a list of 1D arrays of frame indices (or -1 for skips).
    """
    M, N = S.shape
    capped = max_skips is not None
    layers = min(max_skips, M) + 1 if capped else 1

    if N == 0 or M == 0 or top_k <= 0:
        return [], []

    # Optional log(1 + beta * sim) transformation
    if beta is not None and beta > 0:
        S = np.log1p(np.maximum(-0.999, beta * S))

    # F[i, s, r, j] = score of r-th candidate at stage i, layer s, frame j
    F = np.full((M, layers, top_k, N), NEG, dtype=np.float64)
    opt_k = np.full((M, layers, top_k, N), -1, dtype=np.int64)
    opt_r = np.full((M, layers, top_k, N), -1, dtype=np.int64)
    is_skip = np.zeros((M, layers, top_k, N), dtype=bool)

    # Base case: last stage i = M - 1
    for s in range(layers):
        F[M - 1, s, 0] = S[M - 1] - span_penalty_ms * ts
        if capped and s > 0:
            skip_val = -skip_penalty
            if skip_val > F[M - 1, s, 0, 0]:
                F[M - 1, s, 0, :] = skip_val
                is_skip[M - 1, s, 0, :] = True

    # Backward DP loop
    for i in range(M - 2, -1, -1):
        l_bounds, r_bounds = _compute_bounds(ts, gap_of(i + 1))

        for s in range(layers):
            # Top-K range max over candidate ranks from next stage
            max_vals, opt_indices, opt_ranks = sliding_window_topk(
                F[i + 1, s], l_bounds, r_bounds, top_k
            )

            for r in range(top_k):
                valid = max_vals[r] > NEG / 2
                cand_match = np.full(N, NEG, dtype=np.float64)
                cand_match[valid] = S[i, valid] + max_vals[r, valid]

                F[i, s, r] = cand_match
                opt_k[i, s, r] = opt_indices[r]
                opt_r[i, s, r] = opt_ranks[r]

                # Option B: Skip stage i
                if capped and s > 0:
                    cand_skip = F[i + 1, s - 1, r] - skip_penalty
                    take_skip = cand_skip > F[i, s, r]
                    F[i, s, r][take_skip] = cand_skip[take_skip]
                    is_skip[i, s, r][take_skip] = True

    # Flatten initial stage scores F[0] to find global top-K candidate starting points
    F0 = F[0].reshape(-1)
    topk_flat_indices = np.argsort(-F0)[:top_k]

    topk_scores: list[float] = []
    topk_paths: list[np.ndarray] = []

    for flat_idx in topk_flat_indices:
        score_val = float(F0[flat_idx])
        if score_val <= NEG / 2:
            break

        start_s, start_r, start_j = np.unravel_index(int(flat_idx), F[0].shape)
        path = np.full(M, -1, dtype=np.int64)

        curr_s = int(start_s)
        curr_r = int(start_r)
        curr_j = int(start_j)

        for i in range(M):
            skipped = bool(is_skip[i, curr_s, curr_r, curr_j])
            if not skipped:
                path[i] = curr_j
                next_j = int(opt_k[i, curr_s, curr_r, curr_j])
                next_r = int(opt_r[i, curr_s, curr_r, curr_j])
                curr_j = next_j if next_j >= 0 else curr_j
                curr_r = next_r if next_r >= 0 else 0
            else:
                path[i] = -1
                curr_s = max(0, curr_s - 1)

        first_frame = path[0] if path[0] >= 0 else 0
        final_score = float(score_val + span_penalty_ms * ts[first_frame])

        topk_scores.append(final_score)
        topk_paths.append(path)

    return topk_scores, topk_paths


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
    """1-best strictly-increasing assignment of stages to keyframes via backward DP."""
    scores, paths = align_topk(
        S, ts, gap_of, top_k=1,
        skip_penalty=skip_penalty,
        span_penalty_ms=span_penalty_ms,
        max_skips=max_skips,
        beta=beta,
    )
    if not scores:
        return float("-inf"), np.full(S.shape[0], -1, dtype=np.int64)
    return scores[0], paths[0]
