"""High-performance Numba JIT accelerated top-K sliding window max solver for Top-K backward DP.

Ref: temporal/desc.md
"""
from __future__ import annotations

import numba
import numpy as np

NEG_INF = -1e18


@numba.njit(cache=True)
def _sliding_window_topk_numba(
    F_next_topk: np.ndarray,
    l_bounds: np.ndarray,
    r_bounds: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-K sliding window max for backward DP using Numba C JIT compilation.

    F_next_topk: (K_prev, N) matrix of candidate scores from next stage.
    l_bounds   : (N,) array of left window bounds.
    r_bounds   : (N,) array of right window bounds.
    top_k      : K desired candidates.

    Returns:
        max_vals        : (top_k, N) array of top-K scores.
        opt_indices     : (top_k, N) array of corresponding frame indices k.
        opt_rank_indices: (top_k, N) array of corresponding candidate ranks r_prev.
    """
    K_prev, n = F_next_topk.shape

    max_vals = np.full((top_k, n), NEG_INF, dtype=np.float64)
    opt_indices = np.full((top_k, n), -1, dtype=np.int64)
    opt_rank_indices = np.full((top_k, n), -1, dtype=np.int64)

    max_cand = n * K_prev
    cand_v = np.empty(max_cand, dtype=np.float64)
    cand_k = np.empty(max_cand, dtype=np.int64)
    cand_r = np.empty(max_cand, dtype=np.int64)

    for j in range(n):
        l = l_bounds[j]
        r = r_bounds[j]

        if l > r or l >= n or r < 0:
            continue

        r_effective = min(r, n - 1)
        l_effective = max(l, 0)

        if l_effective > r_effective:
            continue

        count = 0
        for k in range(l_effective, r_effective + 1):
            for r_prev in range(K_prev):
                v = F_next_topk[r_prev, k]
                if v > NEG_INF / 2:
                    cand_v[count] = v
                    cand_k[count] = k
                    cand_r[count] = r_prev
                    count += 1

        if count == 0:
            continue

        k_take = min(top_k, count)
        for idx in range(k_take):
            best_idx = idx
            best_val = cand_v[idx]
            for m in range(idx + 1, count):
                if cand_v[m] > best_val:
                    best_val = cand_v[m]
                    best_idx = m

            if best_idx != idx:
                cand_v[idx], cand_v[best_idx] = cand_v[best_idx], cand_v[idx]
                cand_k[idx], cand_k[best_idx] = cand_k[best_idx], cand_k[idx]
                cand_r[idx], cand_r[best_idx] = cand_r[best_idx], cand_r[idx]

            max_vals[idx, j] = cand_v[idx]
            opt_indices[idx, j] = cand_k[idx]
            opt_rank_indices[idx, j] = cand_r[idx]

    return max_vals, opt_indices, opt_rank_indices


def sliding_window_topk(
    F_next_topk: np.ndarray,
    l_bounds: np.ndarray,
    r_bounds: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Top-K max window values for backward DP using Numba C JIT execution."""
    return _sliding_window_topk_numba(F_next_topk, l_bounds, r_bounds, top_k)


def sliding_window_max(
    F_next: np.ndarray,
    l_bounds: np.ndarray,
    r_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """1-best sliding window max wrapper."""
    max_vals, opt_k, _ = sliding_window_topk(np.atleast_2d(F_next), l_bounds, r_bounds, top_k=1)
    return max_vals[0], opt_k[0]
