"""High-performance monotonic sliding window max solver for backward DP.

Provides a Numba JIT accelerated C-speed monotonic deque implementation with a pure
Python fallback. Emits a warning if Numba/C acceleration is not available.
"""
from __future__ import annotations

import warnings
import numpy as np

NEG_INF = -1e18

# Try loading Numba JIT acceleration
_NUMBA_AVAILABLE = False
try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

if not _NUMBA_AVAILABLE:
    warnings.warn(
        "WARNING: Numba/C++ high-performance deque solver is NOT available or loaded! "
        "Falling back to pure Python sliding window solver.",
        RuntimeWarning,
        stacklevel=2,
    )


if _NUMBA_AVAILABLE:
    @numba.njit(cache=True)
    def _sliding_window_max_numba(
        F_next: np.ndarray,
        l_bounds: np.ndarray,
        r_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = F_next.shape[0]
        max_vals = np.full(n, NEG_INF, dtype=np.float64)
        opt_indices = np.full(n, -1, dtype=np.int64)

        # Monotonic deque implemented as a simple fixed-size ring/array buffer
        deque = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        last_pushed = -1

        for j in range(n):
            l = l_bounds[j]
            r = r_bounds[j]

            if l > r or l >= n or r < 0:
                continue

            # Clip r to valid index range
            r_effective = min(r, n - 1)
            l_effective = max(l, 0)

            if l_effective > r_effective:
                continue

            # Push elements up to r_effective onto the deque
            while last_pushed < r_effective:
                last_pushed += 1
                k = last_pushed
                val_k = F_next[k]
                while tail > head and F_next[deque[tail - 1]] <= val_k:
                    tail -= 1
                deque[tail] = k
                tail += 1

            # Pop stale elements from front that are less than l_effective
            while tail > head and deque[head] < l_effective:
                head += 1

            # Extract window maximum
            if tail > head:
                best_k = deque[head]
                if best_k <= r_effective:
                    max_vals[j] = F_next[best_k]
                    opt_indices[j] = best_k

        return max_vals, opt_indices


def _sliding_window_max_python(
    F_next: np.ndarray,
    l_bounds: np.ndarray,
    r_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from collections import deque

    n = F_next.shape[0]
    max_vals = np.full(n, NEG_INF, dtype=np.float64)
    opt_indices = np.full(n, -1, dtype=np.int64)

    deq: deque[int] = deque()
    last_pushed = -1

    for j in range(n):
        l = l_bounds[j]
        r = r_bounds[j]

        if l > r or l >= n or r < 0:
            continue

        r_effective = min(r, n - 1)
        l_effective = max(l, 0)

        if l_effective > r_effective:
            continue

        while last_pushed < r_effective:
            last_pushed += 1
            k = last_pushed
            val_k = F_next[k]
            while deq and F_next[deq[-1]] <= val_k:
                deq.pop()
            deq.append(k)

        while deq and deq[0] < l_effective:
            deq.popleft()

        if deq:
            best_k = deq[0]
            if best_k <= r_effective:
                max_vals[j] = F_next[best_k]
                opt_indices[j] = best_k

    return max_vals, opt_indices


def sliding_window_max(
    F_next: np.ndarray,
    l_bounds: np.ndarray,
    r_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes max_{l(j) <= k <= r(j)} F_next[k] for each j in O(n) time.

    Returns (max_vals, opt_indices).
    """
    if _NUMBA_AVAILABLE:
        return _sliding_window_max_numba(F_next, l_bounds, r_bounds)
    else:
        return _sliding_window_max_python(F_next, l_bounds, r_bounds)
