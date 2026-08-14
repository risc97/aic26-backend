"""The sequence alignment DP. Pure numpy -- no index, no database, no model.

`dp` is indexed by *time cursor*, not by stage. After stages 0..i have been
processed, column 0 means "no stage has matched yet" and column j+1 means "the
last stage that matched landed on keyframe j". The cursor is a sufficient
summary of the past, which is what keeps this linear instead of exponential.

With `max_skips` set, the cursor alone is no longer sufficient -- how many
stages were skipped to reach it now matters -- so `dp` grows a second axis, one
layer per skip count. Matching stays inside a layer; skipping moves up one.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from .types import Gap

NEG = -1e18


def _best_predecessor(dp, ts, gap, span):
    """For each keyframe j, the best dp column to arrive from.

    Legal predecessors are the sentinel (always -- a stage may start the
    sequence) and any earlier keyframe satisfying every `gap` bound. The span
    penalty is folded in via
    `dp[j'] - span * (ts[j] - ts[j'])  ==  (dp[j'] + span*ts[j']) - span*ts[j]`,
    turning a per-pair cost into a constant offset so a plain sliding-window
    max works.

    Both window edges move forward monotonically as j grows, so each frame is
    admitted once and evicted once -- O(n) for the whole stage.
    """
    n = ts.shape[0]
    aug = dp[1:] + span * ts
    best_val = np.full(n, NEG)
    best_col = np.zeros(n, dtype=np.int32)

    min_frames = max(1, gap.min_frames)  # two stages may never share a keyframe
    window: deque[int] = deque()  # keyframe indices ascending, aug descending
    admitted = 0                  # next frame not yet eligible

    for j in range(n):
        # admit frames now far enough behind j; both bounds only ever relax
        newest, latest_ts = j - min_frames, ts[j] - gap.min_gap_ms
        while admitted <= newest and ts[admitted] <= latest_ts:
            while window and aug[window[-1]] <= aug[admitted]:
                window.pop()
            window.append(admitted)
            admitted += 1

        # drop frames now too far behind j
        if gap.max_frames is not None:
            oldest = j - gap.max_frames
            while window and window[0] < oldest:
                window.popleft()
        if gap.max_gap_ms is not None:
            bound = ts[j] - gap.max_gap_ms
            while window and ts[window[0]] < bound:
                window.popleft()

        if window:
            k = window[0]  # deque invariants put the window maximum at the front
            best_val[j] = aug[k] - span * ts[j]
            best_col[j] = k + 1
        if dp[0] > best_val[j]:  # starting fresh from the sentinel
            best_val[j] = dp[0]
            best_col[j] = 0
    return best_val, best_col


def align(S, ts, gap_of: Callable[[int], Gap], *,
          skip_penalty: float = 0.5, span_penalty_ms: float = 0.0,
          max_skips: int | None = None):
    """Best strictly-increasing assignment of stages to keyframes.

    S         -- (M, N) normalized similarity, S[i, j] = stage i vs keyframe j
    ts        -- (N,) keyframe timestamps in ms, ascending
    gap_of    -- stage index -> the Gap guarding arrival at that stage
    max_skips -- hard cap on unmatched stages; None leaves it uncapped, and
                 `skip_penalty` alone decides. 0 demands every stage match.

    Returns (score, path); path[i] is the keyframe index matched to stage i,
    or -1 when that stage was skipped. When the cap makes every alignment
    impossible, the score is -inf and no stage is matched.
    """
    M, N = S.shape
    capped = max_skips is not None
    layers = min(max_skips, M) + 1 if capped else 1

    dp = np.full((layers, N + 1), NEG)
    dp[0, 0] = 0.0  # nothing matched, nothing skipped
    parent = np.zeros((M, layers, N + 1), dtype=np.int32)
    matched = np.zeros((M, layers, N + 1), dtype=bool)
    stay = np.arange(N + 1, dtype=np.int32)

    for i in range(M):
        cur = np.full((layers, N + 1), NEG)
        par = np.zeros((layers, N + 1), dtype=np.int32)
        for s in range(layers):
            # option A -- skip stage i: cursor stays put, one more skip spent.
            # Uncapped, there is only layer 0 and a skip stays inside it.
            src = s - 1 if capped else s
            if src >= 0:
                cur[s] = dp[src] - skip_penalty
                par[s] = stay

            # option B -- match stage i at keyframe j, advancing the cursor
            best_val, best_col = _best_predecessor(dp[s], ts, gap_of(i),
                                                   span_penalty_ms)
            cand = S[i] + best_val
            take = cand > cur[s, 1:]

            cur[s, 1:][take] = cand[take]
            par[s, 1:][take] = best_col[take]
            matched[i, s, 1:] = take

        parent[i] = par
        dp = cur

    layer, col = np.unravel_index(int(np.argmax(dp)), dp.shape)
    best = float(dp[layer, col])
    path = np.full(M, -1, dtype=np.int64)
    if best <= NEG / 2:  # e.g. max_skips=0 with fewer keyframes than stages
        return float("-inf"), path

    layer, col = int(layer), int(col)
    for i in range(M - 1, -1, -1):
        # a skip and a match can land on the same column, so `matched` is what
        # separates "stage i stopped here" from "stage i was passed over"
        hit = bool(matched[i, layer, col])
        if hit:
            path[i] = col - 1
        col = int(parent[i, layer, col])
        if not hit and capped:
            layer -= 1  # a skip arrived from one layer down
    return best, path
