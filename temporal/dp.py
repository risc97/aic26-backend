"""The sequence alignment DP. Pure numpy -- no index, no database, no model.

`dp` is indexed by *time cursor*, not by stage. After stages 0..i have been
processed, column 0 means "no stage has matched yet" and column j+1 means "the
last stage that matched landed on keyframe j". The cursor is a sufficient
summary of the past, which is what keeps this linear instead of exponential.
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
          skip_penalty: float = 0.5, span_penalty_ms: float = 0.0):
    """Best strictly-increasing assignment of stages to keyframes.

    S      -- (M, N) normalized similarity, S[i, j] = stage i against keyframe j
    ts     -- (N,) keyframe timestamps in ms, ascending
    gap_of -- stage index -> the Gap guarding arrival at that stage

    Returns (score, path); path[i] is the keyframe index matched to stage i,
    or -1 when that stage was skipped.
    """
    M, N = S.shape
    dp = np.full(N + 1, NEG)
    dp[0] = 0.0
    parent = np.zeros((M, N + 1), dtype=np.int32)
    matched = np.zeros((M, N + 1), dtype=bool)

    for i in range(M):
        # option A -- skip stage i, cursor stays where it was
        cur = dp - skip_penalty
        par = np.arange(N + 1, dtype=np.int32)

        # option B -- match stage i at keyframe j, advancing the cursor
        best_val, best_col = _best_predecessor(dp, ts, gap_of(i), span_penalty_ms)
        cand = S[i] + best_val
        take = cand > cur[1:]

        cur[1:][take] = cand[take]
        par[1:][take] = best_col[take]

        matched[i, 1:] = take
        parent[i] = par
        dp = cur

    end = int(np.argmax(dp))
    path = np.full(M, -1, dtype=np.int64)
    col = end
    for i in range(M - 1, -1, -1):
        # a skip and a match can land on the same column, so `matched` is what
        # separates "stage i stopped here" from "stage i was passed over"
        if matched[i, col]:
            path[i] = col - 1
        col = int(parent[i, col])
    return float(dp[end]), path
