# Temporal search

Find videos where a sequence of events happens **in order**.

The user describes several moments — "a man walks into a shop", then "he argues
with the cashier", then "police arrive outside". Any one of those queries alone
retrieves mostly noise. The order is the signal, so the search never scores a
keyframe in isolation; it scores *chains* of keyframes moving forward in time.

## The core idea

Only the first query searches the whole corpus. Its hits decide two things at
once: **which videos** are worth looking at, and **where in each video** to start.
Every later query only ever looks forward from those points.

This is what makes it cheap. Instead of `n` corpus-wide searches plus an
alignment step, it's one corpus-wide search, one restricted search, and a linear
walk.

## Vocabulary

The paper this follows talks about *segments*. In this codebase a segment is a
**keyframe** — `segments` and `keyframes` are 1:1 (97,825 rows each), and each
keyframe carries a `segment_id` whose row gives the shot's `frame_start` /
`frame_end`. The vector index is keyed on keyframes, so "segment" and "keyframe"
are interchangeable below. The shot span is used only for IoU.

Corpus shape, for calibration: 873 videos, ~112 keyframes per video.

## Pipeline

```
q1 ──► [1] seeds ──► [2] pool ──► [3] score ──► [4] chain ──► [5] rank ──► hits
        S_R            S_C          ranks         walk         length,RRF
```

### 1. Seeds — `search.py`

Search `q1` across the whole index, keep the top `r` keyframes. This is **S_R**.

Results come back sorted by score, so position `j` *is* rank `j + 1`. That rank
is the `r_1` the RRF formula needs later — no separate ranking pass.

### 2. Pool — `candidates.build_pool`

Decode each seed to its video. For each of those videos, take every keyframe
strictly **after the earliest seed in it**. The union of those is **S_C**.

Cutting at the earliest seed is the same as unioning "everything after seed X"
over every seed in that video — a union of suffixes is the suffix of the
earliest one.

Two things this buys:

- **Size.** a few tens of thousands of candidates instead of all 97,825. How
  many depends on how widely `r` spreads across videos.
- **Ordering for free.** A keyframe before `q1` is simply not in the pool, so
  nothing downstream has to check it.

Videos with no keyframe after their seed are dropped — the chain would be
length 1, which is just keyframe search wearing a temporal hat.

### 3. Score — `candidates.score_pool`

One `index.search` call carrying `q2..qn`, restricted to S_C by allowlist, with
`k = len(allowlist)`. Every candidate gets scored against every remaining query.
No top-k head, because truncation is exactly what sinks vague queries.

Then each query's scores become **ranks**, 1-based, **global over the whole
pool**. Global matters: rank within a video would make every video's best
keyframe rank 1 and carry no information at all.

### 4. Chain — `chains.walk`

Each seed starts one chain. From the seed, walk forward:

```
for each query q2..qn:
    look at keyframes after the previous hop, within max_gap_ms
    take the one with the best rank for this query
    if there are none -> the chain ends here
```

```
   t ──────────────────────────────────────────────►
        seed          q2              q3
   ─────●─────────────●───────────────●──────────────
        │◄── gap ────►│◄──── gap ────►│
        └ from S_R    └ best rank in the window
```

Because the walk stops the first time nothing is reachable, a short chain always
misses a **tail** of queries — never one in the middle. `skipped` in the response
is therefore always contiguous.

`max_gap_ms` is a locality guard. Without it a chain could hop to the
best-ranked keyframe anywhere later in the video, splicing two unrelated
occurrences into one confident but wrong answer.

### 5. Rank — `chains.dedupe`, `chains.select`

**Sort key is `(length, RRF)`, lexicographic.** A 4-hop chain beats a 3-hop chain
regardless of how much better the 3-hop's ranks are. Length carries the
discrimination; RRF only breaks ties.

RRF over the chain's per-query ranks:

```
RRF = Σ  w_i / (k + r_i)
```

`k = 60` smooths hard: rank 1 vs rank 2 is a 2× gap without it, ~2% with it. So
RRF is answering "did each query find something plausible", not "how good was
it". `w_i` lets you weight a query or hop; all 1.0 by default.

**Dedupe.** Seeds on adjacent keyframes of the same shot produce chains that
differ only in where they start. Compare their spans — first hop's segment start
to last hop's segment end — by IoU, and drop any chain overlapping a better one
past `iou_threshold`. Sorting by `(length, RRF)` first means the longer chain
survives and RRF breaks ties, automatically.

Two *genuinely* different occurrences sit at IoU ≈ 0 and both survive, which is
the point.

**Select.** `max_videos` caps distinct videos; `chains_per_video` caps how many
chains each contributes. Once the video budget is spent no new video enters, but
admitted ones can still add chains.

## Parameters

| name | default | meaning |
|---|---|---|
| `r` | 2000 | how deep q1 is retrieved (S_R size) |
| `rrf_k` | 60.0 | RRF smoothing constant |
| `weights` | `None` | `w_i` per query; `None` == all 1.0 |
| `max_gap_ms` | 120_000 | a hop may not jump further ahead than this |
| `iou_threshold` | 0.5 | chains overlapping more than this are duplicates |
| `chains_per_video` | 1 | chains one video may contribute |
| `max_videos` | 100 | distinct videos returned |

## Modules

| file | holds |
|---|---|
| `types.py` | `TemporalParams`, `Hop`, `Chain`, and the API-facing `StageMatch` / `TemporalHit` |
| `candidates.py` | `Pool`, `build_pool`, `score_pool`, `segment_spans` — phases 2–3 |
| `chains.py` | `walk`, `iou`, `dedupe`, `select` — phases 4–5 |
| `search.py` | `temporal_search` — phase 1 inline, then orchestration |

`Chain.key` is `(length, rrf)` and is the single definition of the sort order.
`dedupe` and `select` both use it; don't re-spell it anywhere else.

## Behaviors worth knowing

**q1 dominates.** A video whose first event misses q1's top-`r` is invisible — no
amount of q2/q3 evidence can bring it back. `r` is the first dial to tune when
recall looks wrong.

**Rank scales are asymmetric.** `r_1` tops out at `r` (2000); a later query's
rank ranges over all of S_C (~20k). So q1's RRF term is roughly an order of
magnitude larger than a weakly-ranked later hop's. `weights` is the lever if you
want to correct for it.

**The walk is greedy, not optimal.** It takes the best-ranked keyframe in the
window without checking whether that choice strands the queries after it. If q2's
best match sits near the end of the reachable window, q3 may have nothing left
and the chain dies short — even when a slightly worse q2 match would have carried
a full-length chain. Since length is the primary sort key, this costs real
ranking positions. `max_gap_ms` bounds the damage. The fix, if it becomes a
problem, is a small DP over the same `pool.ranks` arrays inside `walk`; nothing
else in the pipeline would change.

**Paraphrase variants are currently ignored.** `Stage` still types as
`str | Sequence[str]`, but only the first entry is encoded. The API and frontend
still accept a `variants` field that does nothing.

## Testing

`walk`, `iou`, `dedupe` and `select` are pure — a hand-built `Pool` exercises
them with no model, index, or database. `build_pool` and `score_pool` need only
a stub `index` exposing `search` and `contains`, plus a stub `db` with
`keyframes.list_by_videos`, `segments.list_by_videos` and `videos.get`.

Cases that catch real mistakes:

- a seed on a video's last keyframe — the video must be dropped, not crash
- `max_gap_ms` small enough that nothing is reachable — chains must come back
  `length=1` with a contiguous `skipped` tail
- two seeds one keyframe apart — `dedupe` must collapse them
- `chains_per_video=2, max_videos=3` — at most 6 chains across exactly 3 videos
- fewer than 2 stages — must raise, not `IndexError` inside `score_pool`
