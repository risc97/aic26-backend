from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass, field
from db import Keyframe

# One event in the sequence: a single query, or several paraphrases of it.
# Paraphrases are OR'd (max-pooled), so a variant that misses costs nothing.
Stage = str | Sequence[str]


@dataclass
class Gap:
    """How far a stage may sit from the previously matched stage.

    Frame counts are usually the tighter constraint -- shot lengths vary by an
    order of magnitude, so "2 to 3 keyframes later" says far more than any
    millisecond window can.
    """
    min_frames: int = 1             # 2 == "at least 2 keyframes later"
    max_frames: int | None = None   # 3 == "no more than 3 keyframes later"
    min_gap_ms: int = 0
    max_gap_ms: int | None = 120_000  # 2 minutes


@dataclass
class TemporalParams:
    candidate_k: int = 2000 # stage-1 recall
    max_videos: int = 500
    min_stages: int | None = None # video kept if it hits >= min_stages
    gaps: Gap | list[Gap] = field(default_factory=Gap) # one shared, or one per transition
    skip_penalty: float = 0.5 # z-units charged for leaving a stage unmatched
    max_skips: int | None = 0 # hard cap on unmatched stages; 0 == all required
    score_cap: float | None = None # saturate each stage at +-cap z-units; None == unbounded
    span_penalty: float = 0.0 # z-units per minute of total sequence span
    beta: float | None = None # parameter for log(1 + beta * sim) transformation; None == no transformation

    def gap_for(self, stage: int) -> Gap:
        """Constraint on arriving at `stage` from the previously matched stage.

        A list holds one entry per *transition*, so `gaps[i - 1]` guards the hop
        into stage i. Stage 0 always starts from the sentinel, where no gap
        applies, so its value is never used.
        """
        if isinstance(self.gaps, Gap):
            return self.gaps
        if not self.gaps:
            raise ValueError("gaps list is empty")
        return self.gaps[min(max(stage - 1, 0), len(self.gaps) - 1)]


@dataclass
class StageMatch:
    stage: int
    query: str
    score: float
    keyframe: Keyframe
    video_fps: float


@dataclass
class TemporalHit:
    rank: int
    score: float
    video_id: str
    matches: list[StageMatch]
    skipped: list[int]
