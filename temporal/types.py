from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass, field
from db import Keyframe

Stage = str | Sequence[str]

@dataclass
class TemporalParams:
    r: int = 2000 # q1 candidates -> S_R
    rrf_k: float = 60.0
    weights: list[float] | None = None # w_i per query
    max_gap_ms: int | None = 120_000
    iou_threshold: float = 0.5
    chains_per_video: int = 1 # chains one video have
    max_videos: int = 100

    def weight(self, stage: int) -> float:
        """w_i for one query"""
        if not self.weights:
            return 1.0
        return self.weights[stage] if stage < len(self.weights) else 1.0

@dataclass
class Hop:
    stage: int
    column: int # position in the S_C
    keyframe: Keyframe
    score: float # cosine
    rank: int # 

@dataclass
class Chain:
    video_id: str
    hops: list[Hop] # hops[0] is q1
    rrf: float
    start_ms: int # first hop's segment begin
    end_ms: int # last hop's segment

    @property
    def length(self) -> int:
        return len(self.hops)

    @property
    def key(self) -> tuple[int, float]:
        return (self.length, self.rrf)
    
@dataclass
class StageMatch:
    # A hop for API response
    stage: int
    query: str
    score: float
    rank: int
    keyframe: Keyframe
    video_fps: float


@dataclass
class TemporalHit:
    # A chain for API response
    rank: int
    score: float
    length: int
    video_id: str
    matches: list[StageMatch]
    skipped: list[int]