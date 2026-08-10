from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class Video:
    video_id: str
    video_path: str
    duration_ms: int
    fps: float


@dataclass
class Segment:
    segment_id: int
    video_id: str
    frame_start: int
    frame_end: int


@dataclass
class Keyframe:
    keyframe_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    image_path: str
    segment_id: int


def row_to_video(r: sqlite3.Row) -> Video:
    return Video(
        video_id=r["video_id"], video_path=r["video_path"],
        duration_ms=r["duration_ms"], fps=r["fps"],
    )


def row_to_segment(r: sqlite3.Row) -> Segment:
    return Segment(
        segment_id=r["segment_id"], video_id=r["video_id"],
        frame_start=r["frame_start"], frame_end=r["frame_end"],
    )


def row_to_keyframe(r: sqlite3.Row) -> Keyframe:
    return Keyframe(
        keyframe_id=r["keyframe_id"], video_id=r["video_id"],
        frame_idx=r["frame_idx"], timestamp_ms=r["timestamp_ms"],
        image_path=r["image_path"], segment_id=r["segment_id"],
    )
