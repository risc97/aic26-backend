from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,
    video_path    TEXT NOT NULL,
    duration_ms   INTEGER,
    fps           REAL
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_start   INTEGER NOT NULL,
    frame_end     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);

CREATE TABLE IF NOT EXISTS keyframes (
    keyframe_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_idx      INTEGER NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    image_path    TEXT NOT NULL,
    segment_id    INTEGER REFERENCES segments(segment_id) ON DELETE SET NULL,
    UNIQUE(video_id, frame_idx)
);
CREATE INDEX IF NOT EXISTS idx_keyframes_video ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_time ON keyframes(video_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id, frame_idx);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
