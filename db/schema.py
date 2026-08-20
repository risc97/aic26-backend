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
    keyframe_id   TEXT NOT NULL,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_idx      INTEGER NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    image_path    TEXT NOT NULL,
    segment_id    INTEGER REFERENCES segments(segment_id) ON DELETE SET NULL,
    PRIMARY KEY (video_id, keyframe_id),
    UNIQUE(video_id, frame_idx)
);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_time ON keyframes(video_id, timestamp_ms);

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id TEXT NOT NULL,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    time_start_ms INTEGER NOT NULL,
    time_end_ms   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    PRIMARY KEY (video_id, transcript_id)
);
CREATE INDEX IF NOT EXISTS idx_transcripts_video_time ON transcripts(video_id, time_start_ms);

CREATE TABLE IF NOT EXISTS ocr (
    ocr_id        INTEGER PRIMARY KEY,
    video_id      TEXT NOT NULL,
    keyframe_id   TEXT NOT NULL,
    text          TEXT NOT NULL,
    FOREIGN KEY (video_id, keyframe_id)
        REFERENCES keyframes(video_id, keyframe_id) ON DELETE CASCADE,
    UNIQUE (video_id, keyframe_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
    text,
    content     = 'ocr',
    content_rowid = 'ocr_id',
    tokenize    = "unicode61 remove_diacritics 0"
);
CREATE TRIGGER IF NOT EXISTS ocr_ai AFTER INSERT ON ocr BEGIN
    INSERT INTO ocr_fts(rowid, text) VALUES (new.ocr_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS ocr_ad AFTER DELETE ON ocr BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, text) VALUES ('delete', old.ocr_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS ocr_au AFTER UPDATE ON ocr BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, text) VALUES ('delete', old.ocr_id, old.text);
    INSERT INTO ocr_fts(rowid, text) VALUES (new.ocr_id, new.text);
END;


"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
