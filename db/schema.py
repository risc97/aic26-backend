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
    transcript_pk INTEGER PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    time_start_ms INTEGER NOT NULL,
    time_end_ms   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    UNIQUE (video_id, transcript_id)
);
CREATE INDEX IF NOT EXISTS idx_transcripts_video_time ON transcripts(video_id, time_start_ms);

CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    text,
    content       = 'transcripts',
    content_rowid = 'transcript_pk',
    tokenize      = "unicode61 remove_diacritics 0"
);
CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts(rowid, text) VALUES (new.transcript_pk, new.text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, text)
        VALUES ('delete', old.transcript_pk, old.text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, text)
        VALUES ('delete', old.transcript_pk, old.text);
    INSERT INTO transcripts_fts(rowid, text) VALUES (new.transcript_pk, new.text);
END;

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


# Databases written before transcripts_fts existed key transcripts on
# (video_id, transcript_id) and so have no stable rowid for FTS5 to index.
# Rebuild that table once, keeping the rows; CREATE TABLE IF NOT EXISTS cannot.
MIGRATE_TRANSCRIPT_PK = """
DROP TRIGGER IF EXISTS transcripts_ai;
DROP TRIGGER IF EXISTS transcripts_ad;
DROP TRIGGER IF EXISTS transcripts_au;
DROP TABLE IF EXISTS transcripts_fts;

ALTER TABLE transcripts RENAME TO transcripts_old;
CREATE TABLE transcripts (
    transcript_pk INTEGER PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    time_start_ms INTEGER NOT NULL,
    time_end_ms   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    UNIQUE (video_id, transcript_id)
);
INSERT INTO transcripts (video_id, transcript_id, time_start_ms, time_end_ms, text)
    SELECT video_id, transcript_id, time_start_ms, time_end_ms, text
    FROM transcripts_old ORDER BY video_id, transcript_id;
DROP TABLE transcripts_old;
"""


def _needs_transcript_pk(conn: sqlite3.Connection) -> bool:
    columns = conn.execute("PRAGMA table_info(transcripts)").fetchall()
    return bool(columns) and not any(c["name"] == "transcript_pk" for c in columns)


def _backfill_transcripts_fts(conn: sqlite3.Connection) -> None:
    """Index rows stored before transcripts_fts and its triggers existed.

    Counting transcripts_fts would not tell us whether the index is populated:
    on an external-content table a scan reads the content table, so an empty
    index still counts every transcript. The caller checks for the table itself.
    """
    stored = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    if not stored:
        return
    conn.execute("INSERT INTO transcripts_fts(transcripts_fts) VALUES ('rebuild')")
    print(f"indexed {stored} transcript rows into transcripts_fts")


def init_schema(conn: sqlite3.Connection) -> None:
    if _needs_transcript_pk(conn):
        conn.executescript(MIGRATE_TRANSCRIPT_PK)
    fts_is_new = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name = 'transcripts_fts'"
    ).fetchone()[0] == 0
    conn.executescript(SCHEMA)
    if fts_is_new:
        _backfill_transcripts_fts(conn)
    conn.commit()
