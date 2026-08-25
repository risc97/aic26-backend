from __future__ import annotations

import sqlite3
from typing import Iterable

from .connection import transaction
from .models import Transcript, row_to_transcript

COLUMNS = "video_id, transcript_id, time_start_ms, time_end_ms, text"

class TranscriptRepo:
    """Transcript CRUD"""
    def __init__(self, conn: sqlite3.Connection):
            self.conn = conn

    def create(self, video_id: str, transcript_id: str, time_start_ms: int,
               time_end_ms: int, text: str) -> Transcript:
        self.conn.execute(
            f"INSERT INTO transcripts ({COLUMNS}) VALUES (?, ?, ?, ?, ?)",
            (video_id, transcript_id, time_start_ms, time_end_ms, text),
        )
        self.conn.commit()
        return Transcript(transcript_id, video_id, time_start_ms, time_end_ms, text)

    def create_many(
        self, rows: Iterable[tuple[str, str, int, int, str]]
    ) -> int:
        rows = list(rows)
        with transaction(self.conn) as conn:
            conn.executemany(
                f"INSERT INTO transcripts ({COLUMNS}) VALUES (?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def get(self, video_id: str, transcript_id: str) -> Transcript | None:
        row = self.conn.execute(
            "SELECT * FROM transcripts WHERE video_id = ? AND transcript_id = ?",
            (video_id, transcript_id),
        ).fetchone()
        return row_to_transcript(row) if row else None

    def list_by_video(
        self, video_id: str, start_ms: int | None = None, end_ms: int | None = None,
    ) -> list[Transcript]:
        query = "SELECT * FROM transcripts WHERE video_id = ?"
        params: list = [video_id]
        if start_ms is not None:
            query += " AND time_end_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND time_start_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY time_start_ms"
        rows = self.conn.execute(query, params).fetchall()
        return [row_to_transcript(r) for r in rows]

    def search(self, match: str, limit: int = 100) -> list[tuple[Transcript, float]]:
        """Full-text search"""
        rows = self.conn.execute(
            "SELECT t.video_id, t.transcript_id, t.time_start_ms, t.time_end_ms, "
            "t.text, bm25(transcripts_fts) AS score "
            "FROM transcripts_fts JOIN transcripts t "
            "ON t.transcript_pk = transcripts_fts.rowid "
            "WHERE transcripts_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(row_to_transcript(r), -float(r["score"])) for r in rows]

    def delete(self, video_id: str, transcript_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM transcripts WHERE video_id = ? AND transcript_id = ?",
            (video_id, transcript_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_by_video(self, video_id: str) -> int:
        cur = self.conn.execute("DELETE FROM transcripts WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount

    def count_by_video(self, video_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
