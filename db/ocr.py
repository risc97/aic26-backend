from __future__ import annotations

import sqlite3
from typing import Iterable

from .connection import transaction
from .models import Ocr, row_to_ocr

COLUMNS = "video_id, keyframe_id, text"


class OCRRepo:
    """OCR CRUD"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, video_id: str, keyframe_id: str, text: str) -> Ocr:
        self.conn.execute(
            f"INSERT INTO ocr ({COLUMNS}) VALUES (?, ?, ?)",
            (video_id, keyframe_id, text),
        )
        self.conn.commit()
        return Ocr(video_id, keyframe_id, text)

    def create_many(self, rows: Iterable[tuple[str, str, str]]) -> int:
        rows = list(rows)
        with transaction(self.conn) as conn:
            conn.executemany(
                f"INSERT INTO ocr ({COLUMNS}) VALUES (?, ?, ?)", rows
            )
        return len(rows)

    def get(self, video_id: str, keyframe_id: str) -> Ocr | None:
        row = self.conn.execute(
            "SELECT * FROM ocr WHERE video_id = ? AND keyframe_id = ?",
            (video_id, keyframe_id),
        ).fetchone()
        return row_to_ocr(row) if row else None

    def list_by_video(self, video_id: str) -> list[Ocr]:
        rows = self.conn.execute(
            "SELECT * FROM ocr WHERE video_id = ? ORDER BY keyframe_id", (video_id,)
        ).fetchall()
        return [row_to_ocr(r) for r in rows]

    def search(self, match: str, limit: int = 100) -> list[tuple[Ocr, float]]:
        """Full-text search"""
        rows = self.conn.execute(
            "SELECT o.video_id, o.keyframe_id, o.text, bm25(ocr_fts) AS score "
            "FROM ocr_fts JOIN ocr o ON o.ocr_id = ocr_fts.rowid "
            "WHERE ocr_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(row_to_ocr(r), -float(r["score"])) for r in rows]

    def delete(self, video_id: str, keyframe_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM ocr WHERE video_id = ? AND keyframe_id = ?",
            (video_id, keyframe_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_by_video(self, video_id: str) -> int:
        cur = self.conn.execute("DELETE FROM ocr WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount

    def count_by_video(self, video_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM ocr WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
