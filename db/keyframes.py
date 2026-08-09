from __future__ import annotations

import sqlite3
from typing import Iterable

from .connection import transaction
from .models import Keyframe, row_to_keyframe


class KeyframeRepo:
    """Keyframes CRUD"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, video_id: str, frame_idx: int, timestamp_ms: int,
               image_path: str, segment_id: int) -> Keyframe:
        cur = self.conn.execute(
            "INSERT INTO keyframes (video_id, frame_idx, timestamp_ms, image_path, segment_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (video_id, frame_idx, timestamp_ms, image_path, segment_id),
        )
        self.conn.commit()
        return Keyframe(cur.lastrowid, video_id, frame_idx, timestamp_ms, image_path, segment_id)

    def create_many(self, rows: Iterable[tuple[str, int, int, str, int | None]]) -> int:
        rows = list(rows)
        with transaction(self.conn) as conn:
            conn.executemany(
                "INSERT INTO keyframes (video_id, frame_idx, timestamp_ms, image_path, segment_id) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get(self, keyframe_id: int) -> Keyframe | None:
        row = self.conn.execute(
            "SELECT * FROM keyframes WHERE keyframe_id = ?", (keyframe_id,)
        ).fetchone()
        return row_to_keyframe(row) if row else None

    def get_many(self, ids: Iterable[int]) -> list[Keyframe]:
        ids = list(ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM keyframes WHERE keyframe_id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["keyframe_id"]: row_to_keyframe(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def list_by_video(
        self, video_id: str, start_ms: int | None = None, end_ms: int | None = None,
    ) -> list[Keyframe]:
        query = "SELECT * FROM keyframes WHERE video_id = ?"
        params: list = [video_id]
        if start_ms is not None:
            query += " AND timestamp_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND timestamp_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY timestamp_ms"
        rows = self.conn.execute(query, params).fetchall()
        return [row_to_keyframe(r) for r in rows]

    def list_by_segment(self, segment_id: int) -> list[Keyframe]:
        rows = self.conn.execute(
            "SELECT * FROM keyframes WHERE segment_id = ? ORDER BY frame_idx", (segment_id,)
        ).fetchall()
        return [row_to_keyframe(r) for r in rows]

    def update(
        self, keyframe_id: int,
        frame_idx: int | None = None, timestamp_ms: int | None = None,
        image_path: str | None = None, segment_id: int | None = None,
    ) -> Keyframe | None:
        existing = self.get(keyframe_id)
        if existing is None:
            return None
        frame_idx = existing.frame_idx if frame_idx is None else frame_idx
        timestamp_ms = existing.timestamp_ms if timestamp_ms is None else timestamp_ms
        image_path = existing.image_path if image_path is None else image_path
        segment_id = existing.segment_id if segment_id is None else segment_id
        self.conn.execute(
            "UPDATE keyframes SET frame_idx = ?, timestamp_ms = ?, image_path = ?, "
            "segment_id = ? WHERE keyframe_id = ?",
            (frame_idx, timestamp_ms, image_path, segment_id, keyframe_id),
        )
        self.conn.commit()
        return Keyframe(keyframe_id, existing.video_id, frame_idx, timestamp_ms,
                        image_path, segment_id)

    def delete(self, keyframe_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM keyframes WHERE keyframe_id = ?", (keyframe_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_by_video(self, video_id: str) -> int:
        cur = self.conn.execute("DELETE FROM keyframes WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount

    def count_by_video(self, video_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM keyframes WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
