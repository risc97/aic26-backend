from __future__ import annotations

import sqlite3

from .models import Video, row_to_video


class VideoRepo:
    """Videos CRUD"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, video_id: str, video_path: str, duration_ms: int, fps: float) -> Video:
        self.conn.execute(
            "INSERT INTO videos (video_id, video_path, duration_ms, fps) VALUES (?, ?, ?, ?)",
            (video_id, video_path, duration_ms, fps),
        )
        self.conn.commit()
        return Video(video_id, video_path, duration_ms, fps)

    def get(self, video_id: str) -> Video | None:
        row = self.conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row_to_video(row) if row else None

    def list_all(self, limit: int = 1000, offset: int = 0) -> list[Video]:
        rows = self.conn.execute(
            "SELECT * FROM videos ORDER BY video_id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [row_to_video(r) for r in rows]

    def update(
        self, video_id: str, video_path: str | None = None,
        duration_ms: int | None = None, fps: float | None = None,
    ) -> Video | None:
        existing = self.get(video_id)
        if existing is None:
            return None
        video_path = existing.video_path if video_path is None else video_path
        duration_ms = existing.duration_ms if duration_ms is None else duration_ms
        fps = existing.fps if fps is None else fps
        self.conn.execute(
            "UPDATE videos SET video_path = ?, duration_ms = ?, fps = ? WHERE video_id = ?",
            (video_path, duration_ms, fps, video_id),
        )
        self.conn.commit()
        return Video(video_id, video_path, duration_ms, fps)

    def upsert(self, video_id: str, video_path: str, duration_ms: int, fps: float) -> Video:
        self.conn.execute(
            "INSERT INTO videos (video_id, video_path, duration_ms, fps) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(video_id) DO UPDATE SET video_path = excluded.video_path, "
            "duration_ms = excluded.duration_ms, fps = excluded.fps",
            (video_id, video_path, duration_ms, fps),
        )
        self.conn.commit()
        return Video(video_id, video_path, duration_ms, fps)

    def delete(self, video_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount > 0
