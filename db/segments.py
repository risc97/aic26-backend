from __future__ import annotations

import sqlite3
from typing import Iterable

from .connection import transaction
from .models import Segment, row_to_segment


class SegmentRepo:
    """Segments CRUD"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, video_id: str, frame_start: int, frame_end: int) -> Segment:
        cur = self.conn.execute(
            "INSERT INTO segments (video_id, frame_start, frame_end) VALUES (?, ?, ?)",
            (video_id, frame_start, frame_end),
        )
        self.conn.commit()
        return Segment(cur.lastrowid, video_id, frame_start, frame_end)

    def create_many(self, video_id: str, spans: Iterable[tuple[int, int]]) -> list[Segment]:
        created: list[Segment] = []
        with transaction(self.conn) as conn:
            for frame_start, frame_end in spans:
                cur = conn.execute(
                    "INSERT INTO segments (video_id, frame_start, frame_end) VALUES (?, ?, ?)",
                    (video_id, frame_start, frame_end),
                )
                created.append(Segment(cur.lastrowid, video_id, frame_start, frame_end))
        return created

    def get(self, segment_id: int) -> Segment | None:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        return row_to_segment(row) if row else None

    def list_by_video(self, video_id: str) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE video_id = ?", (video_id,)
        ).fetchall()
        return [row_to_segment(r) for r in rows]

    def list_by_videos(self, video_ids: Iterable[str]):
        video_ids = list(video_ids)
        out: dict[str, list[Segment]] = {}
        for i in range(0, len(video_ids), 400):
            chunk = video_ids[i:i+400]
            placeholders = ",".join("?"*len(chunk))
            rows = self.conn.execute(
                f"SELECT * FROM segments WHERE video_id IN ({placeholders}) "
                f"ORDER BY video_id, frame_start",
                chunk,
            ).fetchall()
            for r in rows:
                video_id = r["video_id"]
                if video_id not in out:
                    out[video_id] = []
                out[video_id].append(row_to_segment(r))
        return out

    def update(
        self, segment_id: int,
        frame_start: int | None = None, frame_end: int | None = None,
    ) -> Segment | None:
        existing = self.get(segment_id)
        if existing is None:
            return None
        frame_start = existing.frame_start if frame_start is None else frame_start
        frame_end = existing.frame_end if frame_end is None else frame_end
        self.conn.execute(
            "UPDATE segments SET frame_start = ?, frame_end = ? WHERE segment_id = ?",
            (frame_start, frame_end, segment_id),
        )
        self.conn.commit()
        return Segment(segment_id, existing.video_id, frame_start, frame_end)

    def delete(self, segment_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM segments WHERE segment_id = ?", (segment_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_by_video(self, video_id: str) -> int:
        cur = self.conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount
