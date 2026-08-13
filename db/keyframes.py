from __future__ import annotations

import sqlite3
from typing import Iterable

from .connection import transaction
from .models import Keyframe, row_to_keyframe

COLUMNS = "video_id, keyframe_id, frame_idx, timestamp_ms, image_path, segment_id"


class KeyframeRepo:
    """Keyframes CRUD"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, video_id: str, keyframe_id: str, frame_idx: int, timestamp_ms: int,
               image_path: str, segment_id: int) -> Keyframe:
        self.conn.execute(
            f"INSERT INTO keyframes ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, keyframe_id, frame_idx, timestamp_ms, image_path, segment_id),
        )
        self.conn.commit()
        return Keyframe(keyframe_id, video_id, frame_idx, timestamp_ms, image_path,
                        segment_id)

    def create_many(
        self, rows: Iterable[tuple[str, str, int, int, str, int | None]]
    ) -> int:
        rows = list(rows)
        with transaction(self.conn) as conn:
            conn.executemany(
                f"INSERT INTO keyframes ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def get(self, video_id: str, keyframe_id: str) -> Keyframe | None:
        row = self.conn.execute(
            "SELECT * FROM keyframes WHERE video_id = ? AND keyframe_id = ?",
            (video_id, keyframe_id),
        ).fetchone()
        return row_to_keyframe(row) if row else None

    def get_many(self, keys: Iterable[tuple[str, str]]) -> list[Keyframe]:
        keys = list(keys)
        if not keys:
            return []
        placeholders = ",".join(["(?, ?)"] * len(keys))
        params = [value for key in keys for value in key]
        rows = self.conn.execute(
            f"SELECT * FROM keyframes WHERE (video_id, keyframe_id) IN "
            f"(VALUES {placeholders})",
            params,
        ).fetchall()
        by_key = {(r["video_id"], r["keyframe_id"]): row_to_keyframe(r) for r in rows}
        return [by_key[k] for k in keys if k in by_key]

    def get_by_frame_idx(self, video_id: str, frame_idx: int) -> Keyframe | None:
        row = self.conn.execute(
            "SELECT * FROM keyframes WHERE video_id = ? AND frame_idx = ?",
            (video_id, frame_idx),
        ).fetchone()
        return row_to_keyframe(row) if row else None

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

    def list_by_videos(self, video_ids: Iterable[str]):
        video_ids = list(video_ids)
        out: dict[str, list[Keyframe]] = {}
        for i in range(0, len(video_ids), 400):
            chunk = video_ids[i:i+400]
            placeholders = ",".join("?"*len(chunk))
            rows = self.conn.execute(
                f"SELECT * FROM keyframes WHERE video_id IN ({placeholders}) "
                f"ORDER BY video_id, timestamp_ms",
                chunk,
            ).fetchall()
            for r in rows:
                video_id = r["video_id"]
                if video_id not in out:
                    out[video_id] = []
                out[video_id].append(row_to_keyframe(r))
        return out

    def list_by_segment(self, segment_id: int) -> list[Keyframe]:
        rows = self.conn.execute(
            "SELECT * FROM keyframes WHERE segment_id = ? ORDER BY frame_idx", (segment_id,)
        ).fetchall()
        return [row_to_keyframe(r) for r in rows]

    def update(
        self, video_id: str, keyframe_id: str,
        frame_idx: int | None = None, timestamp_ms: int | None = None,
        image_path: str | None = None, segment_id: int | None = None,
    ) -> Keyframe | None:
        existing = self.get(video_id, keyframe_id)
        if existing is None:
            return None
        frame_idx = existing.frame_idx if frame_idx is None else frame_idx
        timestamp_ms = existing.timestamp_ms if timestamp_ms is None else timestamp_ms
        image_path = existing.image_path if image_path is None else image_path
        segment_id = existing.segment_id if segment_id is None else segment_id
        self.conn.execute(
            "UPDATE keyframes SET frame_idx = ?, timestamp_ms = ?, image_path = ?, "
            "segment_id = ? WHERE video_id = ? AND keyframe_id = ?",
            (frame_idx, timestamp_ms, image_path, segment_id, video_id, keyframe_id),
        )
        self.conn.commit()
        return Keyframe(keyframe_id, video_id, frame_idx, timestamp_ms,
                        image_path, segment_id)

    def delete(self, video_id: str, keyframe_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM keyframes WHERE video_id = ? AND keyframe_id = ?",
            (video_id, keyframe_id),
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
