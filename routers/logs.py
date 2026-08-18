from schemas import LogEntry, Item, TranscriptItem
from config import DEFAULT_MODEL, LOG_DB_PATH
from fastapi import APIRouter
import sqlite3
import json
from pathlib import Path
from db import connect, transaction

router = APIRouter(tags=["Logs"])

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT,
    query TEXT,
    result_limit INTEGER,
    mode TEXT,
    model TEXT,
    results TEXT,
    total INTEGER
);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
"""

def init_logs_schema(conn: sqlite3.Connection):
    conn.executescript(LOG_SCHEMA)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(logs)")}
    if "model" not in columns:
        conn.execute("ALTER TABLE logs ADD COLUMN model TEXT")
    conn.commit()


class LogDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        self.conn = connect(db_path)
        self.logs = LogRepo(self.conn)

    def init_schema(self) -> None:
        init_logs_schema(self.conn)

    def transaction(self):
        return transaction(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LogDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

class LogRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(self, entry: LogEntry):
        self.conn.execute(
            "INSERT INTO logs (request_id, timestamp, query, result_limit, mode, model, results, total) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.request_id, entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'), entry.query, entry.limit, entry.mode,
                entry.model, json.dumps([i.model_dump() for i in entry.results]), entry.total,
            ),
        )
        self.conn.commit()
        return entry

    def get(self, request_id:str) -> LogEntry | None:
        row = self.conn.execute(
            "SELECT * FROM logs WHERE request_id = ?",
            (request_id,)
        ).fetchone()
        return row_to_log(row) if row else None

    def list_recent(self, limit: int = 100, offset: int = 0) -> list[LogEntry]:
        rows = self.conn.execute(
            "SELECT * FROM logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [row_to_log(r) for r in rows]


def row_to_log(r: sqlite3.Row) -> LogEntry:
    results = []
    if r["mode"] == "keyframe":
        results = [Item(**i) for i in json.loads(r["results"])]
    elif r["mode"] == "transcript_semantic":
        results = [TranscriptItem(**i) for i in json.loads(r["results"])]

    return LogEntry(
        request_id=r["request_id"], query=r["query"], limit=r["result_limit"],
        mode=r["mode"], model=r["model"] or DEFAULT_MODEL,
        results=results,
        total=r["total"], timestamp=r["timestamp"],
    )
