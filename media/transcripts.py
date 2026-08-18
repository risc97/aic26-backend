from __future__ import annotations

import csv
from pathlib import Path

TRANSCRIPT_SUFFIX = ".csv"


def find_transcript(transcripts_dir: Path, video_id: str) -> Path | None:
    path = transcripts_dir / f"{video_id}{TRANSCRIPT_SUFFIX}"
    return path if path.is_file() else None


def read_transcript(transcript_file: Path) -> list[tuple[int, int, str]]:
    """Read CSV into a list of rows."""
    with open(transcript_file, newline="", encoding="utf-8") as f:
        return [
            (int(row["time_start_ms"]), int(row["time_end_ms"]), row["text"])
            for row in csv.DictReader(f)
        ]
