from __future__ import annotations

import subprocess
from pathlib import Path


def probe_video(video_path: Path) -> tuple[float, int]:
    """Return (fps, duration_ms) for a video."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]

    fps_str = lines[0] if lines else ""
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 25.0
    else:
        fps = float(fps_str) if fps_str else 25.0

    duration_str = lines[1] if len(lines) > 1 else ""
    try:
        duration_ms = int(round(float(duration_str) * 1000))
    except ValueError:
        duration_ms = 0

    return fps, duration_ms
