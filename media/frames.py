from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from media.preprocess import preprocess


def _run_ffmpeg(video_path: Path, frame_numbers: list[int], codec: str,
               quality: int, temp_dir: Path) -> list[Path]:
    """Run one ffmpeg extraction. Returns frame paths on success, empty list on failure."""
    select_filter = "+".join(f"eq(n\\,{fn})" for fn in frame_numbers)
    result = subprocess.run([
        "ffmpeg", "-c:v", codec, "-i", str(video_path),
        "-vf", f"select='{select_filter}'", "-vsync", "0", "-an",
        "-q:v", str(quality), "-y", "-loglevel", "error",
        str(temp_dir / "frame_%05d.jpg"),
    ], capture_output=True, text=True)
    produced = sorted(temp_dir.glob("frame_*.jpg"))
    if result.returncode == 0 and len(produced) == len(frame_numbers):
        return produced
    if codec != "libdav1d":
        err = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "unknown"
        print(f"    {codec} failed: {err}")
    return []


def extract_frames(video_path: Path, frame_numbers: list[int], out_dir: Path,
                   decoder: str, quality: int, clean_overlay: bool = False) -> list[Path]:
    """Extract all frames with one ffmpeg call, and clean overlay if need"""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"keyframes_{video_path.stem}_"))
    try:
        produced = _run_ffmpeg(video_path, frame_numbers, decoder, quality, temp_dir)
        if not produced and decoder != "libdav1d":
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir = Path(tempfile.mkdtemp(prefix=f"keyframes_{video_path.stem}_"))
            # Fall back to CPU
            produced = _run_ffmpeg(video_path, frame_numbers, "libdav1d", quality, temp_dir)
            if produced:
                print(f"    (fell back to libdav1d)")

        if len(produced) != len(frame_numbers):
            raise RuntimeError(
                f"ffmpeg produced {len(produced)} frames, expected {len(frame_numbers)}"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, src in enumerate(produced):
            dst = out_dir / f"{i + 1:04}.jpg"
            shutil.move(str(src), dst)
            if clean_overlay:
                preprocess(dst)
            saved.append(dst)
        return saved
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
