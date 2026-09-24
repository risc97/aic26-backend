from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from media.preprocess import preprocess
from media.probe import probe_codec, probe_frame_rate

import torch
GPU = torch.cuda.is_available()

# codec -> (GPU decoder, CPU decoder)
DECODERS = {
    "av1":  ("av1_cuvid", "libdav1d"),
    "h264": ("h264", "h264"),
    "hevc": ("hevc_cuvid", "hevc"),
}


def _run_ffmpeg(video_path: Path, frame_numbers: list[int], codec: str,
               quality: int, temp_dir: Path, fps: str = "") -> list[Path]:
    """Run one ffmpeg extraction. Returns frame paths on success, empty list on failure."""
    select_filter = "+".join(f"eq(n\\,{fn})" for fn in frame_numbers)
    vf = f"select='{select_filter}'"
    if fps:
        vf = f"fps={fps},{vf}"

    decode_args = ["-c:v", codec] if codec else []
    result = subprocess.run([
        "ffmpeg", *decode_args, "-i", str(video_path),
        "-vf", vf, "-vsync", "0", "-an",
        "-q:v", str(quality), "-y", "-loglevel", "error",
        str(temp_dir / "frame_%05d.jpg"),
    ], capture_output=True, text=True)
    produced = sorted(temp_dir.glob("frame_*.jpg"))
    if result.returncode == 0 and len(produced) == len(frame_numbers):
        return produced
    if codec:
        if result.returncode != 0:
            err = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "unknown"
        else:
            err = f"decoded {len(produced)}/{len(frame_numbers)} frames"
        print(f"    {codec} failed: {err}")
    return []


def extract_frames(video_path: Path, frame_numbers: list[int], out_dir: Path,
                   decoder: str, quality: int, clean_overlay: bool = False) -> list[Path]:
    """Extract all frames with one ffmpeg call, and clean overlay if need"""
    if decoder == "auto":
        gpu_dec, cpu_dec = DECODERS.get(probe_codec(video_path), ("", ""))
        decoder = gpu_dec if GPU else cpu_dec
    fps = probe_frame_rate(video_path)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"keyframes_{video_path.stem}_"))
    try:
        produced = _run_ffmpeg(video_path, frame_numbers, decoder, quality, temp_dir,fps)
        if not produced and decoder:
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir = Path(tempfile.mkdtemp(prefix=f"keyframes_{video_path.stem}_"))
            # Fall back to CPU
            produced = _run_ffmpeg(video_path, frame_numbers, "", quality, temp_dir)
            if produced:
                print(f"    (fell back to ffmpeg's default decoder)")

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
