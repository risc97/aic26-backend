from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES = (".webm", ".mp4", ".mkv", ".mov")

def find_videos(videos_dir: Path,
                suffixes: tuple[str, ...] = VIDEO_SUFFIXES) -> dict[str, Path]:
    if not videos_dir.is_dir():
        return {}
    videos: dict[str, Path] = {}
    for path in sorted(videos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in videos:
            print(f"[{path.stem}] duplicate video, using {videos[path.stem].name} and ignoring {path.name}")
            continue
        videos[path.stem] = path
    return videos


def list_video_ids(videos_dir: Path,
                   suffixes: tuple[str, ...] = VIDEO_SUFFIXES) -> list[str]:
    return list(find_videos(videos_dir, suffixes))
