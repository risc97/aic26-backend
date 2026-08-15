from __future__ import annotations

from pathlib import Path


def read_scenes(scenes_file: Path) -> list[tuple[int, int]]:
    """Read "<frame_start> <frame_end>" lines into a list of spans."""
    spans = []
    with open(scenes_file) as f:
        for line in f:
            if not line.strip():
                continue
            left, right = line.split()
            spans.append((int(left), int(right)))
    return spans


def pick_frame(span: tuple[int, int], mode: str) -> int:
    if mode == "left":
        return span[0]
    if mode == "right":
        return span[1]
    return (span[0] + span[1]) // 2  # mid


def select_frames(spans: list[tuple[int, int]],
                  mode: str) -> list[tuple[tuple[int, int], int]]:
    seen: set[int] = set()
    kept: list[tuple[tuple[int, int], int]] = []
    for span in spans:
        frame = pick_frame(span, mode)
        if frame in seen:
            continue
        seen.add(frame)
        kept.append((span, frame))
    return kept
