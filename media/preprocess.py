from __future__ import annotations

import json
from pathlib import Path

import cv2

NOISY_AUTHOR = "60 Giây Official"


def has_overlay(media_info_file: Path) -> bool:
    if not media_info_file.is_file():
        return False
    with open(media_info_file, encoding="utf-8") as f:
        info = json.load(f)
    return info.get("author") == NOISY_AUTHOR


def overlay_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Noise regions as (x, y, w, h), in fractions of the frame"""
    return [
        # bottom ticker bar
        (0, int(height*0.90625), width, int(height*0.052)),
        # top-right badge
        (int(width*0.8138), int(height*0.0729),
         int(width*0.1054), int(height*0.084)),
    ]


def preprocess(image_path: Path) -> None:
    """Black out noise regions, in place"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"could not read {image_path}")
    for x, y, w, h in overlay_boxes(*image.shape[:2]):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.imwrite(str(image_path), image)