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


def preprocess(image_path: Path) -> None:
    """Black out noise regions, in place"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"could not read {image_path}")
    height, width = image.shape[:2]

    # bottom ticker bar
    x, y, w, h = 0, int(height*0.90625), width, int(height*0.052)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    # top-right badge
    x, y, w, h = int(width*0.8138), int(height*0.0729), int(width*0.1054), int(height*0.084)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

    cv2.imwrite(str(image_path), image)
