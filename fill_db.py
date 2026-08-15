from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db import MetadataDatabase
from media.keyframes import find_keyframes
from media.probe import probe_video
from media.scenes import read_scenes, select_frames
from media.videos import find_videos
from config import (METADATA_DB_PATH, VIDEOS_DB_PATH, SCENES_DB_PATH, KEYFRAMES_DB_PATH,
                    KEYFRAME_MODE, stored_path)

SCENES_SUFFIX = ".scenes.txt"


def fill_video(db: MetadataDatabase, video_id: str, video_path: Path,
               scenes_file: Path, keyframes_dir: Path, force: bool,
               mode: str = KEYFRAME_MODE) -> bool:
    """Rebuild one video's rows from its keyframes, scenes and the video"""
    images = find_keyframes(keyframes_dir, video_id)
    if not images:
        print(f"[{video_id}] no keyframes in {keyframes_dir / video_id}, skipping")
        return False
    if not scenes_file.is_file():
        print(f"[{video_id}] no scenes at {scenes_file}, skipping")
        return False

    if not force and db.keyframes.count_by_video(video_id) == len(images):
        print(f"[{video_id}] already in the database, skipping")
        return True

    kept = select_frames(read_scenes(scenes_file), mode)
    if len(kept) != len(images):
        print(f"[{video_id}] FAILED: {len(images)} keyframe(s) on disk but "
              f"{len(kept)} scene(s) selected in {mode} mode; re-extract the "
              f"video or check --mode")
        return False

    fps, duration_ms = probe_video(video_path)
    if duration_ms == 0:
        print(f"[{video_id}] warning: could not probe {video_path}, "
              f"assuming {fps:.3f} fps")

    db.videos.upsert(video_id, stored_path(video_path), duration_ms, fps)
    db.keyframes.delete_by_video(video_id)
    db.segments.delete_by_video(video_id)

    segments = db.segments.create_many(video_id, [span for span, _ in kept])
    db.keyframes.create_many([
        (video_id, image.stem, frame, int(round(frame / fps * 1000)),
         stored_path(image), segment.segment_id)
        for (_, frame), segment, image in zip(kept, segments, images)
    ])

    print(f"[{video_id}] stored {len(images)} keyframes and {len(segments)} "
          f"segments ({fps:.3f} fps)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=METADATA_DB_PATH)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DB_PATH)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DB_PATH)
    parser.add_argument("--keyframes-dir", type=Path, default=KEYFRAMES_DB_PATH)
    parser.add_argument("--video", action="append", metavar="VIDEO_ID",
                        help="only store this video id (repeatable)")
    parser.add_argument("--mode", choices=("left", "mid", "right"),
                        default=KEYFRAME_MODE,
                        help=f"which frame in each scene was taken "
                             f"(default: {KEYFRAME_MODE}); must match the "
                             f"--mode extract_keyframes.py ran with")
    parser.add_argument("--force", action="store_true",
                        help="rewrite videos that are already in the database")
    args = parser.parse_args()

    videos = find_videos(args.videos_dir)
    if args.video:
        wanted = set(args.video)
        videos = {vid: path for vid, path in videos.items() if vid in wanted}
    if not videos:
        print(f"no videos found in {args.videos_dir}")
        return 1

    failed = []
    with MetadataDatabase(args.db) as db:
        db.init_schema()
        for video_id, video_path in videos.items():
            scenes_file = args.scenes_dir / f"{video_id}{SCENES_SUFFIX}"
            if not fill_video(db, video_id, video_path, scenes_file,
                              args.keyframes_dir, args.force, mode=args.mode):
                failed.append(video_id)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
