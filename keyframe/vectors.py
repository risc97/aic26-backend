from pathlib import Path
import numpy as np
from .ids import vector_id

def retrieve_keyframe_vector(shard_dir: Path, video_id: str, keyframe_id: str) -> np.ndarray:
    vec_path = shard_dir / f"{video_id}.npy"
    ids_path = shard_dir / f"{video_id}.ids.npy"

    ids = np.asarray(np.load(ids_path, mmap_mode="r"))
    rows = np.flatnonzero(ids == vector_id(video_id, keyframe_id))

    row = int(rows[0])
    vecs = np.load(vec_path, mmap_mode="r")
    return np.ascontiguousarray(vecs[row:row + 1], dtype=np.float32)
