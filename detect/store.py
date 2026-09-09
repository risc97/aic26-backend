from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
PAD_OBJECTNESS = -1e4
CHUNK = 4096

class DetectionStore:
    """Memory-mapped OWLv2 box shards, addressed by packed keyframe id."""
    def __init__(self, shard_dir: Path):
        shard_dir = Path(shard_dir)
        stems = sorted(p.name[:-len(".ids.npy")]
                       for p in shard_dir.glob("*.ids.npy"))
        if not stems:
            raise FileNotFoundError(f"no detection shards in {shard_dir}")

        self.emb, self.box, self.aux, id_chunks = [], [], [], []
        for stem in stems:
            id_chunks.append(np.load(shard_dir / f"{stem}.ids.npy"))
            self.emb.append(np.load(shard_dir / f"{stem}.emb.npy", mmap_mode="r"))
            self.box.append(np.load(shard_dir / f"{stem}.box.npy", mmap_mode="r"))
            self.aux.append(np.load(shard_dir / f"{stem}.aux.npy", mmap_mode="r"))

        lengths = [len(c) for c in id_chunks]
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self.ids = np.concatenate(id_chunks).astype(np.int64)
        self._order = np.argsort(self.ids)
        self._sorted = self.ids[self._order]
        self.dim = int(self.emb[0].shape[2])
        self.top_k = int(self.emb[0].shape[1])

    def __len__(self) -> int:
        return len(self.ids)

    def rows_for(self, vector_ids: np.ndarray) -> np.ndarray:
        """Packed ids -> global rows; -1 where the id has no detections."""
        pos = np.searchsorted(self._sorted, vector_ids)
        pos = np.clip(pos, 0, len(self._sorted) - 1)
        rows = self._order[pos]
        rows[self.ids[rows] != vector_ids] = -1
        return rows

    def _shard_of(self, rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.offsets, rows, side="right") - 1

    def chunks(self, rows: np.ndarray | None = None):
        """Yield (global_rows, emb, box, aux) as float32 torch tensors."""
        if rows is None:
            for s, arr in enumerate(self.emb):
                base = int(self.offsets[s])
                for i in range(0, len(arr), CHUNK):
                    local = np.arange(i, min(i + CHUNK, len(arr)))
                    yield self._gather(s, local, base + local)
            return

        rows = np.asarray(rows)
        rows = np.unique(rows[rows >= 0])
        shards = self._shard_of(rows)
        for s in np.unique(shards):
            sel = rows[shards == s]
            local = sel - int(self.offsets[s])
            for i in range(0, len(local), CHUNK):
                part = local[i:i + CHUNK]
                yield self._gather(s, part, sel[i:i + CHUNK])

    def _gather(self, shard: int, local: np.ndarray, global_rows: np.ndarray):
        def take(arr):
            return torch.from_numpy(
                np.ascontiguousarray(arr[shard][local])).float()
        return global_rows, take(self.emb), take(self.box), take(self.aux)
