from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from .score import ObjectQuery, evaluate, prompts, verdict


class StageCache:
    """Corpus-wide verdicts for one stage spec, memoised on that spec.

    A stage costs one pass over the shards — the only expensive thing this
    endpoint does. Unlike a cosine, a stage's verdict depends on nothing but
    its own spec, so a session that keeps editing its last stage re-scans
    nothing. Roughly 5 bytes per keyframe per cached stage.

    Everything runs on the single-threaded search pool, so no lock.
    """

    def __init__(self, store, encoder, capacity: int = 64):
        self.store, self.encoder = store, encoder
        self.capacity = capacity
        self._cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, objects: Sequence[ObjectQuery], nms_iou: float = 0.5):
        """-> (passes, scores), both length len(store), in store.ids order."""
        key = (tuple(objects), round(nms_iou, 3))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        value = self._scan(list(objects), nms_iou)
        self._cache[key] = value
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return value

    def _scan(self, objects, nms_iou):
        passes = np.zeros(len(self.store), dtype=bool)
        scores = np.zeros(len(self.store), dtype=np.float32)
        queries = self.encoder.encode(prompts(objects))
        for rows, emb, box, aux in self.store.chunks():
            probs, ok = evaluate(emb, box, aux, queries, objects)
            p, s = verdict(box, probs, ok, objects, nms_iou)
            passes[rows], scores[rows] = p.numpy(), s.numpy()
        return passes, scores
