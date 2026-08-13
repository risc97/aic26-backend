from __future__ import annotations

from pathlib import Path

from .search import KeyframeSearcher


class SearcherRegistry:
    """One KeyframeSearcher per model, built on first use.
    """

    def __init__(self, index_paths: dict[str, Path], db_path: Path,
                 ckpt_paths: dict[str, Path | None] | None = None,
                 device: str = "cuda"):
        self.index_paths = index_paths
        self.db_path = db_path
        self.ckpt_paths = ckpt_paths or {}
        self.device = device
        self._searchers: dict[str, KeyframeSearcher] = {}

    def available(self) -> list[str]:
        """Models whose index (and checkpoint, if it needs one) is on disk."""
        return [m for m in self.index_paths if not self._missing(m)]

    def _missing(self, model: str) -> list[Path]:
        index_path = self.index_paths.get(model)
        if index_path is None:
            return []
        ckpt = self.ckpt_paths.get(model)
        return [p for p in (index_path, ckpt) if p is not None and not p.exists()]

    def get(self, model: str) -> KeyframeSearcher:
        searcher = self._searchers.get(model)
        if searcher is not None:
            return searcher

        if model not in self.index_paths:
            raise KeyError(f"unknown model {model!r}")
        missing = self._missing(model)
        if missing:
            raise FileNotFoundError(", ".join(str(p) for p in missing))

        print(f"Loading {model} keyframe searcher on {self.device}...")
        searcher = KeyframeSearcher(
            self.index_paths[model], self.db_path, model,
            ckpt=self.ckpt_paths.get(model), device=self.device,
        )
        self._searchers[model] = searcher
        return searcher

    def close(self) -> None:
        for searcher in self._searchers.values():
            searcher.close()
        self._searchers.clear()
