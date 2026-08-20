from __future__ import annotations

import sqlite3
from pathlib import Path


class SearcherRegistry:
    """One searcher instance per model, built on first use."""

    def __init__(self, index_paths: dict[str, Path], db_path: Path, searcher_cls,
                 ckpt_paths: dict[str, Path | None] | None = None,
                 device: str = "cuda"):
        self.index_paths = index_paths
        self.db_path = db_path
        self.searcher_cls = searcher_cls
        self.ckpt_paths = ckpt_paths or {}
        self.device = device
        self._searchers = {}

    def available(self) -> list[str]:
        """Models whose index (and checkpoint, if it needs one) is on disk."""
        return [m for m in self.index_paths if not self._missing(m)]

    def _missing(self, model: str) -> list[Path]:
        index_path = self.index_paths.get(model)
        if index_path is None:
            return []
        ckpt = self.ckpt_paths.get(model)
        return [p for p in (index_path, ckpt) if p is not None and not p.exists()]

    def get(self, model: str):
        searcher = self._searchers.get(model)
        if searcher is not None:
            return searcher

        if model not in self.index_paths:
            raise KeyError(f"unknown model {model!r}")
        missing = self._missing(model)
        if missing:
            raise FileNotFoundError(", ".join(str(p) for p in missing))

        print(f"Loading {model} searcher ({self.searcher_cls.__name__}) on {self.device}...")
        searcher = self.searcher_cls(
            self.index_paths[model], self.db_path, model,
            ckpt=self.ckpt_paths.get(model), device=self.device,
        )
        self._searchers[model] = searcher
        return searcher

    def close(self) -> None:
        for searcher in self._searchers.values():
            searcher.close()
        self._searchers.clear()


class TableSearcherRegistry:
    """ Searcher for FTS5 exact search over ocr, transcripts """

    def __init__(self, db_path: Path, table: str, searcher_cls, name: str = "exact"):
        if not table.isidentifier():
            raise ValueError(f"table {table!r} is not a plain identifier")
        self.db_path = Path(db_path)
        self.table = table
        self.searcher_cls = searcher_cls
        self.name = name
        self._searcher = None

    def count(self) -> int:
        """Rows in the backing table; 0 if the database or the table is missing."""
        if not self.db_path.exists():
            return 0
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0  # schema not initialised yet
        finally:
            conn.close()

    def available(self) -> list[str]:
        return [self.name] if self.count() else []

    def get(self, model: str | None = None):
        """`model` is accepted and ignored; there is only one searcher."""
        if self._searcher is None:
            if not self.count():
                raise FileNotFoundError(
                    f"table '{self.table}' in {self.db_path} is empty or missing"
                )
            print(f"Loading {self.searcher_cls.__name__} on '{self.table}'...")
            self._searcher = self.searcher_cls(self.db_path)
        return self._searcher

    def close(self) -> None:
        if self._searcher is not None:
            self._searcher.close()
            self._searcher = None
