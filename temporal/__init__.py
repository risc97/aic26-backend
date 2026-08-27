from .types import Chain, Hop, Stage, StageMatch, TemporalHit, TemporalParams

__all__ = [
    "Chain", "Hop", "Stage", "StageMatch", "TemporalHit", "TemporalParams",
    "temporal_search",
]


def temporal_search(*args, **kwargs):
    # imported lazily
    from .search import temporal_search as _search
    return _search(*args, **kwargs)