from .types import Gap, Stage, StageMatch, TemporalHit, TemporalParams
from .dp import align

__all__ = [
    "Gap", "Stage", "StageMatch", "TemporalHit", "TemporalParams",
    "align", "temporal_search",
]

def temporal_search(*args, **kwargs):
    from .search import temporal_search as _search
    return _search(*args, **kwargs)
