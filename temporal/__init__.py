"""
POST /query/temporal

Find sequences with keyframe 1 is action 1, keyframe 2 is action 2
{"stages": [{"query": "action 1"},
            {"query": "action 2"}]}

POST /query/temporal/detect

Find sequences with keyframe 1 has at least 1 motorbike, keyframe 2 has at least 1 bus and 2 person
{"stages": [{"objects": [{"phrase": "motorbike"}]}, 
            {"objects": [{"phrase": "bus"},
                         {"phrase": "person", "min_count": 2}]}]}

"""

from .types import Chain, Hop, Stage, StageMatch, TemporalHit, TemporalParams
from .detect import detect_temporal_search

__all__ = [
    "Chain", "Hop", "Stage", "StageMatch", "TemporalHit", "TemporalParams",
    "temporal_search",
]


def temporal_search(*args, **kwargs):
    # imported lazily
    from .search import temporal_search as _search
    return _search(*args, **kwargs)