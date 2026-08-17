from __future__ import annotations

from .siglip import SigLIP
from .base import Encoder
from .pe import PerceptionEncoder
from .siglip2 import SigLip2
from .gte import Gte

__all__ = ["Encoder", "SigLIP", "SigLip2", "PerceptionEncoder", "MODEL_CHOICES"]

MODEL_CHOICES = {
    "siglip": SigLIP,
    "siglip2": SigLip2,
    "pe": PerceptionEncoder,
    "gte": Gte
}
