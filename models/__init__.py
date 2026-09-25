from __future__ import annotations

from .siglip import SigLIP
from .base import Encoder
from .pe import PerceptionEncoder
from .siglip2 import SigLip2
from .gte import Gte
from .dinov3 import DINOv3

__all__ = ["Encoder", "SigLIP", "SigLip2", "PerceptionEncoder", "DINOv3", "MODEL_CHOICES"]

MODEL_CHOICES = {
    "siglip": SigLIP,
    "siglip2": SigLip2,
    "pe": PerceptionEncoder,
    "gte": Gte,
    "dinov3": DINOv3,
}
