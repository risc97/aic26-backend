from __future__ import annotations

from pathlib import Path

from .encoder import OpenClipEncoder
from .c2lip import C2Lip
from .siglip2 import SigLip2, VARIANT, PRETRAINED, VARIANTS

MODEL_CHOICES = ("c2lip", "siglip2")

__all__ = [
    "OpenClipEncoder", "C2Lip", "SigLip2", "load_encoder",
    "MODEL_CHOICES", "VARIANT", "PRETRAINED", "VARIANTS",
]


def load_encoder(model: str, *, ckpt: str | Path | None = None,
                 variant: str = VARIANT,
                 device: str = "cuda", amp: bool = True) -> OpenClipEncoder:
    """Build the requested encoder.

    c2lip   -- needs `ckpt`, the authors' fine-tune of ViT-B-16-SigLIP.
    siglip2 -- `variant` picks the open_clip config; `ckpt` overrides the
               'webli' hub weights with a local file.
    """
    if model == "c2lip":
        if ckpt is None:
            raise ValueError("c2lip needs --ckpt")
        return C2Lip(ckpt, device=device, amp=amp)
    if model == "siglip2":
        return SigLip2(variant, pretrained=ckpt or PRETRAINED,
                       device=device, amp=amp)
    raise ValueError(f"unknown model {model!r}; choose one of {MODEL_CHOICES}")
