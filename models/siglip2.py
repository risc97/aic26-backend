from __future__ import annotations

from pathlib import Path

import open_clip

from .encoder import OpenClipEncoder

VARIANTS = tuple(m for m in open_clip.list_models() if "SigLIP2" in m)
VARIANT = "ViT-SO400M-16-SigLIP2-512"
PRETRAINED = "webli"


class SigLip2(OpenClipEncoder):
    def __init__(self, variant: str = VARIANT, pretrained: str | Path = PRETRAINED,
                 device: str = "cuda", amp: bool = True):
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown SigLIP2 variant {variant!r}; choose one of:\n  "
                + "\n  ".join(VARIANTS)
            )
        super().__init__(variant, str(pretrained), device=device, amp=amp, tag="siglip2")
