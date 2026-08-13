from __future__ import annotations

from pathlib import Path

from .encoder import OpenClipEncoder

MODEL_NAME = "ViT-B-16-SigLIP"
class C2Lip(OpenClipEncoder):
    """The authors' SigLIP fine-tune, loaded from a checkpoint."""
    def __init__(self, ckpt: str | Path, device: str = "cuda", amp: bool = True):
        super().__init__(MODEL_NAME, str(ckpt), device=device, amp=amp, tag="c2lip")
