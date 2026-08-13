from __future__ import annotations

import numpy as np
import open_clip
import torch
import torch.nn.functional as F


class OpenClipEncoder:
    """Wrapper over open_clip dual encoder (image + text towers)."""

    def __init__(self, model_name: str, pretrained: str, device: str = "cuda",
                 amp: bool = True, tag: str | None = None):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        model.eval()

        self.model = model
        self.model_name = model_name
        self.pretrained = pretrained
        self.tag = tag or model_name.lower()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = torch.device(device)
        self.embed_dim = int(open_clip.get_model_config(model_name)["embed_dim"])
        # fp32 weights + fp16 autocast: same speed as half weights, no overflow risk
        self.amp = amp and self.device.type == "cuda"

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.amp)

    @torch.inference_mode()
    def encode_images(self, pixel_values: torch.Tensor) -> np.ndarray:
        """(B, 3, H, W) preprocessed tensor -> (B, embed_dim) float32."""
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        with self._autocast():
            feats = self.model.encode_image(pixel_values)
        return F.normalize(feats.float(), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """List of queries -> (N, embed_dim) float32."""
        tokens = self.tokenizer(texts).to(self.device)
        with self._autocast():
            feats = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1).cpu().numpy()
