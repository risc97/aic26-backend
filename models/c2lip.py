from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

MODEL_NAME = "ViT-B-16-SigLIP"
EMBED_DIM = 768
IMAGE_SIZE = 224


class C2Lip:
    def __init__(self, ckpt: str | Path, device: str = "cuda", amp: bool = True):
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=str(ckpt), device=device,
        )
        model.eval()

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.device = torch.device(device)
        # fp32 weights + fp16 autocast: same speed as half weights, no overflow risk
        self.amp = amp and self.device.type == "cuda"

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.amp)

    @torch.inference_mode()
    def encode_images(self, pixel_values: torch.Tensor) -> np.ndarray:
        """(B, 3, 224, 224) preprocessed tensor -> (B, 768) float32."""
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        with self._autocast():
            feats = self.model.encode_image(pixel_values)
        return F.normalize(feats.float(), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """List of queries -> (N, 768) float32."""
        tokens = self.tokenizer(texts).to(self.device)
        with self._autocast():
            feats = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1).cpu().numpy()
