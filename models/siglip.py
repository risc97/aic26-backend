from __future__ import annotations

import open_clip

from .base import Encoder

MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"


class SigLIP(Encoder):
    MODEL_NAME = MODEL_NAME

    def _create_model_and_transforms(self, *, pretrained: bool = False, precision: str = "fp32"):
        pt = PRETRAINED if pretrained else None
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=pt, precision=precision,
        )
        self.EMBED_DIM = int(open_clip.get_model_config(MODEL_NAME)["embed_dim"])
        return model, preprocess

    def _forward_image(self, pixel_values):
        return self.model.encode_image(pixel_values)

    def _forward_text(self, tokens):
        return self.model.encode_text(tokens)

    def _get_tokenizer(self):
        return open_clip.get_tokenizer(MODEL_NAME)
