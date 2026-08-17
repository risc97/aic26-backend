from __future__ import annotations

import timm
from timm.data import create_transform, resolve_data_config

from .base import Encoder

MODEL_NAME = "vit_pe_core_large_patch14_336.fb"


class PerceptionEncoder(Encoder):
    MODEL_NAME = MODEL_NAME

    def _create_model_and_transforms(self, *, pretrained: bool = False):
        model = timm.create_model(MODEL_NAME, pretrained=pretrained)
        config = resolve_data_config({}, model=model)
        preprocess = create_transform(**config, is_training=False)
        self.EMBED_DIM = model.embed_dim
        self.IMAGE_SIZE = config["input_size"][1]
        return model, preprocess

    def _forward_image(self, pixel_values):
        feats = self.model.forward_features(pixel_values)
        return self.model.forward_head(feats, pre_logits=True)
