from __future__ import annotations

import timm

from .base import Encoder

MODEL_NAME = "convnext_base.dinov3_lvd1689m"


class DINOv3(Encoder):
    """Image-only encoder; serves /similar/visual."""

    MODEL_NAME = MODEL_NAME
    EMBED_DIM = 1024
    IMAGE_SIZE = 224

    def _create_model_and_transforms(self, *, pretrained: bool = False, precision: str = "fp32"):
        model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
        if precision == "fp16":
            model = model.half()
        cfg = timm.data.resolve_data_config({}, model=model)
        # squash instead of center crop so the whole 16:9 keyframe is embedded
        return model, timm.data.create_transform(**{**cfg, "crop_mode": "squash"})

    def _forward_image(self, pixel_values):
        return self.model(pixel_values)
