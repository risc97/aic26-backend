from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@contextmanager
def _suppress_root_logger():
    log = logging.getLogger("root")
    old = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(old)

class Encoder(nn.Module):
    """Base class for image encoders backed by open_clip / timm.

    Subclasses override:
        MODEL_NAME  (str)
        EMBED_DIM   (int)
        IMAGE_SIZE  (int)
        _create_model_and_transforms()  -> (nn.Module, callable)
        _forward_image()                -> torch.Tensor
        _forward_text()                 -> torch.Tensor  (optional)
        _get_tokenizer()                -> callable       (optional)
    """

    MODEL_NAME: str = ""
    EMBED_DIM: int = 0
    IMAGE_SIZE: int = 0

    def __init__(self, ckpt: str | Path | None = None, device: str = "cuda",
                 precision: str | None = None, compile: bool = False):
        super().__init__()

        if ckpt is not None:
            torch.serialization.add_safe_globals([
                np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType,
                np._core.multiarray.scalar,
            ])
            raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            if isinstance(raw, dict) and "state_dict" in raw:
                sd = raw["state_dict"]
            else:
                sd = raw
            sd = {k.removeprefix("module."): v for k, v in sd.items()}

        pretrained = ckpt is None  # use source weights when no checkpoint given
        if precision is None:
            precision = "fp16" if torch.device(device).type == "cuda" else "fp32"
        with _suppress_root_logger():
            model, preprocess = self._create_model_and_transforms(
                pretrained=pretrained, precision=precision,
            )
        if ckpt is not None:
            model_keys = set(model.state_dict())
            extra = sorted(set(sd) - model_keys)
            if extra:
                print(f"note: ignoring {len(extra)} checkpoint key(s) not in "
                      f"{type(self).__name__}: {extra}")
                sd = {k: v for k, v in sd.items() if k in model_keys}
            try:
                model.load_state_dict(sd, strict=True)
            except RuntimeError as e:
                raise ValueError(
                    f"checkpoint {ckpt} does not match {type(self).__name__} "
                    f"({self.MODEL_NAME}); is it the right architecture?"
                ) from e

        if compile:
            model = torch.compile(model, mode="reduce-overhead")

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = self._get_tokenizer()

        self.to(device)
        self.eval()

    # --- subclass hooks ---

    def _create_model_and_transforms(self, *, pretrained: bool = False) -> tuple[nn.Module, callable]:
        raise NotImplementedError

    def _forward_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _forward_text(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no text encoder")

    def _get_tokenizer(self) -> callable | None:
        return None

    # --- shared ---

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def encode_images(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        feats = self._forward_image(pixel_values)
        return F.normalize(feats.float(), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.tokenizer is None:
            raise NotImplementedError(f"{type(self).__name__} has no text encoder")
        tokens = self.tokenizer(texts).to(self.device)
        feats = self._forward_text(tokens)
        return F.normalize(feats.float(), dim=-1).cpu().numpy()