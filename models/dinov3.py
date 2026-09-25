from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch

from config import DATA_PATH

MODEL_NAME = "convnext_base.dinov3_lvd1689m"
IMAGE_SIZE = 224
# compiled from the source weights on first use; delete it to force a rebuild
ONNX_PATH = DATA_PATH / "checkpoints" / f"{MODEL_NAME}.int8.onnx"


def compile_onnx(dst: Path = ONNX_PATH) -> None:
    """Export to ONNX, then quantize to INT8: dynamic activations, per-channel weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        fp32, prep, int8 = (Path(tmp) / name for name in ("fp32.onnx", "prep.onnx", "int8.onnx"))
        batch = torch.export.Dim("batch", min=1, max=4096)
        torch.onnx.export(model, (torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE),), fp32, dynamo=True,
                          input_names=["pixel_values"], output_names=["embedding"],
                          dynamic_shapes=({0: batch},), external_data=False, verbose=False)
        # ORT's symbolic shape inference asserts on the exported graph's Range node;
        # onnx's own shape inference covers this model
        quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
        # Conv stays fp32: quantizing it (ConvInteger) made the model ~10x slower.
        # 8-bit weights, no reduce_range: AVX2 CPUs without VNNI saturate slightly in u8s8
        # GEMMs, but still measured closer to fp32 than 7-bit weights, at full speed
        quantize_dynamic(str(prep), str(int8), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"])
        os.replace(int8, dst)  # atomic, so an interrupted build never leaves a broken model


class DINOv3:
    """Image-only encoder served from an INT8 ONNX model on CPU; compiled on first use.

    Drop-in for Encoder in KeyframeSearcher. `device` and `precision` are accepted for
    that interface only: the dynamic INT8 kernels are CPU-only.
    """

    MODEL_NAME = MODEL_NAME
    EMBED_DIM = 1024
    IMAGE_SIZE = IMAGE_SIZE

    def __init__(self, ckpt: str | Path | None = None, device: str = "cpu",
                 precision: str | None = None):
        if ckpt is not None:
            raise ValueError(f"{type(self).__name__} is compiled from {MODEL_NAME}'s "
                             f"source weights; checkpoint {ckpt} is not supported")
        if not ONNX_PATH.exists():
            print(f"Compiling {MODEL_NAME} to INT8 ONNX at {ONNX_PATH}...")
            compile_onnx(ONNX_PATH)

        options = ort.SessionOptions()
        # ORT's default pool pins its threads and ran ~35% slower; torch's default is the
        # physical core count
        options.intra_op_num_threads = torch.get_num_threads()
        self.session = ort.InferenceSession(str(ONNX_PATH), options,
                                            providers=["CPUExecutionProvider"])

        cfg = timm.data.resolve_data_config(
            pretrained_cfg=timm.models.get_pretrained_cfg(MODEL_NAME).to_dict())
        # squash instead of center crop, as preprocessing embedded the whole 16:9 keyframe
        self.preprocess = timm.data.create_transform(**{**cfg, "crop_mode": "squash"})

    def to(self, device) -> DINOv3:
        return self  # stays on CPU

    def encode_images(self, pixel_values) -> np.ndarray:
        x = np.ascontiguousarray(pixel_values, dtype=np.float32)
        # one image per run: dynamic quantization scales activations per tensor, so a batch
        # would couple images' embeddings and cost accuracy (and isn't faster on CPU)
        feats = np.concatenate([self.session.run(None, {"pixel_values": x[i:i + 1]})[0]
                                for i in range(len(x))])
        return feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-12)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no text encoder")
