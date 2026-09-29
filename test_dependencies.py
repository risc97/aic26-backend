#!/usr/bin/env python3
"""Verify the aic26 python environment: version pins, wheel ABIs, GPU stack.

Run with the venv active, from either aic26-backend/ or aic26-preprocessing/:

    python test_dependencies.py            # auto-detect profile and GPU
    python test_dependencies.py --strict   # treat warnings as failures
    python test_dependencies.py -v         # full tracebacks

Exit code is 1 if anything FAILed (or WARNed under --strict).
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# --- expected pins (keep in sync with requirements.txt) -----------------------
PY_EXPECTED = (3, 12)
TORCH_EXPECTED = "2.13.0"
TORCHVISION_EXPECTED = "0.28.0"
PADDLE_EXPECTED = "3.3.1"
NUMPY_MIN = (2, 0)
NUMPY_MAX_PREPROCESSING = (2, 4)  # paddlex pins numpy<2.4
GPU_EXPECTED_ARCH = "sm_86"       # RTX 3060 = Ampere GA106

# --- tiny harness ------------------------------------------------------------


class Skip(Exception):
    """Check does not apply to this environment."""


class Warn(Exception):
    """Non-fatal problem."""


ROWS: list[tuple[str, str, str]] = []
VERBOSE = False
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
TINT = {"PASS": "32", "WARN": "33", "FAIL": "31", "SKIP": "90", "INFO": "36"}


def _emit(status: str, name: str, detail: str) -> None:
    ROWS.append((status, name, detail))
    tag = f"\033[{TINT[status]}m{status:<4}\033[0m" if COLOR else f"{status:<4}"
    print(f"  {tag}  {name:<34} {detail}")


def run(name: str, fn, *, optional: bool = False) -> None:
    try:
        _emit("PASS", name, fn() or "")
    except Skip as e:
        _emit("SKIP", name, str(e))
    except Warn as e:
        _emit("WARN", name, str(e))
    except Exception as e:  # noqa: BLE001 - a failed check must not stop the run
        detail = f"{type(e).__name__}: {e}"
        _emit("WARN" if optional else "FAIL", name, detail)
        if VERBOSE:
            traceback.print_exc()


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m" if COLOR else f"\n{title}")


def version_of(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        raise Skip(f"{dist} not installed") from None


def tuple_of(ver: str) -> tuple[int, ...]:
    out = []
    for part in ver.split("+")[0].split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


# --- environment -------------------------------------------------------------


def detect_profile() -> str:
    text = ""
    req = Path("requirements.txt")
    if req.exists():
        text = req.read_text()
    if "paddleocr" in text or "k2==" in text:
        return "preprocessing"
    if text:
        return "backend"
    return "preprocessing" if importlib.util.find_spec("paddleocr") else "backend"


def check_python() -> str:
    got = sys.version_info[:2]
    detail = f"{sys.version.split()[0]} at {sys.executable}"
    if got != PY_EXPECTED:
        raise Warn(f"expected {'.'.join(map(str, PY_EXPECTED))}, got {detail}")
    return detail


def check_venv() -> str:
    if sys.prefix == sys.base_prefix:
        raise Warn("not running inside a venv")
    return sys.prefix


def check_driver_path() -> str:
    """Nix wheels need libcuda.so from the impure driver path."""
    if not Path("/run/opengl-driver/lib").is_dir():
        raise Skip("/run/opengl-driver/lib absent (no NixOS GPU host)")
    if "/run/opengl-driver/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        raise Warn("driver dir exists but is not on LD_LIBRARY_PATH")
    return "on LD_LIBRARY_PATH"


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise Warn("ffmpeg not on PATH")
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0][:60]


# --- numpy / torch core ------------------------------------------------------


def check_numpy(profile: str) -> str:
    import numpy as np

    ver = tuple_of(np.__version__)
    if ver < NUMPY_MIN:
        raise Exception(f"numpy {np.__version__} < 2.0 -- models/base.py needs np._core")
    if profile == "preprocessing" and ver >= NUMPY_MAX_PREPROCESSING:
        raise Warn(f"numpy {np.__version__} exceeds paddlex's <2.4 cap")
    return np.__version__


def check_numpy_core_scalar() -> str:
    """models/base.py reaches into np._core.multiarray.scalar for torch.load."""
    import numpy as np

    core = getattr(np, "_core", None) or np.core
    if not hasattr(core, "multiarray") or not hasattr(core.multiarray, "scalar"):
        raise Exception(f"{core.__name__}.multiarray.scalar missing")
    return f"{core.__name__}.multiarray.scalar present"


def check_torch() -> str:
    import torch

    detail = f"{torch.__version__} (cuda build: {torch.version.cuda})"
    if tuple_of(torch.__version__) != tuple_of(TORCH_EXPECTED):
        raise Warn(f"expected {TORCH_EXPECTED} (k2/kaldifeat ABI), got {detail}")
    return detail


def check_torchvision() -> str:
    """A torch/torchvision mismatch shows up as a missing C++ symbol here."""
    import torch
    import torchvision

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
    if tuple_of(torchvision.__version__) != tuple_of(TORCHVISION_EXPECTED):
        raise Warn(f"expected {TORCHVISION_EXPECTED}, got {torchvision.__version__}")
    return f"{torchvision.__version__}, ops.nms OK"


def check_weights_only_load() -> str:
    """Reproduce the models/base.py checkpoint path end to end."""
    import numpy as np
    import torch

    core = getattr(np, "_core", None) or np.core
    torch.serialization.add_safe_globals(
        [np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType, core.multiarray.scalar]
    )
    payload = {"w": torch.zeros(2), "scale": np.float64(1.5), "dt": np.dtype("int64")}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        torch.save(payload, p)
        got = torch.load(str(p), map_location="cpu", weights_only=True)
    assert got["scale"] == 1.5
    return "torch.load(weights_only=True) with numpy scalars OK"


# --- GPU ---------------------------------------------------------------------


def check_cuda_available() -> str:
    import torch

    if not torch.cuda.is_available():
        raise Skip("no CUDA device visible")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    return f"{name} sm_{cap[0]}{cap[1]}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free"


def check_gpu_arch() -> str:
    import torch

    if not torch.cuda.is_available():
        raise Skip("no CUDA device visible")
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    if arch not in archs:
        raise Exception(f"wheel has no {arch} kernels; built for {archs}")
    if arch != GPU_EXPECTED_ARCH:
        raise Warn(f"device is {arch}, expected {GPU_EXPECTED_ARCH}")
    return f"{arch} in wheel arch list"


def check_cudnn_conv() -> str:
    """Exercises the bundled nvidia-cudnn-cu13 / nccl shared objects."""
    import torch

    if not torch.cuda.is_available():
        raise Skip("no CUDA device visible")
    x = torch.randn(8, 3, 32, 32, device="cuda")
    w = torch.randn(4, 3, 3, 3, device="cuda")
    torch.nn.functional.conv2d(x, w)
    torch.cuda.synchronize()
    return f"conv2d on GPU OK (cudnn {torch.backends.cudnn.version()})"


# --- opencv: the double-install and numpy-2 ABI risks ------------------------


def check_opencv_single_install() -> str:
    names = sorted(
        n
        for d in md.distributions()
        if (n := (d.metadata["Name"] or "")).lower().startswith("opencv")
    )
    if not names:
        raise Skip("no opencv distribution installed")
    if len(names) > 1:
        raise Exception(f"{len(names)} competing cv2 builds: {names}")
    return names[0]


def check_opencv_numpy_abi() -> str:
    """opencv-contrib-python 4.10.0.84 predates numpy 2; prove the bridge works."""
    import cv2
    import numpy as np

    img = np.zeros((16, 16, 3), np.uint8)
    img[4:8, 4:8] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    back = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert back.shape == img.shape
    flavor = "contrib" if hasattr(cv2, "xfeatures2d") else "base"
    return f"{cv2.__version__} ({flavor}), numpy roundtrip OK"


# --- third-party stacks ------------------------------------------------------


def check_turbovec() -> str:
    import turbovec

    return f"{version_of('turbovec')} (imported on py{sys.version_info.minor} via abi3)"


def check_numba() -> str:
    import numba
    from numba import njit

    @njit(cache=False)
    def _inc(x):
        return x + 1

    assert _inc(1) == 2
    return f"{numba.__version__}, njit OK"


def check_k2() -> str:
    import k2
    import torch

    rt = k2.RaggedTensor([[1, 2], [3]])
    if torch.cuda.is_available():
        rt = rt.to("cuda")
    built = getattr(getattr(k2, "version", None), "torch_version", "?")
    return f"{k2.__version__} (built for torch {built})"


def check_kaldifeat() -> str:
    import kaldifeat

    kaldifeat.Fbank(kaldifeat.FbankOptions())
    return f"{kaldifeat.__version__}"


def check_paddle() -> str:
    import paddle

    detail = paddle.__version__
    if tuple_of(detail) != tuple_of(PADDLE_EXPECTED):
        raise Warn(f"expected {PADDLE_EXPECTED}, got {detail}")
    return detail


def check_paddle_gpu() -> str:
    import paddle

    if paddle.device.cuda.device_count() == 0:
        raise Skip("paddle sees no GPU")
    paddle.set_device("gpu")
    x = paddle.rand([64, 64])
    float((x @ x).sum())
    return f"matmul on {paddle.device.cuda.get_device_name(0)}"


def check_paddleocr() -> str:
    from paddleocr import PaddleOCR  # noqa: F401 - not instantiated: downloads models

    return f"{version_of('paddleocr')} (class import only, no model download)"


def check_albumentations_numpy() -> str:
    """vietocr hard-pins albumentations==1.4.2, which predates numpy 2."""
    import albumentations as A
    import numpy as np

    pipe = A.Compose([A.Resize(8, 8)])
    out = pipe(image=np.zeros((16, 16, 3), np.uint8))["image"]
    assert out.shape == (8, 8, 3)
    return f"{version_of('albumentations')}, transform on numpy array OK"


def check_vietocr() -> str:
    from vietocr.tool.predictor import Predictor  # noqa: F401

    return f"{version_of('vietocr')}, predictor module imports"


def check_pillow() -> str:
    import PIL

    return f"{PIL.__version__} (vietocr pins 10.2.0; last cp312-only wheel)"


def check_soundfile() -> str:
    import soundfile

    return f"{soundfile.__version__} (libsndfile {soundfile.__libsndfile_version__})"


def check_sherpa_onnx() -> str:
    import sherpa_onnx

    return getattr(sherpa_onnx, "__version__", version_of("sherpa-onnx"))


def check_parquet_roundtrip() -> str:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    engines = []
    with tempfile.TemporaryDirectory() as d:
        for engine in ("pyarrow", "fastparquet"):
            if importlib.util.find_spec(engine) is None:
                continue
            p = Path(d) / f"{engine}.parquet"
            df.to_parquet(p, engine=engine)
            assert pd.read_parquet(p, engine=engine).equals(df)
            engines.append(f"{engine} {version_of(engine)}")
    if not engines:
        raise Skip("no parquet engine installed")
    return ", ".join(engines)


def check_transnetv2() -> str:
    if importlib.util.find_spec("transnetv2_pytorch") is None:
        raise Warn("transnetv2_pytorch not installed (vendored editable install)")
    import transnetv2_pytorch  # noqa: F401

    return "imports"


def check_kaggle() -> str:
    if importlib.util.find_spec("kaggle") is None:
        raise Skip("kaggle not installed")
    try:
        import kaggle  # noqa: F401
    except (OSError, ValueError) as e:
        raise Warn(f"installed but not configured: {e}") from None
    return version_of("kaggle")


def simple_imports(profile: str) -> list[tuple[str, str]]:
    common = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic_settings", "pydantic-settings"),
        ("huggingface_hub", "huggingface_hub"),
        ("ffmpeg", "ffmpeg-python"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("open_clip", "open_clip_torch"),
        ("timm", "timm"),
        ("transformers", "transformers"),
        ("sentencepiece", "sentencepiece"),
    ]
    if profile == "preprocessing":
        common += [("gdown", "gdown")]
    return common


# --- resolver health ---------------------------------------------------------


def check_metadata_consistency() -> str:
    """pip check, in-process: catches a silently downgraded numpy/pillow."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        raise Skip("packaging not installed") from None

    installed = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name:
            installed[name] = dist.version

    problems = []
    for dist in md.distributions():
        owner = dist.metadata["Name"] or "?"
        for raw in dist.requires or []:
            try:
                req = Requirement(raw)
                if req.marker is not None and not req.marker.evaluate():
                    continue
            except Exception:  # noqa: BLE001 - extras markers cannot be evaluated here
                continue
            have = installed.get(req.name.lower().replace("_", "-"))
            if have is None or not req.specifier:
                continue
            if not req.specifier.contains(have, prereleases=True):
                problems.append(f"{owner} requires {req.name}{req.specifier}, have {have}")

    if problems:
        raise Warn("; ".join(sorted(set(problems))[:6]))
    return f"{len(installed)} distributions, all requirements satisfied"


# --- main --------------------------------------------------------------------


def main() -> int:
    global VERBOSE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=("auto", "backend", "preprocessing"), default="auto")
    ap.add_argument("--strict", action="store_true", help="treat WARN as failure")
    ap.add_argument("-v", "--verbose", action="store_true", help="print tracebacks")
    args = ap.parse_args()
    VERBOSE = args.verbose

    profile = detect_profile() if args.profile == "auto" else args.profile
    print(f"aic26 dependency check -- profile: {profile}")

    section("environment")
    run("python version", check_python)
    run("virtualenv", check_venv)
    run("nvidia driver libs", check_driver_path)
    run("ffmpeg binary", check_ffmpeg)

    section("numpy / torch")
    run("numpy version", lambda: check_numpy(profile))
    run("numpy _core.scalar", check_numpy_core_scalar)
    run("torch version", check_torch)
    run("torchvision pairing", check_torchvision)
    run("weights_only checkpoint", check_weights_only_load)

    section("gpu")
    run("cuda device", check_cuda_available)
    run("device arch support", check_gpu_arch)
    run("cudnn conv2d", check_cudnn_conv)

    section("opencv")
    run("single cv2 install", check_opencv_single_install)
    run("cv2 <-> numpy abi", check_opencv_numpy_abi)

    section("packages")
    for mod, dist in simple_imports(profile):
        run(dist, lambda m=mod, d=dist: (__import__(m), version_of(d))[1])
    run("turbovec", check_turbovec)
    run("pillow", check_pillow)
    run("parquet roundtrip", check_parquet_roundtrip, optional=True)

    if profile == "preprocessing":
        section("preprocessing extras")
        run("numba", check_numba)
        run("kaggle", check_kaggle, optional=True)
        run("soundfile", check_soundfile)
        run("sherpa-onnx", check_sherpa_onnx)
        run("k2", check_k2)
        run("kaldifeat", check_kaldifeat)
        run("transnetv2_pytorch", check_transnetv2, optional=True)

        section("ocr stack")
        run("paddlepaddle", check_paddle)
        run("paddlepaddle gpu", check_paddle_gpu)
        run("paddleocr", check_paddleocr)
        run("albumentations <-> numpy", check_albumentations_numpy)
        run("vietocr", check_vietocr)

    section("resolver health")
    run("installed metadata", check_metadata_consistency)

    counts = {s: sum(1 for st, _, _ in ROWS if st == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print(
        f"\n{counts['PASS']} passed, {counts['WARN']} warned, "
        f"{counts['FAIL']} failed, {counts['SKIP']} skipped"
    )
    for status, name, detail in ROWS:
        if status == "FAIL" or (status == "WARN" and args.strict):
            print(f"  -> {status} {name}: {detail}")

    return 1 if counts["FAIL"] or (args.strict and counts["WARN"]) else 0


if __name__ == "__main__":
    sys.exit(main())