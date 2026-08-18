from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

METADATA_DB_PATH = DATA_PATH / "metadata.db"
KEYFRAMES_DB_PATH = DATA_PATH / "keyframes"
VIDEOS_DB_PATH = DATA_PATH / "videos"
KF_INDEX_PATH = DATA_PATH / "index"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"

DEFAULT_MODEL = "siglip2"
KF_INDEX_PATHS = {
    "c2lip": KF_INDEX_PATH / "c2lip-keyframes.tvim",
    "siglip2": KF_INDEX_PATH / "siglip2-keyframes.tvim",
}
CKPT_PATHS = {
    "c2lip": CKPT_PATH,
    "siglip2": None,
}

LOG_DB_PATH = "logs.db"