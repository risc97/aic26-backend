from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

METADATA_DB_PATH = DATA_PATH / "metadata.db"
KEYFRAMES_DB_PATH = DATA_PATH / "keyframes"
VIDEOS_DB_PATH = DATA_PATH / "videos"
SCENES_DB_PATH = DATA_PATH / "staging"
KF_INDEX_PATH = DATA_PATH / "index"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"

DEFAULT_MODEL = "c2lip"
KF_INDEX_PATHS = {
    "c2lip": KF_INDEX_PATH / "c2lip-keyframes.tvim",
    "siglip2": KF_INDEX_PATH / "siglip2-keyframes.tvim",
}
CKPT_PATHS = {
    "c2lip": CKPT_PATH,
    "siglip2": None,
}

LOG_DB_PATH = "logs.db"
KEYFRAME_MODE = "mid"

def stored_path(path: str | Path) -> str:
    """Convert a path to be relative to DATA_PATH.
    For example:
    DATA_PATH = Path('/workspace/data')
    stored_path(/workspace/data/keyframes/001.jpg) = keyframes/001.jpg
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(DATA_PATH))
    except ValueError:
        return str(resolved)