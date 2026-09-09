from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

METADATA_DB_PATH = DATA_PATH / "metadata.db"
KEYFRAMES_DB_PATH = DATA_PATH / "keyframes"
VIDEOS_DB_PATH = DATA_PATH / "videos"
SCENES_DB_PATH = DATA_PATH / "staging"
TRANSCRIPTS_DB_PATH = DATA_PATH / "transcripts"
OCR_DB_PATH = DATA_PATH / "ocr"
KF_INDEX_PATH = DATA_PATH / "index"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"


DEFAULT_MODEL = "siglip2"
KF_INDEX_PATHS = {
    "siglip": KF_INDEX_PATH / "siglip-keyframes.tvim",
    "siglip2": KF_INDEX_PATH / "siglip2-keyframes.tvim",
    "pe": KF_INDEX_PATH / "pe-keyframes.tvim"
}

SHARD_DIR = DATA_PATH / "embeddings"
SHARD_DIRS = {model: SHARD_DIR / model for model in KF_INDEX_PATHS}

CKPT_PATHS = {
    "siglip": CKPT_PATH,
    "siglip2": None,
    "pe": DATA_PATH / "checkpoints" / "PE-Core-L14-336_npc_xac_epoch_2.pt"
}

TRANSCRIPT_INDEX_PATHS = {
    "gte": KF_INDEX_PATH / "gte-transcripts.tvim",
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

DETECT_DIR = DATA_PATH / "detections"
DEFAULT_DETECT_MODEL = "owlv2-base"
DETECT_REPOS = {
    "owlv2-base": "google/owlv2-base-patch16-ensemble",
    "owlv2-large": "google/owlv2-large-patch14-ensemble",
}
DETECT_SHARD_DIRS = {model: DETECT_DIR / model for model in DETECT_REPOS}
DETECT_PROMPT = "a photo of a {}"
