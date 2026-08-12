from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

METADATA_DB_PATH = DATA_PATH / "metadata.db"
KEYFRAMES_DB_PATH = DATA_PATH / "keyframes"
VIDEOS_DB_PATH = DATA_PATH / "videos"
KF_INDEX_PATH = DATA_PATH / "index" / "keyframes.tvim"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"

LOG_DB_PATH = "logs.db"