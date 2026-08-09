from pathlib import Path
import sys

DATA_PATH = Path("../aic/data")
METADATA_DB_PATH = DATA_PATH / "metadata.db" 
KEYFRAMES_DB_PATH = DATA_PATH / "keyframes"
VIDEOS_DB_PATH = DATA_PATH / "videos"
KF_INDEX_PATH = DATA_PATH / "index" / "keyframes.tvim"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"

LOG_DB_PATH = "logs.db"