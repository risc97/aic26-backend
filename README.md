# [HCMC AIC 2026] Backend

Backend for the TRISC video search engine at the HCMC AI Challenge 2026

## Overview

The system is split into three repositories:

| Repository | Role |
| --- | --- |
| [aic26-preprocessing](https://github.com/risc97/aic26-preprocessing) | Process raw videos into keyframes, embeddings, OCR, transcripts and search indices |
| **[aic26-backend](https://github.com/risc97/aic26-backend)** | FastAPI server to searches indices and serves keyframes and videos |
| [aic26-frontend](https://github.com/risc97/aic26-frontend) | UI web app for searching, reviewing and submitting results |


Organize these repositories like this structure:

```text
aic26/
├── data/                  # input videos + preprocessing output
├── aic26-preprocessing/
├── aic26-backend/
└── aic26-frontend/
```
## Tech stack

- **Server:** [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **Vector index:** [turbovec](https://pypi.org/project/turbovec/)
- **Metadata:** SQLite (`metadata.db`)
- **Models:** C2LIP, SigLIP2, PE, DINOv3, GTE, OWLv2

## Project structure

```text
aic26-backend/
├── main.py              # App entry, router registration
├── config.py            # Data paths, index paths, default models
├── schemas.py           # Request / response models
├── fill_db.py           # Build metadata.db from preprocessing output
├── routers/             # keyframe, video, query, similar, logs
├── models/              # Query encoders
├── db/                  # SQLite access
├── temporal/            # Multi-event (temporal) search
├── detect/              # Object-count search
├── transcript/          # Transcript search (semantic + exact)
├── ocr/                 # OCR text search
└── media/               # Read videos, scenes, keyframes from disk
```

## Usage

### Prerequisites
- **Python** 3.10+, tested on 3.12
- **CUDA GPU** (recommended)
- Output from [aic26-preprocessing](https://github.com/risc97/aic26-preprocessing) in `../data/`

### Installation
```bash
pwd # Make sure you are in aic26/
git clone https://github.com/risc97/aic26-backend.git
cd aic26-backend
```

- **NixOS**: run `nix-shell`
- **Other**: create a venv, then `pip install -r requirements.txt`

Build the metadata database:
```bash
python fill_db.py
```

### Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

### Configuration

All paths and defaults live in [config.py](./config.py):
- **`DATA_PATH`**: data root, `../data` by default
- **`DEFAULT_MODEL`**: text-image model for search (`siglip2`)
- **`KF_INDEX_PATHS`**, **`VISUAL_INDEX_PATHS`**, **`TRANSCRIPT_INDEX_PATHS`**: `.tvim` index files
- **`CKPT_PATHS`**: C2LIP and fine-tuned PE checkpoints

---

## API

| Method | Path | Usage |
| --- | --- | --- |
| POST | `/query/keyframe` | Text → keyframes |
| POST | `/query/transcript/semantic` | Text → transcript segments (GTE) |
| POST | `/query/transcript/exact` | Exact / fuzzy match in transcripts |
| POST | `/query/ocr` | Exact / fuzzy match in on-screen text |
| POST | `/query/temporal` | Sequence of text events → Sequence of keyframes |
| POST | `/query/detect` | Object names + counts → keyframes |
| POST | `/query/temporal/detect` | Sequence of object events → videos |
| GET | `/similar/semantic/{video_id}/{keyframe_id}` | Semantically similar keyframes |
| POST | `/similar/semantic/upload` | Same, from an uploaded image |
| GET | `/similar/visual/{video_id}/{keyframe_id}` | Visually similar keyframes (DINOv3) |
| POST | `/similar/visual/upload` | Same, from an uploaded image |
| GET | `/keyframe/{video_id}/keyframes` | All keyframes of a video |
| GET, HEAD | `/keyframe/{video_id}/{keyframe_id}` | Keyframe JPEG |
| GET, HEAD | `/video/{video_id}` | Video file |
| GET | `/logs` | Recent requests |
| GET | `/logs/{request_id}` | One logged request |
| GET | `/health` | `{"status": "ok"}` |

Example:
```bash
curl -X POST localhost:8000/query/keyframe \
  -H "Content-Type: application/json" \
  -d '{"query": "a man riding a bicycle", "limit": 5, "model": "siglip2"}'
```