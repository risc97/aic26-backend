# AIC26 Backend

## Overview
This project include multiple repositories:
- [aic26-preprcessing](https://github.com/risc97/aic26-preprocessing): Turn raw videos into keyframes, embeddings, and a search index
- [aic26-backend](https://github.com/risc97/aic26-backend): FastAPI server that searches the index and serves keyframes and videos

## Project structure

You may organize the project structure like this:
```
aic26/
├── data/
│   ├── videos/            # L21_V001.webm ...
│   ├── media-info/        # L21_V001.json ...
│   ├── staging/           # L21_V001.scenes.txt ... (from TransNetV2)
│   ├── keyframes/         # L21_V001/0001.jpg ...
│   ├── embeddings/        # embedding shards
│   ├── index/             # embedding index
│   ├── checkpoints/       # c2lip.pt
│   └── metadata.db
├── aic26-preprocessing/
└── aic26-backend/
```

## Install

```bash
pwd # Make sure you are in aic26/
git clone https://github.com/risc97/aic26-backend.git
cd aic26-backend
```

- **NixOS**: run `nix-shell`. It creates `.venv` and installs `requirements.txt`.
- **Other**: make a Python 3.10 venv, then `pip install -r requirements.txt`.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive docs.

## API

| Method | Path | What it does |
| --- | --- | --- |
| POST | `/query` | search keyframes by text |
| GET, HEAD | `/keyframe/{video_id}/{keyframe_id}` | the keyframe JPEG |
| GET, HEAD | `/video/{video_id}` | the video file |
| GET | `/health` | returns `{"status": "ok"}` |

Search:

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "a man riding a bicycle", "limit": 5}'
```

```json
{
  "results": [
    { "video_id": "L21_V001", "keyframe_id": "0042" },
    { "video_id": "L21_V007", "keyframe_id": "0113" }
  ],
  "total": 2
}
```