from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from config import CKPT_PATH, KF_INDEX_PATH, LOG_DB_PATH, METADATA_DB_PATH
from routers import keyframe, query, video
from routers.logs import LogDatabase


def build_searcher():
    missing = [p for p in (KF_INDEX_PATH, CKPT_PATH) if not p.exists()]
    if missing:
        print(f"warning: search disabled, missing {', '.join(str(p) for p in missing)}")
        return None

    import torch
    from keyframe import KeyframeSearcher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading keyframe searcher on {device}...")
    return KeyframeSearcher(KF_INDEX_PATH, METADATA_DB_PATH, CKPT_PATH, device=device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to databases...")
    with LogDatabase(LOG_DB_PATH) as db:
        db.init_schema()

    app.state.search_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search")
    app.state.searcher = app.state.search_pool.submit(build_searcher).result()

    yield

    print("Closing connections...")
    if app.state.searcher is not None:
        app.state.search_pool.submit(app.state.searcher.close).result()
    app.state.search_pool.shutdown()

app = FastAPI(
    title="AIC26 - Backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

app.include_router(keyframe.router, prefix="/keyframe")
app.include_router(video.router, prefix="/video")
app.include_router(query.router, prefix="/query")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
