from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import torch

from keyframe import SearcherRegistry
from config import CKPT_PATHS, KF_INDEX_PATHS, LOG_DB_PATH, METADATA_DB_PATH
from routers import keyframe, query, video
from routers.logs import LogDatabase


def build_registry():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    registry = SearcherRegistry(KF_INDEX_PATHS, METADATA_DB_PATH, CKPT_PATHS, device=device)

    available = registry.available()
    unavailable = [m for m in KF_INDEX_PATHS if m not in available]
    if unavailable:
        print(f"warning: search disabled for {', '.join(unavailable)} (index or checkpoint missing)")
    if not available:
        print("warning: no search index found, /query will return 503")
    else:
        print(f"Search models available: {', '.join(available)} (loaded on first use, {device})")
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to databases...")
    with LogDatabase(LOG_DB_PATH) as db:
        db.init_schema()

    app.state.search_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search")
    app.state.searchers = app.state.search_pool.submit(build_registry).result()

    yield

    print("Closing connections...")
    app.state.search_pool.submit(app.state.searchers.close).result()
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
