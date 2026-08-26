from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import torch

from registry import SearcherRegistry, TableSearcherRegistry
from keyframe import KeyframeSearcher
from ocr import OcrSearcher
from transcript import TranscriptExactSearcher, TranscriptSemanticSearcher
from config import (
    CKPT_PATHS, KF_INDEX_PATHS, TRANSCRIPT_INDEX_PATHS, LOG_DB_PATH, METADATA_DB_PATH,
)
from routers import keyframe, query, video, similar, logs
from routers.logs import LogDatabase


def build_keyframe_registry(device: str):
    registry = SearcherRegistry(
        KF_INDEX_PATHS, METADATA_DB_PATH, KeyframeSearcher, CKPT_PATHS, device=device,
    )

    available = registry.available()
    unavailable = [m for m in KF_INDEX_PATHS if m not in available]
    if unavailable:
        print(f"warning: keyframe search disabled for {', '.join(unavailable)} (index or checkpoint missing)")
    if not available:
        print("warning: no keyframe search index found, mode='keyframe' will return 503")
    else:
        print(f"Keyframe search models available: {', '.join(available)} (loaded on first use, {device})")
    return registry


def build_transcript_registry(device: str):
    registry = SearcherRegistry(
        TRANSCRIPT_INDEX_PATHS, METADATA_DB_PATH, TranscriptSemanticSearcher, device=device,
    )

    available = registry.available()
    if not available:
        print("warning: no transcript search index found, mode='transcript_semantic' will return 503")
    else:
        print(f"Transcript search models available: {', '.join(available)} (loaded on first use, {device})")
    return registry


def build_table_registry(mode: str, table: str, searcher_cls):
    """Registry for an FTS5 mode backed by one metadata table."""
    registry = TableSearcherRegistry(METADATA_DB_PATH, table, searcher_cls)

    count = registry.count()
    if not count:
        print(f"warning: table '{table}' is empty, mode='{mode}' will return 503 "
              f"(run fill_db.py)")
    else:
        print(f"{mode} search available: {count} '{table}' rows (loaded on first use)")
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to databases...")
    with LogDatabase(LOG_DB_PATH) as db:
        db.init_schema()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.search_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search")
    app.state.searchers = {
        "keyframe": app.state.search_pool.submit(build_keyframe_registry, device).result(),
        "transcript_semantic": app.state.search_pool.submit(build_transcript_registry, device).result(),
        "transcript_exact": app.state.search_pool.submit(
            build_table_registry, "transcript_exact", "transcripts",
            TranscriptExactSearcher).result(),
        "ocr_exact": app.state.search_pool.submit(
            build_table_registry, "ocr_exact", "ocr", OcrSearcher).result(),
    }

    yield

    print("Closing connections...")
    for registry in app.state.searchers.values():
        app.state.search_pool.submit(registry.close).result()
    app.state.search_pool.shutdown()

app = FastAPI(
    title="AIC26 - Backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

app.include_router(keyframe.router, prefix="/keyframe")
app.include_router(video.router, prefix="/video")
app.include_router(query.router, prefix="/query")
app.include_router(similar.router, prefix="/similar")
app.include_router(logs.router, prefix="/logs")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
