from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers import keyframe, video

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to databases...")
    yield
    print("Closing connections...")

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

@app.get("/health")
async def health_check():
    return {"status": "ok"}
