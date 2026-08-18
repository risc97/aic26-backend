from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime

from config import DEFAULT_MODEL
ModelName = Literal["siglip", "siglip2"]

class Item(BaseModel):
    keyframe_id: str
    video_id: str
    timestamp_ms: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="Truy vấn đoạn video cần tìm")
    limit: int = Field(100, description="Số lượng kết quả trả về")
    mode: str = Field("keyframe", description="Chế độ search")
    model: ModelName = Field(DEFAULT_MODEL, description="Model dùng để search")

class QueryResponse(BaseModel):
    results: List[Item] = Field(default_factory=list)
    total: int = 0

class Video(BaseModel):
    video_id: str
    duration_ms: int
    fps: float
    keyframe_count: int | None = None

class Keyframe(BaseModel):
    keyframe_id: int
    video_id: str
    frame_idx: int
    timestamp_ms: int
    segment_id: int
    image_url: str
    video_url: str

class SearchHit(Keyframe):
    rank: int
    score: float

class LogEntry(QueryRequest, QueryResponse):
    request_id: str
    timestamp: datetime | None